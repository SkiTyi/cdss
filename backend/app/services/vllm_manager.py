"""
vllm subprocess manager for local LLM assistants.

Lifecycle:
  stopped → starting → running   (healthy /v1/models response)
                    ↘ failed     (process died OR /v1/models never came up)

We only inject the bare-minimum CLI args that the manager needs to track
the server (model path, served-model-name, host, port, optional LoRA, and
auto --tensor-parallel-size for multi-GPU). Anything else — including
log/sampling/scheduling tweaks like --disable-log-requests,
--gpu-memory-utilization, --dtype, etc — is the user's responsibility via
`extra_vllm_args`. This avoids breakage when vllm renames or removes flags
between versions.
"""
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from ..config import settings
from ._subprocess_env import build_subprocess_env


# Port range we'll allocate for vllm servers. Avoids 8000 (FastAPI default)
# and leaves room for ~99 concurrent local assistants.
_PORT_MIN = 8011
_PORT_MAX = 8099

# Where to drop vllm stdout/stderr logs.
_LOG_DIR = Path(settings.training_runs_dir).parent / "vllm_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _find_free_port() -> int:
    for port in range(_PORT_MIN, _PORT_MAX + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无可用端口 ({_PORT_MIN}-{_PORT_MAX} 全被占用)")


def _ping_vllm(port: int, model_name: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Probe /v1/models. Return (ready, reason).

    Match strategy is intentionally lax — different vllm versions report the
    served model name with different casing / quoting / path prefixes, and we
    don't want a one-character typo to leave the assistant stuck in 'starting'
    forever even though the server is healthy. We accept:
      * exact equality                              → 'ok'
      * case-insensitive / basename / substring     → 'ok-loose:<actual>'
      * server up with ≥1 model but no name match   → 'ok-any:returned=[...]'
        (treat as ready; caller surfaces this as a warning so user can fix
        the model_name field, but the assistant becomes usable)
    Failures (HTTP error, exception, no models) return False with a short
    reason string suitable for logging.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"http://127.0.0.1:{port}/v1/models")
        if r.status_code != 200:
            return False, f"http_{r.status_code}"
        data = r.json().get("data", [])
        if not data:
            return False, "no_data"
        ids = [str(m.get("id", "")) for m in data]
        mn = (model_name or "").strip()
        # 1. exact
        if mn in ids:
            return True, "ok"
        # 2. case-insensitive / basename / substring
        mn_lc = mn.lower()
        for mid in ids:
            mid_lc = mid.lower()
            mid_base = mid.rstrip("/").rsplit("/", 1)[-1].lower()
            if mid_lc == mn_lc or mid_base == mn_lc or mn_lc in mid_lc or mid_lc in mn_lc:
                return True, f"ok-loose:{mid}"
        # 3. server up but name doesn't match at all — still consider ready
        return True, f"ok-any:returned={ids},expected={mn}"
    except Exception as e:
        return False, f"err:{type(e).__name__}:{e}"


def _resolve_vllm_executable() -> list[str]:
    """Find the vllm CLI that lives in the same Python env as the backend."""
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return [str(candidate)]
    # Fallback: invoke the OpenAI-compat server module directly.
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


def _has_flag(args: list[str], *flag_names: str) -> bool:
    """True if any of `flag_names` already appears in user-supplied args."""
    for a in args or []:
        a_low = str(a).lstrip("-").split("=")[0]
        if a_low in [f.lstrip("-") for f in flag_names]:
            return True
    return False


class VLLMManager:
    """Singleton tracking running vllm subprocesses keyed by assistant_id."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._procs = {}              # assistant_id -> Popen
            cls._instance._monitors = {}           # assistant_id -> Thread
            cls._instance._allocated_ports = set() # ports we've handed out (avoid races)
            cls._instance._port_lock = threading.Lock()
        return cls._instance

    # ── public API ─────────────────────────────────────────────────────────

    def _allocate_port(self) -> int:
        """Atomic port allocation. Holds a lock so two concurrent start()
        calls can't both grab the same port (the OS-level test-bind in
        _find_free_port has a window before vllm itself binds, during which
        a second caller could mistakenly think the port is free).
        """
        with self._port_lock:
            for port in range(_PORT_MIN, _PORT_MAX + 1):
                if port in self._allocated_ports:
                    continue
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("127.0.0.1", port))
                        self._allocated_ports.add(port)
                        return port
                    except OSError:
                        continue
            raise RuntimeError(
                f"无可用端口 ({_PORT_MIN}-{_PORT_MAX} 全被占用)"
            )

    def _release_port(self, port: int):
        if port is None:
            return
        with self._port_lock:
            self._allocated_ports.discard(port)

    def start(self, assistant, db_factory) -> int:
        """Spawn vllm serve. Returns assigned port. Raises on immediate failure."""
        if assistant.id in self._procs and self._procs[assistant.id].poll() is None:
            raise RuntimeError("助手已在运行中")
        if not assistant.model_path:
            raise RuntimeError("本地助手未指定 model_path")
        served_name = (assistant.model_name or "").strip()
        if not served_name:
            raise RuntimeError("本地助手未指定 model_name (即 vllm 的 served-model-name)")

        port = self._allocate_port()
        log_path = str(_LOG_DIR / f"assistant_{assistant.id}.log")

        # Essential, manager-controlled flags only. Everything else is in extra.
        base = _resolve_vllm_executable()
        # `vllm` CLI takes "serve" subcommand + positional model; the
        # python -m fallback uses --model.
        if base[0].endswith("vllm"):
            cmd = base + ["serve", assistant.model_path]
        else:
            cmd = base + ["--model", assistant.model_path]
        cmd += [
            "--served-model-name", served_name,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        # ── GPU placement ──────────────────────────────────────────────────
        # gpu_ids: None=auto (don't touch CUDA_VISIBLE_DEVICES),
        #          [n]=single, [n,m,...]=multi (auto-add tensor-parallel)
        gpu_ids = assistant.gpu_ids
        env = build_subprocess_env({"PYTHONUNBUFFERED": "1"})
        is_multi_gpu = isinstance(gpu_ids, list) and len(gpu_ids) > 1
        if isinstance(gpu_ids, list) and gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # Auto-inject --tensor-parallel-size for multi-GPU unless the user
            # already specified it themselves in extra_vllm_args.
            if is_multi_gpu and not _has_flag(
                    assistant.extra_vllm_args or [],
                    "--tensor-parallel-size", "-tp"):
                cmd += ["--tensor-parallel-size", str(len(gpu_ids))]

        # Multi-GPU vllm spawns one worker per GPU via Python multiprocessing.
        # vllm's default `fork` start method deadlocks when the parent has
        # any CUDA state — symptom: log stops at "Started engine process"
        # and never produces another line until our 600s monitor times out.
        # Forcing `spawn` is the canonical fix (vllm docs + GH issues).
        if is_multi_gpu:
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # ── max-model-len (still a first-class field on the form) ─────────
        if assistant.max_model_len and not _has_flag(
                assistant.extra_vllm_args or [], "--max-model-len"):
            cmd += ["--max-model-len", str(assistant.max_model_len)]

        # ── LoRA adapter ───────────────────────────────────────────────────
        if assistant.lora_adapter_path and not _has_flag(
                assistant.extra_vllm_args or [], "--enable-lora"):
            cmd += [
                "--enable-lora",
                "--lora-modules",
                f"{served_name}-lora={assistant.lora_adapter_path}",
            ]

        # ── user-supplied extras (verbatim) ────────────────────────────────
        for extra in (assistant.extra_vllm_args or []):
            if extra:
                cmd.append(str(extra))

        log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.write(f"$ CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError as e:
            log_f.close()
            raise RuntimeError(f"无法启动 vllm: {e}")

        self._procs[assistant.id] = proc

        # Persist starting state immediately so the UI sees feedback while
        # vllm spends its first ~30s loading weights.
        self._update_db(db_factory, assistant.id, {
            "status": "starting",
            "process_pid": proc.pid,
            "port": port,
            "base_url": f"http://127.0.0.1:{port}/v1",
            "log_file": log_path,
            "error_message": None,
        })

        # Background thread: poll /v1/models until healthy, or until the
        # process dies, then update DB status accordingly. Multi-GPU startup
        # is slower (per-GPU CUDA graph capture + NCCL handshake), so we
        # scale the wait budget by GPU count.
        n_gpus = len(gpu_ids) if isinstance(gpu_ids, list) and gpu_ids else 1
        max_wait = 600 + 300 * max(0, n_gpus - 1)   # +5min per extra GPU
        t = threading.Thread(
            target=self._monitor_until_ready,
            args=(assistant.id, port, served_name,
                  proc, log_path, db_factory, max_wait),
            daemon=True,
        )
        t.start()
        self._monitors[assistant.id] = t

        return port

    def stop(self, assistant_id: int, db_factory):
        proc = self._procs.get(assistant_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.pop(assistant_id, None)

        # Release the port back into the pool. Read it from DB since the
        # in-memory mapping doesn't track per-assistant port directly.
        from ..models.models import LLMAssistant
        db = db_factory()
        try:
            row = db.query(LLMAssistant).filter_by(id=assistant_id).first()
            if row and row.port:
                self._release_port(row.port)
        finally:
            db.close()

        self._update_db(db_factory, assistant_id, {
            "status": "stopped",
            "process_pid": None,
            "port": None,
            "base_url": None,
        })

    def is_alive(self, assistant_id: int) -> bool:
        proc = self._procs.get(assistant_id)
        return proc is not None and proc.poll() is None

    def liveness_check(self, assistant, db_factory):
        """Reconcile DB status with subprocess reality (call on demand)."""
        if assistant.type != "local":
            return
        if assistant.status in ("running", "starting"):
            proc = self._procs.get(assistant.id)
            if not proc or proc.poll() is not None:
                # Free the port we'd allocated; vllm is gone.
                self._release_port(assistant.port)
                self._update_db(db_factory, assistant.id, {
                    "status": "failed" if assistant.status == "starting" else "stopped",
                    "process_pid": None,
                    "port": None,
                    "base_url": None,
                    "error_message": "进程已退出（可能被外部 kill）",
                })

    # ── private ────────────────────────────────────────────────────────────

    def _monitor_until_ready(self, assistant_id, port, served_name,
                             proc, log_path, db_factory,
                             max_wait_sec: int = 600):
        """Poll /v1/models. Mark running once healthy; failed if proc dies.

        Every ping result (success or failure) is appended to the vllm log
        file with a [manager] prefix so the user can `tail -f` the log and
        diagnose why startup is taking too long.
        """
        def _log(line: str):
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[manager] {line}\n")
            except Exception:
                pass

        start = time.time()
        last_reason = ""
        poll_count = 0
        while True:
            if proc.poll() is not None:
                # Process died before becoming healthy → read tail of log for diag.
                tail = ""
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        tail = "".join(f.readlines()[-40:])
                except Exception:
                    pass
                self._release_port(port)
                self._update_db(db_factory, assistant_id, {
                    "status": "failed",
                    "process_pid": None,
                    "port": None,
                    "base_url": None,
                    "error_message": f"vllm 进程退出 (code={proc.returncode}); 末尾日志:\n{tail[-2000:]}",
                })
                self._procs.pop(assistant_id, None)
                return

            ready, reason = _ping_vllm(port, served_name)
            if ready:
                _log(f"ready ({reason}) after {int(time.time()-start)}s")
                fields = {"status": "running"}
                # Loose / mismatched match — still ready, but flag for visibility
                if reason != "ok":
                    fields["error_message"] = (
                        f"健康检查通过但模型名匹配宽松：{reason} — "
                        f"如下游调用失败，请检查助手的 model_name 字段是否与 "
                        f"vllm 的 served-model-name 完全一致"
                    )
                else:
                    fields["error_message"] = None
                self._update_db(db_factory, assistant_id, fields)
                return

            last_reason = reason
            poll_count += 1
            # Heartbeat every ~20s so users can see the manager is still trying
            if poll_count % 10 == 0:
                _log(f"still waiting after {poll_count*2}s — last ping: {reason}")

            if time.time() - start > max_wait_sec:
                _log(f"timed out after {max_wait_sec}s — last ping: {last_reason}")
                # Don't kill the process — vllm may still be loading. Just stop monitoring.
                self._update_db(db_factory, assistant_id, {
                    "status": "starting",
                    "error_message": (
                        f"启动超过 {max_wait_sec}s 仍未就绪；最后健康检查：{last_reason}；"
                        f"请查看日志 {log_path}"
                    ),
                })
                return

            time.sleep(2.0)

    def _update_db(self, db_factory, assistant_id: int, fields: dict):
        from ..models.models import LLMAssistant
        db = db_factory()
        try:
            row = db.query(LLMAssistant).filter_by(id=assistant_id).first()
            if not row:
                return
            for k, v in fields.items():
                setattr(row, k, v)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


vllm_manager = VLLMManager()
