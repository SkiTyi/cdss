"""
Training process manager.

`config["gpu_ids"]` semantics:
  - None / missing → auto: do NOT override CUDA_VISIBLE_DEVICES,
                    let the runtime see whatever the parent shell exposed.
  - []             → force CPU: CUDA_VISIBLE_DEVICES="" (hides all GPUs).
  - [0]            → single GPU: python train_script.py with CUDA_VISIBLE_DEVICES=0.
  - [0, 1, ...]    → multi-GPU DDP: torchrun --nproc_per_node=N
                    with CUDA_VISIBLE_DEVICES=0,1,...

Metrics are read from a shared metrics_file written by rank-0; this works
identically for single-GPU and multi-GPU runs.  The process stdout/stderr is
captured separately for error / progress messages.
"""
import json
import os
import socket
import sys
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from ._subprocess_env import build_subprocess_env


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TrainingManager:
    """Singleton that tracks and manages training subprocesses."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processes = {}  # exp_id -> Popen
        return cls._instance

    # ------------------------------------------------------------------ public

    def start(self, exp_id: int, config: dict, db_factory) -> int:
        """
        Launch the training subprocess and begin monitoring.
        Returns the PID of the launched process.
        """
        from ..config import settings

        gpu_ids = config.get("gpu_ids")            # None | [] | [int,...]
        n_gpus = len(gpu_ids) if gpu_ids else 0
        metrics_file: str = config.get("metrics_file", "")

        run_dir = Path(settings.training_runs_dir) / str(exp_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        config_file = str(run_dir / "config.json")
        log_file = str(run_dir / "train.log")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # ── build environment ──────────────────────────────────────────────
        env = build_subprocess_env({"PYTHONUNBUFFERED": "1"})
        if gpu_ids is not None:
            # Explicit choice — either force-CPU ("") or pin to a subset.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        # else: gpu_ids is None → auto, leave CUDA_VISIBLE_DEVICES unchanged.

        # ── build command ──────────────────────────────────────────────────
        script_path = Path(__file__).parent / "train_script.py"
        if n_gpus > 1:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(n_gpus),
                "--master_port", str(_find_free_port()),
                "--rdzv_backend", "c10d",
                str(script_path), "--config", config_file,
            ]
        else:
            cmd = [sys.executable, str(script_path), "--config", config_file]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._processes[exp_id] = proc

        thread = threading.Thread(
            target=self._monitor,
            args=(exp_id, proc, metrics_file, log_file, db_factory),
            daemon=True,
        )
        thread.start()

        return proc.pid

    def stop(self, exp_id: int):
        proc = self._processes.get(exp_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._processes.pop(exp_id, None)

    def is_running(self, exp_id: int) -> bool:
        proc = self._processes.get(exp_id)
        return proc is not None and proc.poll() is None

    # ------------------------------------------------------------------ private

    def _monitor(self, exp_id: int, proc: subprocess.Popen,
                 metrics_file: str, log_file: str, db_factory):
        """
        Monitor training in the background.

        Two parallel sub-tasks:
          1. Read process stdout → log DB (captures torchrun/error messages)
          2. Tail metrics_file   → metrics DB + log DB (works for any #GPUs)
        """
        # Sub-thread: read stdout/stderr of the process
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(exp_id, proc, log_file, db_factory),
            daemon=True,
        )
        stdout_thread.start()

        # Main thread: tail the metrics file
        self._tail_metrics_file(exp_id, proc, metrics_file, db_factory)

        # Wait for process to finish and update experiment status
        return_code = proc.wait()
        stdout_thread.join(timeout=10)
        self._processes.pop(exp_id, None)

        db = db_factory()
        try:
            from ..models.models import TrainingExperiment
            exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
            if exp and exp.status == "running":
                exp.status = "completed" if return_code == 0 else "failed"
                exp.completed_at = datetime.utcnow()
                if return_code != 0:
                    exp.error_message = f"进程退出码 {return_code}"
                db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    def _read_stdout(self, exp_id: int, proc: subprocess.Popen,
                     log_file: str, db_factory):
        """Capture process stdout; store non-JSON lines as info/error logs."""
        from ..models.models import TrainingLog

        with open(log_file, "w", encoding="utf-8") as lf:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lf.write(line + "\n")
                lf.flush()

                # Skip blank lines and lines that are already in metrics_file
                # (rank-0 emits JSON to both stdout and the file; we deduplicate
                # by only storing non-JSON lines here)
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    # Valid JSON → already handled by _tail_metrics_file; skip.
                    continue
                except ValueError:
                    pass

                # Non-JSON stdout line (torchrun startup, Python tracebacks, etc.)
                level = "error" if "error" in line.lower() or "traceback" in line.lower() else "info"
                db = db_factory()
                try:
                    db.add(TrainingLog(experiment_id=exp_id, level=level, message=line))
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()

    def _tail_metrics_file(self, exp_id: int, proc: subprocess.Popen,
                           metrics_file: str, db_factory):
        """Tail the metrics JSON-lines file written by rank-0 of the training script."""
        from ..models.models import TrainingExperiment, TrainingMetric, TrainingLog

        if not metrics_file:
            return  # Nothing to tail

        # Wait until file is created (up to 10 minutes)
        for _ in range(600):
            if os.path.exists(metrics_file):
                break
            if proc.poll() is not None:
                return  # Process died before creating the file
            time.sleep(1)

        if not os.path.exists(metrics_file):
            return

        with open(metrics_file, "r", encoding="utf-8") as mf:
            while True:
                line = mf.readline()
                if not line:
                    if proc.poll() is not None:
                        break  # Process finished, no more data
                    time.sleep(0.3)
                    continue

                line = line.rstrip("\n")
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except ValueError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "metrics":
                    db = db_factory()
                    try:
                        extra = {
                            k: v for k, v in data.items()
                            if k not in ("type", "step", "epoch",
                                         "train_loss", "eval_loss", "learning_rate")
                        }
                        db.add(TrainingMetric(
                            experiment_id=exp_id,
                            step=data.get("step", 0),
                            epoch=data.get("epoch"),
                            train_loss=data.get("train_loss"),
                            eval_loss=data.get("eval_loss"),
                            learning_rate=data.get("learning_rate"),
                            extra_metrics=extra,
                        ))
                        exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                        if exp:
                            if data.get("eval_loss") is not None:
                                if (exp.best_eval_loss is None
                                        or data["eval_loss"] < exp.best_eval_loss):
                                    exp.best_eval_loss = data["eval_loss"]
                            if data.get("step"):
                                exp.current_step = data["step"]
                        db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "info":
                    db = db_factory()
                    try:
                        exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                        if exp and data.get("total_steps"):
                            exp.total_steps = data["total_steps"]
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "final_eval":
                    # Persist the final validation metrics into experiment.config
                    db = db_factory()
                    try:
                        exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                        if exp:
                            cfg = dict(exp.config or {})
                            cfg["final_eval"] = {
                                "eval_loss": data.get("eval_loss"),
                                "token_accuracy": data.get("token_accuracy"),
                                "eval_samples": data.get("eval_samples"),
                                "eval_tokens": data.get("eval_tokens"),
                            }
                            exp.config = cfg
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "baseline_eval":
                    # Persist baseline (pre-train) validation metrics so the UI
                    # can show base-vs-finetuned comparison.
                    db = db_factory()
                    try:
                        exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                        if exp:
                            cfg = dict(exp.config or {})
                            cfg["baseline_eval"] = {
                                "eval_loss": data.get("eval_loss"),
                                "token_accuracy": data.get("token_accuracy"),
                                "eval_samples": data.get("eval_samples"),
                                "eval_tokens": data.get("eval_tokens"),
                            }
                            exp.config = cfg
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "completed":
                    # Capture the final saved-weights output dir.
                    out = data.get("output_dir")
                    if out:
                        db = db_factory()
                        try:
                            exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                            if exp:
                                cfg = dict(exp.config or {})
                                cfg["final_output_dir"] = out
                                exp.config = cfg
                                db.commit()
                        except Exception:
                            db.rollback()
                        finally:
                            db.close()

                # Store every line as a log entry (progress, metrics, etc.)
                level = "metrics" if msg_type == "metrics" else (
                    "error" if msg_type == "error" else "info")
                db = db_factory()
                try:
                    db.add(TrainingLog(experiment_id=exp_id, level=level, message=line))
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()


training_manager = TrainingManager()
