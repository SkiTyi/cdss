"""
Pretrain (CPT) subprocess manager.

Mirrors `trainer.py` (SFT manager): same launch model (single-GPU python /
multi-GPU torchrun), same metrics.jsonl tail strategy, same gpu_ids semantics.

Why a separate manager rather than parameterizing trainer.py:
  - Different DB tables (PretrainExperiment / PretrainMetric / PretrainLog)
  - Different `final_eval` payload (perplexity instead of token_accuracy)
  - Lets each loop evolve independently without cross-contamination
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


class PretrainManager:
    """Singleton tracking CPT subprocesses."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processes = {}
        return cls._instance

    # ── public ────────────────────────────────────────────────────────────

    def start(self, exp_id: int, config: dict, db_factory) -> int:
        from ..config import settings

        gpu_ids = config.get("gpu_ids")
        n_gpus = len(gpu_ids) if gpu_ids else 0
        metrics_file: str = config.get("metrics_file", "")

        run_dir = Path(settings.pretrain_runs_dir) / str(exp_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_file = str(run_dir / "config.json")
        log_file = str(run_dir / "pretrain.log")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        env = build_subprocess_env({"PYTHONUNBUFFERED": "1"})
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        script_path = Path(__file__).parent / "pretrain_script.py"
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        self._processes[exp_id] = proc

        threading.Thread(
            target=self._monitor,
            args=(exp_id, proc, metrics_file, log_file, db_factory),
            daemon=True,
        ).start()

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

    # ── private ───────────────────────────────────────────────────────────

    def _monitor(self, exp_id, proc, metrics_file, log_file, db_factory):
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(exp_id, proc, log_file, db_factory),
            daemon=True,
        )
        stdout_thread.start()
        self._tail_metrics_file(exp_id, proc, metrics_file, db_factory)

        return_code = proc.wait()
        stdout_thread.join(timeout=10)
        self._processes.pop(exp_id, None)

        db = db_factory()
        try:
            from ..models.models import PretrainExperiment
            exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
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

    def _read_stdout(self, exp_id, proc, log_file, db_factory):
        from ..models.models import PretrainLog
        with open(log_file, "w", encoding="utf-8") as lf:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lf.write(line + "\n")
                lf.flush()
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    continue  # JSON → handled by metrics tail
                except ValueError:
                    pass
                level = "error" if "error" in line.lower() or "traceback" in line.lower() else "info"
                db = db_factory()
                try:
                    db.add(PretrainLog(experiment_id=exp_id, level=level, message=line))
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()

    def _tail_metrics_file(self, exp_id, proc, metrics_file, db_factory):
        from ..models.models import PretrainExperiment, PretrainMetric, PretrainLog
        if not metrics_file:
            return
        for _ in range(600):
            if os.path.exists(metrics_file):
                break
            if proc.poll() is not None:
                return
            time.sleep(1)
        if not os.path.exists(metrics_file):
            return

        with open(metrics_file, "r", encoding="utf-8") as mf:
            while True:
                line = mf.readline()
                if not line:
                    if proc.poll() is not None:
                        break
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
                        extra = {k: v for k, v in data.items()
                                 if k not in ("type", "step", "epoch",
                                              "train_loss", "eval_loss",
                                              "perplexity", "learning_rate")}
                        db.add(PretrainMetric(
                            experiment_id=exp_id,
                            step=data.get("step", 0),
                            epoch=data.get("epoch"),
                            train_loss=data.get("train_loss"),
                            eval_loss=data.get("eval_loss"),
                            perplexity=data.get("perplexity"),
                            learning_rate=data.get("learning_rate"),
                            extra_metrics=extra,
                        ))
                        exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
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
                        exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
                        if exp and data.get("total_steps"):
                            exp.total_steps = data["total_steps"]
                            cfg = dict(exp.config or {})
                            cfg["corpus_stats"] = {
                                "train_blocks": data.get("train_blocks"),
                                "val_blocks": data.get("val_blocks"),
                                "total_tokens": data.get("total_tokens"),
                                "block_size": data.get("block_size"),
                            }
                            exp.config = cfg
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "final_eval":
                    db = db_factory()
                    try:
                        exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
                        if exp:
                            cfg = dict(exp.config or {})
                            cfg["final_eval"] = {
                                "eval_loss": data.get("eval_loss"),
                                "perplexity": data.get("perplexity"),
                                "val_blocks": data.get("val_blocks"),
                            }
                            exp.config = cfg
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                elif msg_type == "completed":
                    out = data.get("output_dir")
                    if out:
                        db = db_factory()
                        try:
                            exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
                            if exp:
                                cfg = dict(exp.config or {})
                                cfg["final_output_dir"] = out
                                exp.config = cfg
                                db.commit()
                        except Exception:
                            db.rollback()
                        finally:
                            db.close()

                # Also store every line as a log entry.
                level = "metrics" if msg_type == "metrics" else (
                    "error" if msg_type == "error" else "info")
                db = db_factory()
                try:
                    db.add(PretrainLog(experiment_id=exp_id, level=level, message=line))
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()


pretrain_manager = PretrainManager()
