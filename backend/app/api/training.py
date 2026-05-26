import asyncio
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models.models import Dataset, DatasetItem, TrainingExperiment, TrainingLog, TrainingMetric
from ..config import settings

router = APIRouter(prefix="/training", tags=["training"])


# ─────────────────────────────── Pydantic schemas ────────────────────────────

class CreateExperimentRequest(BaseModel):
    name: str
    base_model: str          # local model path or HuggingFace model id
    output_dir: Optional[str] = None
    dataset_id: Optional[int] = None
    train_ratio: float = 0.9
    config: Optional[dict] = {}


class SubmitMetricsRequest(BaseModel):
    step: int
    epoch: Optional[float] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    extra_metrics: Optional[dict] = {}


# ─────────────────────────────── helpers ─────────────────────────────────────

def _serialize_exp(e: TrainingExperiment) -> dict:
    cfg = e.config or {}
    dataset_name = e.dataset.name if getattr(e, "dataset", None) else None
    return {
        "id": e.id,
        "name": e.name,
        "base_model": e.base_model,
        "dataset_id": e.dataset_id,
        "dataset_name": dataset_name,
        "config": cfg,
        "status": e.status,
        "best_eval_loss": e.best_eval_loss,
        "process_pid": e.process_pid,
        "log_file": e.log_file,
        "error_message": e.error_message,
        "total_steps": e.total_steps,
        "current_step": e.current_step,
        "baseline_eval": cfg.get("baseline_eval"),
        "final_eval": cfg.get("final_eval"),
        "final_output_dir": cfg.get("final_output_dir") or cfg.get("output_dir") or "",
        "started_at": e.started_at.isoformat() if e.started_at else None,
        "completed_at": e.completed_at.isoformat() if e.completed_at else None,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


def _export_dataset(dataset_id: int, train_ratio: float, run_dir: Path) -> tuple[str, Optional[str]]:
    """
    Export dataset items as JSONL, split train/val.
    Returns (train_file_path, val_file_path_or_None).
    """
    db = SessionLocal()
    try:
        items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
        if not items:
            raise ValueError(f"Dataset {dataset_id} 没有数据条目")

        rows = [
            {
                "instruction": it.instruction,
                "input": it.input or "",
                "output": it.output,
                "system_prompt": it.system_prompt or "",
            }
            for it in items
        ]
    finally:
        db.close()

    random.shuffle(rows)
    n_train = max(1, int(len(rows) * train_ratio))
    train_rows = rows[:n_train]
    val_rows = rows[n_train:]

    train_file = str(run_dir / "train.jsonl")
    val_file = str(run_dir / "val.jsonl") if val_rows else None

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if val_rows:
        with open(val_file, "w", encoding="utf-8") as f:
            for r in val_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return train_file, val_file


# ─────────────────────────────── CRUD endpoints ──────────────────────────────

@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    exps = db.query(TrainingExperiment).order_by(TrainingExperiment.id.desc()).all()
    return [_serialize_exp(e) for e in exps]


@router.post("/experiments")
def create_experiment(req: CreateExperimentRequest, db: Session = Depends(get_db)):
    cfg = req.config or {}
    exp = TrainingExperiment(
        name=req.name,
        base_model=req.base_model,
        dataset_id=req.dataset_id,
        config={
            **cfg,
            "train_ratio": req.train_ratio,
            "output_dir": req.output_dir or "",
        },
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _serialize_exp(exp)


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    return _serialize_exp(exp)


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    if exp.status == "running":
        raise HTTPException(400, "请先停止训练再删除实验")
    db.query(TrainingMetric).filter(TrainingMetric.experiment_id == exp_id).delete()
    db.query(TrainingLog).filter(TrainingLog.experiment_id == exp_id).delete()
    db.delete(exp)
    db.commit()
    return {"ok": True}


@router.patch("/experiments/{exp_id}/status")
def update_status(exp_id: int, status: str, db: Session = Depends(get_db)):
    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    exp.status = status
    if status == "running" and not exp.started_at:
        exp.started_at = datetime.utcnow()
    if status in ("completed", "failed", "stopped"):
        exp.completed_at = datetime.utcnow()
    db.commit()
    return _serialize_exp(exp)


# ─────────────────────────────── training control ────────────────────────────

@router.post("/experiments/{exp_id}/start")
def start_training(exp_id: int, db: Session = Depends(get_db)):
    from ..services.trainer import training_manager

    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    if exp.status == "running":
        raise HTTPException(400, "训练已在运行中")
    if training_manager.is_running(exp_id):
        raise HTTPException(400, "进程已在运行中")

    cfg = exp.config or {}
    model_path = exp.base_model
    if not model_path:
        raise HTTPException(400, "请先配置基座模型路径")

    dataset_id = exp.dataset_id
    if not dataset_id:
        raise HTTPException(400, "请先关联训练数据集")

    run_dir = Path(settings.training_runs_dir) / str(exp_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = float(cfg.get("train_ratio", 0.9))
    try:
        train_file, val_file = _export_dataset(dataset_id, train_ratio, run_dir)
    except ValueError as e:
        raise HTTPException(400, str(e))

    output_dir = cfg.get("output_dir") or str(run_dir / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # rank-0 of the training script writes one JSON event per line here;
    # the manager tails this file to drive metric/log streaming.
    metrics_file = str(run_dir / "metrics.jsonl")
    Path(metrics_file).unlink(missing_ok=True)

    # gpu_ids semantics (see services/trainer.py):
    #   None / missing → auto    | [] → CPU only
    #   [0]            → single  | [0,1,...] → multi-GPU DDP
    gpu_ids = cfg.get("gpu_ids", None)
    if gpu_ids is not None and not isinstance(gpu_ids, list):
        raise HTTPException(400, "gpu_ids 必须是数组（或不传以使用自动模式）")
    if isinstance(gpu_ids, list):
        try:
            gpu_ids = [int(g) for g in gpu_ids]
        except (TypeError, ValueError):
            raise HTTPException(400, "gpu_ids 中包含非整数")

    train_config = {
        "model_path": model_path,
        "output_dir": output_dir,
        "train_file": train_file,
        "val_file": val_file,
        "metrics_file": metrics_file,
        "gpu_ids": gpu_ids,
        # Hyperparameters from config
        "learning_rate": cfg.get("learning_rate", 2e-4),
        "num_epochs": cfg.get("num_epochs", 3),
        "batch_size": cfg.get("batch_size", 4),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 4),
        "max_seq_length": cfg.get("max_seq_length", 2048),
        "warmup_ratio": cfg.get("warmup_ratio", 0.05),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "logging_steps": cfg.get("logging_steps", 10),
        "eval_steps": cfg.get("eval_steps", 50),
        "save_steps": cfg.get("save_steps", 100),
        # LoRA
        "use_lora": cfg.get("use_lora", True),
        "lora_r": cfg.get("lora_r", 16),
        "lora_alpha": cfg.get("lora_alpha", 32),
        "lora_dropout": cfg.get("lora_dropout", 0.05),
        "lora_target_modules": cfg.get("lora_target_modules", "all-linear"),
        # Precision
        "use_4bit": cfg.get("use_4bit", False),
        "use_bf16": cfg.get("use_bf16", False),
    }

    log_file = str(run_dir / "train.log")

    pid = training_manager.start(exp_id, train_config, SessionLocal)

    exp.status = "running"
    exp.started_at = datetime.utcnow()
    exp.completed_at = None
    exp.process_pid = pid
    exp.log_file = log_file
    exp.error_message = None
    exp.total_steps = None
    exp.current_step = 0
    db.commit()

    return {"ok": True, "pid": pid, "run_dir": str(run_dir)}


@router.post("/experiments/{exp_id}/stop")
def stop_training(exp_id: int, db: Session = Depends(get_db)):
    from ..services.trainer import training_manager

    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")

    training_manager.stop(exp_id)

    exp.status = "stopped"
    exp.completed_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


# ─────────────────────────────── metrics ─────────────────────────────────────

@router.post("/experiments/{exp_id}/metrics")
def submit_metrics(exp_id: int, req: SubmitMetricsRequest, db: Session = Depends(get_db)):
    exp = db.query(TrainingExperiment).filter(TrainingExperiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    metric = TrainingMetric(
        experiment_id=exp_id,
        step=req.step,
        epoch=req.epoch,
        train_loss=req.train_loss,
        eval_loss=req.eval_loss,
        learning_rate=req.learning_rate,
        extra_metrics=req.extra_metrics or {},
    )
    db.add(metric)
    if req.eval_loss is not None:
        if exp.best_eval_loss is None or req.eval_loss < exp.best_eval_loss:
            exp.best_eval_loss = req.eval_loss
    db.commit()
    return {"ok": True}


@router.get("/experiments/{exp_id}/metrics")
def get_metrics(exp_id: int, db: Session = Depends(get_db)):
    metrics = (
        db.query(TrainingMetric)
        .filter(TrainingMetric.experiment_id == exp_id)
        .order_by(TrainingMetric.step)
        .all()
    )
    return [
        {
            "step": m.step,
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "eval_loss": m.eval_loss,
            "learning_rate": m.learning_rate,
            "extra_metrics": m.extra_metrics,
            "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
        }
        for m in metrics
    ]


@router.get("/experiments/{exp_id}/logs")
def get_logs(exp_id: int, limit: int = 200, since_id: int = 0, db: Session = Depends(get_db)):
    logs = (
        db.query(TrainingLog)
        .filter(TrainingLog.experiment_id == exp_id, TrainingLog.id > since_id)
        .order_by(TrainingLog.id)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": lg.id,
            "level": lg.level,
            "message": lg.message,
            "created_at": lg.created_at.isoformat() if lg.created_at else None,
        }
        for lg in logs
    ]


# ─────────────────────────────── SSE stream ──────────────────────────────────

@router.get("/experiments/{exp_id}/logs/stream")
async def stream_logs(exp_id: int, since_id: int = 0):
    """Server-Sent Events endpoint for real-time training logs and metrics."""

    async def generate():
        last_id = since_id
        consecutive_idle = 0

        while True:
            db = SessionLocal()
            try:
                exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
                if not exp:
                    yield f"data: {json.dumps({'error': '实验不存在'})}\n\n"
                    return

                new_logs = (
                    db.query(TrainingLog)
                    .filter(TrainingLog.experiment_id == exp_id, TrainingLog.id > last_id)
                    .order_by(TrainingLog.id)
                    .limit(100)
                    .all()
                )

                for lg in new_logs:
                    payload = {
                        "id": lg.id,
                        "level": lg.level,
                        "message": lg.message,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    last_id = lg.id

                # Send current experiment status / progress
                status_payload = {
                    "status": exp.status,
                    "current_step": exp.current_step,
                    "total_steps": exp.total_steps,
                    "best_eval_loss": exp.best_eval_loss,
                }
                yield f"event: status\ndata: {json.dumps(status_payload)}\n\n"

                is_done = exp.status in ("completed", "failed", "stopped")
                if not new_logs:
                    consecutive_idle += 1
                else:
                    consecutive_idle = 0
            finally:
                db.close()

            if is_done:
                yield f"event: done\ndata: {json.dumps({'status': exp.status})}\n\n"
                return

            # Adaptive polling: slow down when idle
            sleep_time = 1.0 if consecutive_idle < 5 else 2.0
            await asyncio.sleep(sleep_time)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────── GPU info ────────────────────────────────────

@router.get("/gpu-info")
def get_gpu_info():
    """Query GPU availability via nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_used_mb": int(parts[3]),
                            "memory_free_mb": int(parts[4]),
                            "utilization_pct": int(parts[5]),
                            "temperature_c": int(parts[6]),
                        }
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"gpus": gpus, "cuda_available": len(gpus) > 0}


# ─────────────────────────────── stats ───────────────────────────────────────

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(TrainingExperiment).count()
    running = db.query(TrainingExperiment).filter(TrainingExperiment.status == "running").count()
    completed = db.query(TrainingExperiment).filter(TrainingExperiment.status == "completed").count()
    failed = db.query(TrainingExperiment).filter(TrainingExperiment.status == "failed").count()
    return {"total": total, "running": running, "completed": completed, "failed": failed}
