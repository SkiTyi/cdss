"""CPT (continued pre-training) experiment CRUD + lifecycle."""
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..models.models import (
    Document, PretrainExperiment, PretrainMetric, PretrainLog,
)

router = APIRouter(prefix="/pretraining", tags=["pretraining"])


# ─────────────────────────────── schemas ─────────────────────────────────

class CorpusFilter(BaseModel):
    document_types: Optional[List[str]] = None   # subset of [case_report, guideline]
    min_content_length: int = 0
    doc_limit: Optional[int] = None
    eval_ratio: float = 0.05                     # held-out for perplexity


class CreatePretrainRequest(BaseModel):
    name: str
    base_model: str
    output_dir: Optional[str] = None
    corpus_filter: Optional[CorpusFilter] = None
    config: Optional[dict] = {}


# ─────────────────────────────── helpers ─────────────────────────────────

def _serialize(e: PretrainExperiment) -> dict:
    cfg = e.config or {}
    return {
        "id": e.id,
        "name": e.name,
        "base_model": e.base_model,
        "corpus_filter": e.corpus_filter or {},
        "config": cfg,
        "status": e.status,
        "best_eval_loss": e.best_eval_loss,
        "process_pid": e.process_pid,
        "log_file": e.log_file,
        "error_message": e.error_message,
        "total_steps": e.total_steps,
        "current_step": e.current_step,
        "corpus_stats": cfg.get("corpus_stats"),
        "final_eval": cfg.get("final_eval"),
        "final_output_dir": cfg.get("final_output_dir") or cfg.get("output_dir") or "",
        "started_at": e.started_at.isoformat() if e.started_at else None,
        "completed_at": e.completed_at.isoformat() if e.completed_at else None,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


def _query_corpus_docs(db, filt: dict):
    """Apply CorpusFilter to documents and return ORM rows."""
    q = db.query(Document)
    types = filt.get("document_types") or []
    if types and "all" not in types:
        q = q.filter(Document.type.in_(types))
    min_len = int(filt.get("min_content_length") or 0)
    if min_len > 0:
        q = q.filter(func.coalesce(func.length(Document.content), 0) >= min_len)
    q = q.order_by(Document.id)
    rows = q.all()
    limit = filt.get("doc_limit")
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _export_corpus(exp_id: int, filt: dict, run_dir: Path):
    """Dump filtered documents as JSONL train/eval splits.

    Returns (train_path, eval_path_or_None, train_count, eval_count).
    """
    db = SessionLocal()
    try:
        rows = _query_corpus_docs(db, filt)
        if not rows:
            raise ValueError("过滤后无可用文档")

        eval_ratio = float(filt.get("eval_ratio") or 0.0)
        eval_ratio = max(0.0, min(eval_ratio, 0.5))
        n_eval = int(round(len(rows) * eval_ratio)) if eval_ratio > 0 else 0
        eval_rows = rows[-n_eval:] if n_eval > 0 else []
        train_rows = rows[: len(rows) - n_eval] if n_eval > 0 else rows

        train_path = run_dir / "corpus_train.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for d in train_rows:
                if not d.content:
                    continue
                f.write(json.dumps({"text": d.content}, ensure_ascii=False) + "\n")

        eval_path = None
        if eval_rows:
            eval_path = run_dir / "corpus_eval.jsonl"
            with open(eval_path, "w", encoding="utf-8") as f:
                for d in eval_rows:
                    if not d.content:
                        continue
                    f.write(json.dumps({"text": d.content}, ensure_ascii=False) + "\n")
        return str(train_path), (str(eval_path) if eval_path else None), len(train_rows), len(eval_rows)
    finally:
        db.close()


# ─────────────────────────────── corpus preview ──────────────────────────

@router.post("/preview-corpus")
def preview_corpus(filt: CorpusFilter, db: Session = Depends(get_db)):
    """Tell the user how many docs / chars will go into a CPT run with these filters."""
    rows = _query_corpus_docs(db, filt.model_dump())
    total_chars = 0
    by_type = {"case_report": 0, "guideline": 0}
    for d in rows[:5000]:  # cap to avoid scanning gigantic corpora in this preview
        if d.content:
            total_chars += len(d.content)
        by_type[d.type] = by_type.get(d.type, 0) + 1
    return {
        "doc_count": len(rows),
        "by_type": by_type,
        "total_chars_sampled": total_chars,
        "sampled": min(len(rows), 5000),
        "estimated_tokens": int(total_chars / 1.6),  # rough avg for Chinese mix
        "eval_split_count": int(round(len(rows) * (filt.eval_ratio or 0))),
    }


# ─────────────────────────────── CRUD ────────────────────────────────────

@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    rows = db.query(PretrainExperiment).order_by(PretrainExperiment.id.desc()).all()
    return [_serialize(e) for e in rows]


@router.post("/experiments")
def create_experiment(req: CreatePretrainRequest, db: Session = Depends(get_db)):
    cfg = req.config or {}
    if req.output_dir:
        cfg["output_dir"] = req.output_dir
    exp = PretrainExperiment(
        name=req.name,
        base_model=req.base_model,
        corpus_filter=(req.corpus_filter.model_dump() if req.corpus_filter else {}),
        config=cfg,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _serialize(exp)


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    return _serialize(exp)


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    if exp.status == "running":
        raise HTTPException(400, "请先停止再删除")
    db.query(PretrainMetric).filter(PretrainMetric.experiment_id == exp_id).delete()
    db.query(PretrainLog).filter(PretrainLog.experiment_id == exp_id).delete()
    db.delete(exp)
    db.commit()
    return {"ok": True}


# ─────────────────────────────── lifecycle ───────────────────────────────

@router.post("/experiments/{exp_id}/start")
def start(exp_id: int, db: Session = Depends(get_db)):
    from ..services.pretrainer import pretrain_manager

    exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    if exp.status == "running":
        raise HTTPException(400, "已在运行")
    if pretrain_manager.is_running(exp_id):
        raise HTTPException(400, "进程已在运行")
    if not exp.base_model:
        raise HTTPException(400, "未配置基座模型")

    cfg = exp.config or {}
    run_dir = Path(settings.pretrain_runs_dir) / str(exp_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export corpus
    try:
        train_file, eval_file, n_train, n_eval = _export_corpus(
            exp_id, exp.corpus_filter or {}, run_dir)
    except ValueError as e:
        raise HTTPException(400, str(e))

    output_dir = cfg.get("output_dir") or str(run_dir / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics_file = str(run_dir / "metrics.jsonl")
    Path(metrics_file).unlink(missing_ok=True)

    # gpu_ids semantics same as SFT
    gpu_ids = cfg.get("gpu_ids", None)
    if gpu_ids is not None and not isinstance(gpu_ids, list):
        raise HTTPException(400, "gpu_ids 必须是数组")
    if isinstance(gpu_ids, list):
        try:
            gpu_ids = [int(g) for g in gpu_ids]
        except (TypeError, ValueError):
            raise HTTPException(400, "gpu_ids 中包含非整数")

    train_config = {
        "model_path": exp.base_model,
        "output_dir": output_dir,
        "corpus_file": train_file,
        "eval_corpus_file": eval_file,
        "metrics_file": metrics_file,
        "gpu_ids": gpu_ids,
        # CPT defaults intentionally lower-LR than SFT; user can override.
        "learning_rate": cfg.get("learning_rate", 5e-5),
        "num_epochs": cfg.get("num_epochs", 1),
        "batch_size": cfg.get("batch_size", 2),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
        "block_size": cfg.get("block_size", 4096),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "logging_steps": cfg.get("logging_steps", 10),
        "eval_steps": cfg.get("eval_steps", 200),
        "save_steps": cfg.get("save_steps", 500),
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

    pid = pretrain_manager.start(exp_id, train_config, SessionLocal)

    exp.status = "running"
    exp.started_at = datetime.utcnow()
    exp.completed_at = None
    exp.process_pid = pid
    exp.log_file = str(run_dir / "pretrain.log")
    exp.error_message = None
    exp.total_steps = None
    exp.current_step = 0
    # Stash corpus stats so the UI shows them even before the script reports back
    cfg = dict(exp.config or {})
    cfg["corpus_export"] = {"train_docs": n_train, "eval_docs": n_eval}
    exp.config = cfg
    db.commit()
    return {"ok": True, "pid": pid, "run_dir": str(run_dir),
            "train_docs": n_train, "eval_docs": n_eval}


@router.post("/experiments/{exp_id}/stop")
def stop(exp_id: int, db: Session = Depends(get_db)):
    from ..services.pretrainer import pretrain_manager
    exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
    if not exp:
        raise HTTPException(404, "实验不存在")
    pretrain_manager.stop(exp_id)
    exp.status = "stopped"
    exp.completed_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


# ─────────────────────────────── metrics / logs ──────────────────────────

@router.get("/experiments/{exp_id}/metrics")
def get_metrics(exp_id: int, db: Session = Depends(get_db)):
    rows = (db.query(PretrainMetric)
            .filter(PretrainMetric.experiment_id == exp_id)
            .order_by(PretrainMetric.step).all())
    return [
        {
            "step": m.step, "epoch": m.epoch,
            "train_loss": m.train_loss, "eval_loss": m.eval_loss,
            "perplexity": m.perplexity,
            "learning_rate": m.learning_rate,
            "extra_metrics": m.extra_metrics,
            "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
        }
        for m in rows
    ]


@router.get("/experiments/{exp_id}/logs")
def get_logs(exp_id: int, limit: int = 200, since_id: int = 0, db: Session = Depends(get_db)):
    rows = (db.query(PretrainLog)
            .filter(PretrainLog.experiment_id == exp_id, PretrainLog.id > since_id)
            .order_by(PretrainLog.id).limit(limit).all())
    return [
        {"id": lg.id, "level": lg.level, "message": lg.message,
         "created_at": lg.created_at.isoformat() if lg.created_at else None}
        for lg in rows
    ]


@router.get("/experiments/{exp_id}/logs/stream")
async def stream_logs(exp_id: int, since_id: int = 0):
    async def generate():
        last_id = since_id
        consecutive_idle = 0
        while True:
            db = SessionLocal()
            try:
                exp = db.query(PretrainExperiment).filter_by(id=exp_id).first()
                if not exp:
                    yield f"data: {json.dumps({'error': '实验不存在'})}\n\n"
                    return
                new_logs = (db.query(PretrainLog)
                            .filter(PretrainLog.experiment_id == exp_id, PretrainLog.id > last_id)
                            .order_by(PretrainLog.id).limit(100).all())
                for lg in new_logs:
                    yield f"data: {json.dumps({'id': lg.id, 'level': lg.level, 'message': lg.message}, ensure_ascii=False)}\n\n"
                    last_id = lg.id
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
            await asyncio.sleep(1.0 if consecutive_idle < 5 else 2.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ─────────────────────────────── stats ───────────────────────────────────

@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {
        "total": db.query(PretrainExperiment).count(),
        "running": db.query(PretrainExperiment).filter_by(status="running").count(),
        "completed": db.query(PretrainExperiment).filter_by(status="completed").count(),
        "failed": db.query(PretrainExperiment).filter_by(status="failed").count(),
    }
