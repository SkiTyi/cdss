"""Evaluation runs CRUD + lifecycle."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..models.models import (
    Dataset, EvaluationItem, EvaluationRun, LLMAssistant,
)
from ..services.evaluator import start_evaluation_thread

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


# ─────────────────────────────── Pydantic ────────────────────────────────

class CreateEvaluationRequest(BaseModel):
    name: str
    dataset_id: int
    candidate_assistant_id: int
    judge_assistant_id: int
    baseline_assistant_id: Optional[int] = None
    sample_limit: Optional[int] = None
    auto_start: bool = True


# ─────────────────────────────── helpers ─────────────────────────────────

def _serialize_run(r: EvaluationRun) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "dataset_id": r.dataset_id,
        "dataset_name": r.dataset.name if r.dataset else None,
        "candidate_assistant_id": r.candidate_assistant_id,
        "candidate_name": r.candidate.name if r.candidate else None,
        "baseline_assistant_id": r.baseline_assistant_id,
        "baseline_name": r.baseline.name if r.baseline else None,
        "judge_assistant_id": r.judge_assistant_id,
        "judge_name": r.judge.name if r.judge else None,
        "sample_limit": r.sample_limit,
        "status": r.status,
        "phase": r.phase or "pending",
        "candidate_score": r.candidate_score,
        "baseline_score": r.baseline_score,
        "candidate_pass_rate": r.candidate_pass_rate,
        "baseline_pass_rate": r.baseline_pass_rate,
        "progress_total": r.progress_total,
        "progress_done": r.progress_done,
        "error_message": r.error_message,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _serialize_item(i: EvaluationItem) -> dict:
    return {
        "id": i.id, "run_id": i.run_id, "dataset_item_id": i.dataset_item_id,
        "instruction": i.instruction, "expected_output": i.expected_output,
        "candidate_response": i.candidate_response,
        "candidate_score": i.candidate_score,
        "candidate_reasoning": i.candidate_reasoning,
        "baseline_response": i.baseline_response,
        "baseline_score": i.baseline_score,
        "baseline_reasoning": i.baseline_reasoning,
        "error_message": i.error_message,
        "created_at": i.created_at.isoformat() if i.created_at else None,
    }


# ─────────────────────────────── CRUD ────────────────────────────────────

@router.get("")
def list_runs(db: Session = Depends(get_db)):
    rows = db.query(EvaluationRun).order_by(EvaluationRun.id.desc()).all()
    return [_serialize_run(r) for r in rows]


@router.post("")
def create_run(req: CreateEvaluationRequest, db: Session = Depends(get_db)):
    if not db.query(Dataset).filter_by(id=req.dataset_id).first():
        raise HTTPException(404, "数据集不存在")
    if not db.query(LLMAssistant).filter_by(id=req.candidate_assistant_id).first():
        raise HTTPException(404, "candidate 助手不存在")
    if not db.query(LLMAssistant).filter_by(id=req.judge_assistant_id).first():
        raise HTTPException(404, "judge 助手不存在")
    if req.baseline_assistant_id and not db.query(LLMAssistant).filter_by(id=req.baseline_assistant_id).first():
        raise HTTPException(404, "baseline 助手不存在")

    run = EvaluationRun(
        name=req.name,
        dataset_id=req.dataset_id,
        candidate_assistant_id=req.candidate_assistant_id,
        baseline_assistant_id=req.baseline_assistant_id,
        judge_assistant_id=req.judge_assistant_id,
        sample_limit=req.sample_limit,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if req.auto_start:
        start_evaluation_thread(run.id, SessionLocal)

    return _serialize_run(run)


@router.get("/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter_by(id=run_id).first()
    if not r:
        raise HTTPException(404, "评估任务不存在")
    return _serialize_run(r)


@router.post("/{run_id}/start")
def start_run(run_id: int, restart_mode: str = "resume", db: Session = Depends(get_db)):
    """Start (or resume) a run.

    - restart_mode=resume (default): keep existing item rows + cached responses;
      runner skips items that already have output for the current phase. Use this
      when re-launching after switching the vllm-served model between phases.
    - restart_mode=fresh: drop all item rows and re-score from scratch.
    """
    r = db.query(EvaluationRun).filter_by(id=run_id).first()
    if not r:
        raise HTTPException(404, "评估任务不存在")
    if r.status == "running":
        raise HTTPException(400, "已在运行")

    if restart_mode == "fresh":
        db.query(EvaluationItem).filter_by(run_id=r.id).delete()
        r.candidate_score = None
        r.baseline_score = None
        r.candidate_pass_rate = None
        r.baseline_pass_rate = None
        r.progress_done = 0
        r.progress_total = 0
        r.phase = "pending"
    elif restart_mode != "resume":
        raise HTTPException(400, f"未知 restart_mode：{restart_mode}")

    r.error_message = None
    r.is_cancelled = False
    r.completed_at = None
    db.commit()
    start_evaluation_thread(r.id, SessionLocal)
    db.refresh(r)
    return _serialize_run(r)


@router.post("/{run_id}/cancel")
def cancel_run(run_id: int, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter_by(id=run_id).first()
    if not r:
        raise HTTPException(404, "评估任务不存在")
    if r.status != "running":
        raise HTTPException(400, "只有运行中的任务可以取消")
    r.is_cancelled = True
    db.commit()
    return {"ok": True}


@router.delete("/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter_by(id=run_id).first()
    if not r:
        raise HTTPException(404, "评估任务不存在")
    if r.status == "running":
        raise HTTPException(400, "请先取消再删除")
    db.query(EvaluationItem).filter_by(run_id=r.id).delete()
    db.delete(r)
    db.commit()
    return {"ok": True}


@router.get("/{run_id}/items")
def list_items(run_id: int, page: int = 1, page_size: int = 20,
               db: Session = Depends(get_db)):
    q = db.query(EvaluationItem).filter_by(run_id=run_id).order_by(EvaluationItem.id)
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [_serialize_item(i) for i in items],
    }
