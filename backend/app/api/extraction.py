from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
from pydantic import BaseModel
from ..database import get_db
from ..models.models import ExtractionJob, DiagnosticInstance
from ..services.extractor import (
    run_extraction_job, CASE_PROMPT, GUIDELINE_PROMPT, CLINICAL_REASONING_PROMPT,
)

router = APIRouter(prefix="/extraction", tags=["extraction"])


# Phase 1: task_type ∈ {case_extract, guideline_synth, case_reasoning}
# (augment is reserved for Step 1.3 and not yet wired into create_job)
_VALID_TASK_TYPES = {"case_extract", "guideline_synth", "case_reasoning"}


class CreateJobRequest(BaseModel):
    name: str
    task_type: str = "case_extract"
    prompt_template: Optional[str] = None
    assistant_id: Optional[int] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    doc_limit: Optional[int] = None
    # document_type is now derived from task_type for most cases but kept
    # so future extensions (e.g. mixed-source extraction) can override.
    document_type: Optional[str] = None


def _serialize_job(j: ExtractionJob):
    return {
        "id": j.id, "name": j.name,
        "document_type": j.document_type,
        "task_type": j.task_type or "case_extract",
        "assistant_id": j.assistant_id,
        "model": j.model, "base_url": j.base_url,
        "has_api_key": bool(j.api_key),
        "status": j.status,
        "total_docs": j.total_docs, "processed_docs": j.processed_docs,
        "failed_docs": j.failed_docs, "error_message": j.error_message,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "created_at": j.created_at.isoformat() if j.created_at else None,
    }


def _serialize_instance(i: DiagnosticInstance):
    return {
        "id": i.id,
        "presentation": i.presentation,
        "answer": i.answer,
        "diagnosis_label": i.diagnosis_label,
        "specialty": i.specialty,
        "difficulty": i.difficulty,
        "synthesis_strategy": i.synthesis_strategy,
        "parent_instance_id": i.parent_instance_id,
        "source_doc_id": i.source_doc_id,
        "job_id": i.job_id,
        "quality_score": i.quality_score,
        "is_approved": i.is_approved,
        "created_at": i.created_at.isoformat() if i.created_at else None,
    }


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(ExtractionJob).order_by(ExtractionJob.id.desc()).all()
    return [_serialize_job(j) for j in jobs]


@router.post("/jobs")
def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    base_url = (req.base_url or "").strip() or None
    api_key = (req.api_key or "").strip() or None
    if not req.assistant_id and base_url and api_key is None:
        if not any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1")):
            raise HTTPException(400, "远程 base_url 必须提供 api_key（仅 localhost 可省略）")

    task_type = (req.task_type or "case_extract").strip()
    if task_type not in _VALID_TASK_TYPES:
        raise HTTPException(400, f"未知的 task_type：{task_type}（合法值：{sorted(_VALID_TASK_TYPES)}）")

    # task_type implies document type
    document_type = req.document_type or {
        "guideline_synth": "guideline",
        "case_extract": "case_report",
        "case_reasoning": "case_report",
    }[task_type]

    if req.assistant_id:
        from ..models.models import LLMAssistant
        if not db.query(LLMAssistant).filter_by(id=req.assistant_id).first():
            raise HTTPException(404, "指定的助手不存在")

    job = ExtractionJob(
        name=req.name,
        document_type=document_type,
        task_type=task_type,
        prompt_template=req.prompt_template,
        assistant_id=req.assistant_id,
        model=req.model,
        base_url=base_url,
        api_key=api_key,
        doc_limit=req.doc_limit,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(run_extraction_job, job.id, db)
    return _serialize_job(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "任务不存在")
    return _serialize_job(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status != "running":
        raise HTTPException(400, "只有运行中的任务可以暂停")
    job.is_cancelled = True
    db.commit()
    return {"id": job.id, "message": "已发送暂停信号，任务将在处理完当前文档后停止"}


@router.post("/jobs/{job_id}/restart")
def restart_job(job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status == "running":
        raise HTTPException(400, "任务正在运行中")
    # Wipe the old instances produced by this job so the restart starts clean.
    db.query(DiagnosticInstance).filter(DiagnosticInstance.job_id == job_id).delete()
    job.status = "pending"
    job.is_cancelled = False
    job.processed_docs = 0
    job.failed_docs = 0
    job.total_docs = 0
    job.error_message = None
    job.started_at = None
    job.completed_at = None
    db.commit()
    background_tasks.add_task(run_extraction_job, job.id, db)
    return _serialize_job(job)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status == "running":
        job.is_cancelled = True
        db.commit()
    db.query(DiagnosticInstance).filter(DiagnosticInstance.job_id == job_id).delete()
    db.delete(job)
    db.commit()
    return {"message": "删除成功"}


# ─────────────────────────── instances ────────────────────────────────────

@router.get("/instances")
def list_instances(
    job_id: Optional[int] = None,
    synthesis_strategy: Optional[str] = None,
    diagnosis_label: Optional[str] = None,
    is_approved: Optional[bool] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    q = db.query(DiagnosticInstance)
    if job_id:
        q = q.filter(DiagnosticInstance.job_id == job_id)
    if synthesis_strategy:
        q = q.filter(DiagnosticInstance.synthesis_strategy == synthesis_strategy)
    if diagnosis_label:
        q = q.filter(DiagnosticInstance.diagnosis_label == diagnosis_label)
    if is_approved is not None:
        q = q.filter(DiagnosticInstance.is_approved == is_approved)
    total = q.count()
    items = q.order_by(DiagnosticInstance.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [_serialize_instance(i) for i in items],
    }


@router.patch("/instances/{instance_id}/approve")
def approve_instance(instance_id: int, db: Session = Depends(get_db)):
    item = db.query(DiagnosticInstance).filter(DiagnosticInstance.id == instance_id).first()
    if not item:
        raise HTTPException(404, "实例不存在")
    item.is_approved = not item.is_approved
    db.commit()
    return {"id": item.id, "is_approved": item.is_approved}


# ─────────────────────────── stats / defaults ─────────────────────────────

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_jobs = db.query(ExtractionJob).count()
    running = db.query(ExtractionJob).filter(ExtractionJob.status == "running").count()
    total_instances = db.query(DiagnosticInstance).count()
    approved = db.query(DiagnosticInstance).filter(DiagnosticInstance.is_approved == True).count()
    return {
        "total_jobs": total_jobs,
        "running_jobs": running,
        "total_instances": total_instances,
        "approved_instances": approved,
    }


@router.get("/diagnosis-distribution")
def diagnosis_distribution(top: int = 30, db: Session = Depends(get_db)):
    """Histogram of diagnosis_label counts — used by Step 1.4 sampler UI.

    Surfaces both the head (top-N most common) and the long tail (count of
    labels with only 1 instance), so operators can see at a glance whether
    the dataset is dominated by a few diseases.
    """
    rows = (
        db.query(DiagnosticInstance.diagnosis_label,
                 func.count(DiagnosticInstance.id).label("n"))
        .filter(DiagnosticInstance.diagnosis_label.isnot(None))
        .filter(DiagnosticInstance.diagnosis_label != "")
        .group_by(DiagnosticInstance.diagnosis_label)
        .order_by(func.count(DiagnosticInstance.id).desc())
        .all()
    )
    total = sum(n for _, n in rows)
    head = [{"label": lbl, "count": n} for lbl, n in rows[:top]]
    singletons = sum(1 for _, n in rows if n == 1)
    return {
        "distinct_labels": len(rows),
        "total_instances": total,
        "singletons": singletons,
        "head": head,
    }


@router.get("/prompts/defaults")
def get_default_prompts():
    return {
        "case_extract":    CASE_PROMPT,
        "guideline_synth": GUIDELINE_PROMPT,
        "case_reasoning":  CLINICAL_REASONING_PROMPT,
    }
