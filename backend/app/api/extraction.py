from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
from ..database import get_db
from ..models.models import ExtractionJob, KnowledgeItem
from ..services.extractor import (
    run_extraction_job, CASE_PROMPT, GUIDELINE_PROMPT, CLINICAL_REASONING_PROMPT,
)

router = APIRouter(prefix="/extraction", tags=["extraction"])


class CreateJobRequest(BaseModel):
    name: str
    document_type: str = "case_report"  # case_report | guideline | all
    task_type: str = "qa_extraction"    # qa_extraction | clinical_reasoning_synthesis
    prompt_template: Optional[str] = None
    assistant_id: Optional[int] = None  # if set, takes precedence over base_url/model/api_key
    model: Optional[str] = None
    base_url: Optional[str] = None      # LLM API base url override
    api_key: Optional[str] = None       # may be empty when base_url is localhost
    doc_limit: Optional[int] = None  # 限制处理文档数量，None 表示全部


def _serialize_job(j: ExtractionJob):
    return {
        "id": j.id, "name": j.name, "document_type": j.document_type,
        "task_type": j.task_type or "qa_extraction",
        "assistant_id": j.assistant_id,
        "model": j.model, "base_url": j.base_url,
        "has_api_key": bool(j.api_key),     # never expose the key itself
        "status": j.status,
        "total_docs": j.total_docs, "processed_docs": j.processed_docs,
        "failed_docs": j.failed_docs, "error_message": j.error_message,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "created_at": j.created_at.isoformat() if j.created_at else None,
    }


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(ExtractionJob).order_by(ExtractionJob.id.desc()).all()
    return [_serialize_job(j) for j in jobs]


@router.post("/jobs")
def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    base_url = (req.base_url or "").strip() or None
    api_key = (req.api_key or "").strip() or None
    # If a non-local base_url is provided (and no assistant_id), an api_key is required.
    if not req.assistant_id and base_url and api_key is None:
        if not any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1")):
            raise HTTPException(400, "远程 base_url 必须提供 api_key（仅 localhost 可省略）")

    task_type = (req.task_type or "qa_extraction").strip()
    if task_type not in ("qa_extraction", "clinical_reasoning_synthesis"):
        raise HTTPException(400, f"未知的 task_type：{task_type}")

    # Reasoning synthesis is meaningful only on case reports — fix doc type
    document_type = req.document_type
    if task_type == "clinical_reasoning_synthesis":
        document_type = "case_report"

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
    # 重置状态，清除旧的知识条目
    db.query(KnowledgeItem).filter(KnowledgeItem.job_id == job_id).delete()
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
    db.query(KnowledgeItem).filter(KnowledgeItem.job_id == job_id).delete()
    db.delete(job)
    db.commit()
    return {"message": "删除成功"}


@router.get("/knowledge")
def list_knowledge(
    job_id: Optional[int] = None,
    knowledge_type: Optional[str] = None,
    is_approved: Optional[bool] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    q = db.query(KnowledgeItem)
    if job_id:
        q = q.filter(KnowledgeItem.job_id == job_id)
    if knowledge_type:
        q = q.filter(KnowledgeItem.knowledge_type == knowledge_type)
    if is_approved is not None:
        q = q.filter(KnowledgeItem.is_approved == is_approved)
    total = q.count()
    items = q.order_by(KnowledgeItem.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [
            {
                "id": i.id, "job_id": i.job_id, "document_id": i.document_id,
                "knowledge_type": i.knowledge_type, "content": i.content,
                "quality_score": i.quality_score, "is_approved": i.is_approved,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in items
        ],
    }


@router.patch("/knowledge/{item_id}/approve")
def approve_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    if not item:
        raise HTTPException(404, "知识条目不存在")
    item.is_approved = not item.is_approved
    db.commit()
    return {"id": item.id, "is_approved": item.is_approved}


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_jobs = db.query(ExtractionJob).count()
    running = db.query(ExtractionJob).filter(ExtractionJob.status == "running").count()
    total_items = db.query(KnowledgeItem).count()
    approved = db.query(KnowledgeItem).filter(KnowledgeItem.is_approved == True).count()
    return {"total_jobs": total_jobs, "running_jobs": running, "total_items": total_items, "approved_items": approved}


@router.get("/prompts/defaults")
def get_default_prompts():
    return {
        "case_report": CASE_PROMPT,
        "guideline": GUIDELINE_PROMPT,
        "clinical_reasoning": CLINICAL_REASONING_PROMPT,
    }
