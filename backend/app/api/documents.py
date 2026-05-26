from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from ..database import get_db
from ..models.models import Document, KnowledgeItem
from ..services.data_loader import load_cases, load_guidelines

router = APIRouter(prefix="/documents", tags=["documents"])


def _serialize(d: Document):
    return {
        "id": d.id, "type": d.type, "title": d.title,
        "source_path": d.source_path, "status": d.status,
        "metadata": d.doc_metadata,
        "content_length": len(d.content) if d.content else 0,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


def _detach_knowledge_refs(db: Session, doc_ids: List[int]):
    """NULL out knowledge_items.document_id for soon-to-be-deleted documents.

    SQLite doesn't enforce FKs by default, but leaving dangling document_id
    values makes later joins surprising. Knowledge items themselves are kept
    — they remain valid for dataset construction even after their source
    document is removed.
    """
    if not doc_ids:
        return
    db.query(KnowledgeItem).filter(KnowledgeItem.document_id.in_(doc_ids)).update(
        {KnowledgeItem.document_id: None}, synchronize_session=False
    )


@router.get("")
def list_documents(
    type: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    min_content_length: int = 0,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    q = db.query(Document)
    if type:
        q = q.filter(Document.type == type)
    if status:
        q = q.filter(Document.status == status)
    if search:
        q = q.filter(Document.title.contains(search))
    if min_content_length and min_content_length > 0:
        # length() in SQLite returns NULL for NULL content — coalesce to 0.
        q = q.filter(func.coalesce(func.length(Document.content), 0) >= min_content_length)
    total = q.count()
    items = q.order_by(Document.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": [_serialize(d) for d in items]}


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Document).count()
    cases = db.query(Document).filter(Document.type == "case_report").count()
    guidelines = db.query(Document).filter(Document.type == "guideline").count()
    extracted = db.query(Document).filter(Document.status == "extracted").count()
    # Surface short-doc count so users know how many would be filtered out at common thresholds.
    short_100 = db.query(Document).filter(
        func.coalesce(func.length(Document.content), 0) < 100
    ).count()
    short_500 = db.query(Document).filter(
        func.coalesce(func.length(Document.content), 0) < 500
    ).count()
    return {
        "total": total,
        "case_reports": cases,
        "guidelines": guidelines,
        "extracted": extracted,
        "short_lt_100": short_100,
        "short_lt_500": short_500,
    }


@router.post("/load")
def load_documents(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    def _load():
        load_cases(db)
        load_guidelines(db)
    background_tasks.add_task(_load)
    return {"message": "数据加载已启动，请稍后刷新查看结果"}


# ── deletion endpoints ────────────────────────────────────────────────────
# Deletes only remove rows from the documents table — the underlying files
# on disk are untouched, so re-running /load can always rebuild the DB.

class DeleteBulkRequest(BaseModel):
    ids: List[int]


class DeleteShortRequest(BaseModel):
    threshold: int                       # min character count to keep
    type: Optional[str] = None           # case_report | guideline | None (=all)


@router.post("/delete-bulk")
def delete_bulk(req: DeleteBulkRequest, db: Session = Depends(get_db)):
    if not req.ids:
        return {"deleted": 0}
    _detach_knowledge_refs(db, req.ids)
    deleted = db.query(Document).filter(Document.id.in_(req.ids)).delete(
        synchronize_session=False
    )
    db.commit()
    return {"deleted": int(deleted)}


@router.post("/delete-short")
def delete_short(req: DeleteShortRequest, db: Session = Depends(get_db)):
    if req.threshold <= 0:
        raise HTTPException(400, "threshold 必须大于 0")
    q = db.query(Document).filter(
        func.coalesce(func.length(Document.content), 0) < req.threshold
    )
    if req.type in ("case_report", "guideline"):
        q = q.filter(Document.type == req.type)
    ids = [d.id for d in q.with_entities(Document.id).all()]
    if not ids:
        return {"deleted": 0}
    _detach_knowledge_refs(db, ids)
    deleted = db.query(Document).filter(Document.id.in_(ids)).delete(
        synchronize_session=False
    )
    db.commit()
    return {"deleted": int(deleted)}


@router.post("/reset")
def reset_documents(db: Session = Depends(get_db)):
    """Wipe the entire documents table. Files on disk are not touched —
    POST /documents/load rebuilds everything."""
    db.query(KnowledgeItem).update(
        {KnowledgeItem.document_id: None}, synchronize_session=False
    )
    deleted = db.query(Document).delete(synchronize_session=False)
    db.commit()
    return {"deleted": int(deleted)}


@router.delete("/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "文档不存在")
    _detach_knowledge_refs(db, [doc_id])
    db.delete(doc)
    db.commit()
    return {"deleted": 1}


@router.get("/{doc_id}")
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "文档不存在")
    result = _serialize(doc)
    result["content"] = doc.content or ""
    return result
