import json
import random
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from ..database import get_db
from ..models.models import Dataset, DatasetItem, KnowledgeItem

router = APIRouter(prefix="/datasets", tags=["datasets"])


class CreateDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None
    format: str = "alpaca"
    knowledge_item_ids: Optional[List[int]] = None
    job_id: Optional[int] = None
    system_prompt: Optional[str] = "你是一位专业的临床医学助手，请根据患者的病情描述给出专业的诊断分析和治疗建议。"


def _serialize_dataset(d: Dataset):
    return {
        "id": d.id, "name": d.name, "description": d.description,
        "format": d.format, "item_count": d.item_count, "status": d.status,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.id.desc()).all()
    return [_serialize_dataset(d) for d in datasets]


@router.post("")
def create_dataset(req: CreateDatasetRequest, db: Session = Depends(get_db)):
    dataset = Dataset(name=req.name, description=req.description, format=req.format)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # collect knowledge items — supports both QA pairs and clinical reasoning samples.
    # Both store content as {"question": ..., "answer": ..., ...}, so they share the
    # same conversion path; we just widen the knowledge_type filter.
    q = db.query(KnowledgeItem).filter(
        KnowledgeItem.knowledge_type.in_(("qa_pair", "clinical_reasoning"))
    )
    if req.knowledge_item_ids:
        q = q.filter(KnowledgeItem.id.in_(req.knowledge_item_ids))
    elif req.job_id:
        q = q.filter(KnowledgeItem.job_id == req.job_id)

    items = q.all()
    count = 0
    for ki in items:
        content = ki.content or {}
        question = content.get("question", "")
        answer = content.get("answer", "")
        if not question or not answer:
            continue
        di = DatasetItem(
            dataset_id=dataset.id,
            knowledge_item_id=ki.id,
            instruction=question,
            input="",
            output=answer,
            system_prompt=req.system_prompt,
        )
        db.add(di)
        count += 1

    dataset.item_count = count
    dataset.status = "ready"
    db.commit()
    db.refresh(dataset)
    return _serialize_dataset(dataset)


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not d:
        raise HTTPException(404, "数据集不存在")
    return _serialize_dataset(d)


@router.get("/{dataset_id}/items")
def list_items(dataset_id: int, page: int = 1, page_size: int = 20, db: Session = Depends(get_db)):
    q = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id)
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [
            {"id": i.id, "instruction": i.instruction, "input": i.input, "output": i.output, "system_prompt": i.system_prompt}
            for i in items
        ],
    }


@router.get("/{dataset_id}/export")
def export_dataset(dataset_id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not d:
        raise HTTPException(404, "数据集不存在")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

    if d.format == "alpaca":
        data = [{"instruction": i.instruction, "input": i.input or "", "output": i.output, "system": i.system_prompt or ""} for i in items]
    elif d.format == "sharegpt":
        data = [{"conversations": [{"from": "human", "value": i.instruction}, {"from": "gpt", "value": i.output}]} for i in items]
    else:
        data = [{"instruction": i.instruction, "input": i.input, "output": i.output} for i in items]

    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in data)
    filename = f"{d.name}.jsonl"
    return StreamingResponse(
        iter([content]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not d:
        raise HTTPException(404, "数据集不存在")
    db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).delete()
    db.delete(d)
    db.commit()
    return {"message": "已删除"}


# ─────────────────────────────── random split ───────────────────────────────

class SplitDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None
    sample_size: Optional[int] = None    # take exactly N items
    ratio: Optional[float] = None        # OR take this fraction (0 < r <= 1)
    seed: Optional[int] = None           # reproducible split


@router.post("/{dataset_id}/split")
def split_dataset(dataset_id: int, req: SplitDatasetRequest, db: Session = Depends(get_db)):
    """Randomly subsample an existing dataset into a new (smaller) dataset.

    Useful for spinning up a quick eval set without rebuilding from knowledge items.
    """
    src = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not src:
        raise HTTPException(404, "源数据集不存在")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    if not items:
        raise HTTPException(400, "源数据集没有条目，无法切分")

    # Resolve target size
    if req.sample_size is not None and req.sample_size > 0:
        n = min(int(req.sample_size), len(items))
    elif req.ratio is not None and 0 < req.ratio <= 1:
        n = max(1, int(round(len(items) * req.ratio)))
    else:
        raise HTTPException(400, "请指定 sample_size 或 ratio (0~1)")

    if db.query(Dataset).filter(Dataset.name == req.name).first():
        raise HTTPException(400, f"已存在同名数据集：{req.name}")

    rng = random.Random(req.seed) if req.seed is not None else random.Random()
    picked = rng.sample(items, n)

    new_ds = Dataset(
        name=req.name,
        description=req.description or f"从 #{src.id} ({src.name}) 随机切分 {n}/{len(items)} 条",
        format=src.format,
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)

    for it in picked:
        db.add(DatasetItem(
            dataset_id=new_ds.id,
            knowledge_item_id=it.knowledge_item_id,
            instruction=it.instruction,
            input=it.input,
            output=it.output,
            system_prompt=it.system_prompt,
        ))
    new_ds.item_count = n
    new_ds.status = "ready"
    db.commit()
    db.refresh(new_ds)
    return _serialize_dataset(new_ds)
