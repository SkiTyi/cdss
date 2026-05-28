import json
import random
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
from pydantic import BaseModel
from ..database import get_db
from ..models.models import Dataset, DatasetItem, DiagnosticInstance

router = APIRouter(prefix="/datasets", tags=["datasets"])


# Sampling strategies for dataset construction.
#   none              — take all candidates (after filters), no rebalancing
#   proportional      — same as none + optional max_per_disease cap; preserves
#                       the natural disease distribution but caps head classes
#   uniform_by_disease — take min(max_per_disease or smallest_bucket, bucket_size)
#                       from every label; rebalances toward uniform across diseases
_SAMPLING_STRATEGIES = {"none", "proportional", "uniform_by_disease"}


class CreateDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None
    format: str = "alpaca"
    # Source selection — pick one:
    instance_ids: Optional[List[int]] = None    # explicit set
    job_id: Optional[int] = None                # all instances from a job
    approved_only: bool = False                 # restrict to is_approved=True
    # Phase 1 Step 1.4: filtering + sampling
    include_strategies: Optional[List[str]] = None  # whitelist of synthesis_strategy values
    sampling_strategy: str = "proportional"
    max_per_disease: Optional[int] = None
    seed: Optional[int] = None                  # for reproducible sampling
    system_prompt: Optional[str] = "你是一位专业的临床医学助手，请根据患者的病情描述给出专业的诊断分析和治疗建议。"


class PreviewSourceRequest(BaseModel):
    job_id: Optional[int] = None
    instance_ids: Optional[List[int]] = None
    approved_only: bool = False
    include_strategies: Optional[List[str]] = None
    sampling_strategy: str = "proportional"
    max_per_disease: Optional[int] = None
    top: int = 30   # how many diagnoses to surface in the head histogram


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


def _build_source_query(db, *, job_id=None, instance_ids=None,
                        approved_only=False, include_strategies=None):
    """Common filter assembly for both preview-source and create_dataset.

    Returns a SQLAlchemy query over DiagnosticInstance with the given
    filters applied. Caller is responsible for execution and any further
    in-memory sampling.
    """
    q = db.query(DiagnosticInstance)
    if instance_ids:
        q = q.filter(DiagnosticInstance.id.in_(instance_ids))
    elif job_id:
        q = q.filter(DiagnosticInstance.job_id == job_id)
    if approved_only:
        q = q.filter(DiagnosticInstance.is_approved == True)
    if include_strategies:
        q = q.filter(DiagnosticInstance.synthesis_strategy.in_(include_strategies))
    return q


def _apply_sampling(instances, strategy, max_per_disease, seed=None):
    """In-memory sampler. Groups by diagnosis_label then applies the policy.

    Empty string and None are merged into one "(未标注)" bucket so untagged
    rows don't silently get dropped or all collide into a giant bucket
    under different keys.
    """
    if strategy not in _SAMPLING_STRATEGIES:
        raise HTTPException(400, f"未知的 sampling_strategy：{strategy}")

    if strategy == "none":
        return list(instances)

    rng = random.Random(seed) if seed is not None else random.Random()

    buckets = defaultdict(list)
    for inst in instances:
        buckets[inst.diagnosis_label or "(未标注)"].append(inst)

    if not buckets:
        return []

    if strategy == "proportional":
        # Keep natural distribution; only cap each bucket if max_per_disease set.
        result = []
        for label, items in buckets.items():
            if max_per_disease and len(items) > max_per_disease:
                items = rng.sample(items, max_per_disease)
            result.extend(items)
        return result

    if strategy == "uniform_by_disease":
        # Take the same N from each bucket; if N not specified, use the
        # smallest bucket size (truly uniform), else cap each at min(N, size).
        if max_per_disease:
            target = max_per_disease
        else:
            target = min(len(items) for items in buckets.values())
        result = []
        for label, items in buckets.items():
            if len(items) > target:
                items = rng.sample(items, target)
            result.extend(items)
        return result

    return list(instances)  # unreachable


@router.post("/preview-source")
def preview_source(req: PreviewSourceRequest, db: Session = Depends(get_db)):
    """Show distribution + projected count BEFORE building a dataset.

    Returns:
      total_candidates:   rows passing the filters (before sampling)
      projected_count:    rows that WOULD end up in the dataset after sampling
      distinct_diagnoses: number of unique diagnosis_label values
      strategy_breakdown: {synthesis_strategy: count} over candidates
      head:               top-N diagnosis labels with (count, projected_count)
      singletons:         labels with only 1 candidate (long-tail signal)
    """
    q = _build_source_query(db,
        job_id=req.job_id, instance_ids=req.instance_ids,
        approved_only=req.approved_only,
        include_strategies=req.include_strategies)

    candidates = q.all()
    total_candidates = len(candidates)
    if total_candidates == 0:
        return {
            "total_candidates": 0, "projected_count": 0,
            "distinct_diagnoses": 0, "strategy_breakdown": {},
            "head": [], "singletons": 0,
        }

    # Strategy breakdown
    strategy_breakdown = defaultdict(int)
    label_count = defaultdict(int)
    for inst in candidates:
        strategy_breakdown[inst.synthesis_strategy or "(未知)"] += 1
        label_count[inst.diagnosis_label or "(未标注)"] += 1

    # Projected per-label counts under the requested sampling policy
    if req.sampling_strategy == "none":
        projected_per_label = dict(label_count)
    elif req.sampling_strategy == "proportional":
        cap = req.max_per_disease
        projected_per_label = {
            lbl: (min(n, cap) if cap else n) for lbl, n in label_count.items()
        }
    elif req.sampling_strategy == "uniform_by_disease":
        target = req.max_per_disease or min(label_count.values())
        projected_per_label = {lbl: min(n, target) for lbl, n in label_count.items()}
    else:
        raise HTTPException(400, f"未知的 sampling_strategy：{req.sampling_strategy}")

    projected_count = sum(projected_per_label.values())

    # Head histogram — top-N by candidate count
    sorted_labels = sorted(label_count.items(), key=lambda kv: kv[1], reverse=True)
    head = [
        {"label": lbl, "count": n, "projected_count": projected_per_label.get(lbl, 0)}
        for lbl, n in sorted_labels[:req.top]
    ]
    singletons = sum(1 for _, n in label_count.items() if n == 1)

    return {
        "total_candidates": total_candidates,
        "projected_count": projected_count,
        "distinct_diagnoses": len(label_count),
        "strategy_breakdown": dict(strategy_breakdown),
        "head": head,
        "singletons": singletons,
    }


@router.post("")
def create_dataset(req: CreateDatasetRequest, db: Session = Depends(get_db)):
    if req.sampling_strategy not in _SAMPLING_STRATEGIES:
        raise HTTPException(400, f"未知的 sampling_strategy：{req.sampling_strategy}")

    dataset = Dataset(name=req.name, description=req.description, format=req.format)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    candidates = _build_source_query(db,
        job_id=req.job_id, instance_ids=req.instance_ids,
        approved_only=req.approved_only,
        include_strategies=req.include_strategies).all()

    sampled = _apply_sampling(candidates, req.sampling_strategy,
                              req.max_per_disease, req.seed)

    count = 0
    for inst in sampled:
        presentation = (inst.presentation or "").strip()
        answer = (inst.answer or "").strip()
        if not presentation or not answer:
            continue
        di = DatasetItem(
            dataset_id=dataset.id,
            instance_id=inst.id,
            instruction=presentation,
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
            {"id": i.id, "instruction": i.instruction, "input": i.input,
             "output": i.output, "system_prompt": i.system_prompt,
             "instance_id": i.instance_id}
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
    sample_size: Optional[int] = None
    ratio: Optional[float] = None
    seed: Optional[int] = None


@router.post("/{dataset_id}/split")
def split_dataset(dataset_id: int, req: SplitDatasetRequest, db: Session = Depends(get_db)):
    src = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not src:
        raise HTTPException(404, "源数据集不存在")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    if not items:
        raise HTTPException(400, "源数据集没有条目，无法切分")

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
            instance_id=it.instance_id,
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
