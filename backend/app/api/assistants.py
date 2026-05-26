"""LLM Assistant CRUD + lifecycle endpoints."""
import subprocess
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..models.models import LLMAssistant, TrainingExperiment
from ..services.vllm_manager import vllm_manager
from ..services.llm_client import is_local_endpoint

router = APIRouter(prefix="/assistants", tags=["assistants"])


# ─────────────────────────────── GPU info ────────────────────────────────
# Mirrors /api/training/gpu-info but lives here so the assistant UI stays
# self-contained and we don't introduce a cross-page coupling.

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
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_pct": int(parts[5]),
                        "temperature_c": int(parts[6]),
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"gpus": gpus, "cuda_available": len(gpus) > 0}


# ─────────────────────────────── Pydantic schemas ─────────────────────────

class CreateAssistantRequest(BaseModel):
    name: str
    type: str                                # 'local' | 'remote'
    description: Optional[str] = None
    # remote
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    # local
    model_path: Optional[str] = None
    max_model_len: Optional[int] = None
    extra_vllm_args: Optional[List[str]] = None
    lora_adapter_path: Optional[str] = None
    gpu_ids: Optional[List[int]] = None       # None=auto, [0]=single, [0,1]=multi
    source_experiment_id: Optional[int] = None


class UpdateAssistantRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    max_model_len: Optional[int] = None
    extra_vllm_args: Optional[List[str]] = None
    lora_adapter_path: Optional[str] = None
    gpu_ids: Optional[List[int]] = None


def _serialize(a: LLMAssistant, include_secrets: bool = False) -> dict:
    return {
        "id": a.id,
        "name": a.name,
        "type": a.type,
        "description": a.description,
        "base_url": a.base_url,
        "model_name": a.model_name,
        "has_api_key": bool(a.api_key),
        # only return key when explicitly asked (e.g., admin edit)
        "api_key": a.api_key if include_secrets else None,
        "model_path": a.model_path,
        "max_model_len": a.max_model_len,
        "extra_vllm_args": a.extra_vllm_args or [],
        "lora_adapter_path": a.lora_adapter_path,
        "gpu_ids": a.gpu_ids,
        "port": a.port,
        "process_pid": a.process_pid,
        "status": a.status,
        "error_message": a.error_message,
        "log_file": a.log_file,
        "source_experiment_id": a.source_experiment_id,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _normalize_gpu_ids(raw):
    """Validate / coerce gpu_ids payload. Returns None or list[int]."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise HTTPException(400, "gpu_ids 必须是数组（或不传以使用自动模式）")
    try:
        ids = [int(g) for g in raw]
    except (TypeError, ValueError):
        raise HTTPException(400, "gpu_ids 中包含非整数")
    if any(g < 0 for g in ids):
        raise HTTPException(400, "gpu_ids 不允许负数")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "gpu_ids 不允许重复")
    # Empty list isn't meaningful for vllm (it needs at least one GPU).
    return ids if ids else None


def _validate(req, existing: Optional[LLMAssistant] = None):
    """Common create/update validation. Mutates `existing` if given."""
    typ = req.type if hasattr(req, "type") else (existing.type if existing else None)
    if typ not in ("local", "remote"):
        raise HTTPException(400, "type 必须为 'local' 或 'remote'")
    if typ == "remote":
        base_url = (req.base_url or (existing.base_url if existing else "") or "").strip()
        api_key = (req.api_key or (existing.api_key if existing else "") or "").strip()
        if not base_url:
            raise HTTPException(400, "远程助手必须填写 base_url")
        if not api_key and not is_local_endpoint(base_url):
            raise HTTPException(400, "远程 base_url 必须提供 api_key（仅 localhost 可空）")
    else:
        model_path = (req.model_path or (existing.model_path if existing else "") or "").strip()
        if not model_path:
            raise HTTPException(400, "本地助手必须填写 model_path")
    model_name = (req.model_name or (existing.model_name if existing else "") or "").strip()
    if not model_name:
        raise HTTPException(400, "必须填写 model_name (即 vllm 的 served-model-name)")


# ─────────────────────────────── CRUD ─────────────────────────────────────

@router.get("")
def list_assistants(db: Session = Depends(get_db)):
    rows = db.query(LLMAssistant).order_by(LLMAssistant.id.desc()).all()
    # liveness reconciliation — cheap, only checks subprocess presence
    for a in rows:
        vllm_manager.liveness_check(a, SessionLocal)
    rows = db.query(LLMAssistant).order_by(LLMAssistant.id.desc()).all()
    return [_serialize(a) for a in rows]


@router.post("")
def create_assistant(req: CreateAssistantRequest, db: Session = Depends(get_db)):
    _validate(req)
    if db.query(LLMAssistant).filter(LLMAssistant.name == req.name).first():
        raise HTTPException(400, f"已存在同名助手：{req.name}")

    if req.type == "remote":
        # For remote assistants, base_url + model_name + api_key are stored directly
        # and used at call-time; no process to start.
        a = LLMAssistant(
            name=req.name, type="remote",
            description=req.description,
            base_url=(req.base_url or "").strip(),
            model_name=(req.model_name or "").strip(),
            api_key=(req.api_key or "").strip() or None,
            status="running",  # remote assistants are always "ready"
        )
    else:
        a = LLMAssistant(
            name=req.name, type="local",
            description=req.description,
            model_path=(req.model_path or "").strip(),
            model_name=(req.model_name or "").strip(),
            max_model_len=req.max_model_len,
            extra_vllm_args=req.extra_vllm_args or [],
            lora_adapter_path=(req.lora_adapter_path or "").strip() or None,
            gpu_ids=_normalize_gpu_ids(req.gpu_ids),
            source_experiment_id=req.source_experiment_id,
            status="stopped",
        )
    db.add(a)
    db.commit()
    db.refresh(a)
    return _serialize(a)


@router.get("/{assistant_id}")
def get_assistant(assistant_id: int, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    vllm_manager.liveness_check(a, SessionLocal)
    db.refresh(a)
    return _serialize(a)


@router.patch("/{assistant_id}")
def update_assistant(assistant_id: int, req: UpdateAssistantRequest, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    if a.type == "local" and a.status in ("running", "starting"):
        raise HTTPException(400, "请先停止本地助手再修改配置")

    if req.name is not None:
        # uniqueness on rename
        existing = db.query(LLMAssistant).filter(
            LLMAssistant.name == req.name, LLMAssistant.id != assistant_id
        ).first()
        if existing:
            raise HTTPException(400, f"已存在同名助手：{req.name}")
        a.name = req.name
    for f in ("description", "base_url", "model_name", "model_path",
              "max_model_len", "lora_adapter_path"):
        v = getattr(req, f)
        if v is not None:
            setattr(a, f, v.strip() if isinstance(v, str) else v)
    if req.api_key is not None:
        a.api_key = (req.api_key or "").strip() or None
    if req.extra_vllm_args is not None:
        a.extra_vllm_args = req.extra_vllm_args
    if req.gpu_ids is not None:
        a.gpu_ids = _normalize_gpu_ids(req.gpu_ids)

    _validate(req, existing=a)
    db.commit()
    db.refresh(a)
    return _serialize(a)


@router.delete("/{assistant_id}")
def delete_assistant(assistant_id: int, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    if a.type == "local" and a.status in ("running", "starting"):
        vllm_manager.stop(assistant_id, SessionLocal)
    db.delete(a)
    db.commit()
    return {"ok": True}


# ─────────────────────────────── lifecycle ────────────────────────────────

@router.post("/{assistant_id}/start")
def start_assistant(assistant_id: int, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    if a.type != "local":
        raise HTTPException(400, "远程助手无需启动")
    if a.status in ("running", "starting"):
        raise HTTPException(400, f"助手当前状态：{a.status}")
    try:
        port = vllm_manager.start(a, SessionLocal)
    except Exception as e:
        raise HTTPException(400, str(e))
    db.refresh(a)
    return {"ok": True, "port": port, **_serialize(a)}


@router.post("/{assistant_id}/stop")
def stop_assistant(assistant_id: int, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    if a.type != "local":
        raise HTTPException(400, "远程助手无需停止")
    vllm_manager.stop(assistant_id, SessionLocal)
    db.refresh(a)
    return _serialize(a)


@router.get("/{assistant_id}/log")
def get_assistant_log(assistant_id: int, tail: int = 200, db: Session = Depends(get_db)):
    a = db.query(LLMAssistant).filter_by(id=assistant_id).first()
    if not a:
        raise HTTPException(404, "助手不存在")
    if not a.log_file:
        return {"lines": []}
    try:
        with open(a.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return {"lines": lines[-tail:]}
    except FileNotFoundError:
        return {"lines": []}


# ─────────────────────────────── shortcut ────────────────────────────────

class FromExperimentRequest(BaseModel):
    name: str
    description: Optional[str] = None
    max_model_len: Optional[int] = None


@router.post("/from-experiment/{exp_id}")
def create_from_experiment(exp_id: int, req: FromExperimentRequest, db: Session = Depends(get_db)):
    """Pre-fill a local assistant from a finished training experiment."""
    exp = db.query(TrainingExperiment).filter_by(id=exp_id).first()
    if not exp:
        raise HTTPException(404, "训练实验不存在")
    cfg = exp.config or {}
    output_dir = cfg.get("final_output_dir") or cfg.get("output_dir")
    if not output_dir:
        raise HTTPException(400, "实验尚未生成输出目录")

    if db.query(LLMAssistant).filter(LLMAssistant.name == req.name).first():
        raise HTTPException(400, f"已存在同名助手：{req.name}")

    use_lora = bool(cfg.get("use_lora"))

    if use_lora:
        # base model + LoRA adapter
        a = LLMAssistant(
            name=req.name,
            type="local",
            description=req.description or f"由训练实验 #{exp_id} ({exp.name}) 生成",
            model_path=exp.base_model,
            model_name=req.name.replace(" ", "_"),
            lora_adapter_path=output_dir,
            max_model_len=req.max_model_len or cfg.get("max_seq_length"),
            source_experiment_id=exp_id,
            status="stopped",
            extra_vllm_args=[],
        )
    else:
        # full model dir
        a = LLMAssistant(
            name=req.name,
            type="local",
            description=req.description or f"由训练实验 #{exp_id} ({exp.name}) 生成",
            model_path=output_dir,
            model_name=req.name.replace(" ", "_"),
            max_model_len=req.max_model_len or cfg.get("max_seq_length"),
            source_experiment_id=exp_id,
            status="stopped",
            extra_vllm_args=[],
        )
    db.add(a)
    db.commit()
    db.refresh(a)
    return _serialize(a)
