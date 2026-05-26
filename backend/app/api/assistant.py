from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import re

from ..database import get_db
from ..models.models import Document, LLMAssistant
from ..services.llm_client import resolve_assistant, chat_completion, is_local_endpoint
from ..config import settings

router = APIRouter(prefix="/assistant", tags=["assistant"])

DIAGNOSIS_PROMPT = """你是一位经验丰富的临床医学专家。请根据以下患者病情描述，给出专业的诊断分析和治疗建议。

患者病情：
{case_description}

请提供：
1. 初步诊断（最可能的诊断）
2. 诊断依据（关键症状和体征分析）
3. 鉴别诊断（需要排除的疾病）
4. 建议检查（进一步明确诊断的检查）
5. 治疗方案（初步治疗建议）

请以结构化的方式回答，语言专业、清晰。"""


class DiagnoseRequest(BaseModel):
    case_description: str
    assistant_id: Optional[int] = None  # if set, use a configured assistant
    model: Optional[str] = None         # legacy: model name override (only when assistant_id is None)


def _resolve_call_params(req: DiagnoseRequest, db: Session) -> dict:
    """Pick base_url/model_name/api_key from assistant or fall back to global .env."""
    if req.assistant_id:
        a = db.query(LLMAssistant).filter_by(id=req.assistant_id).first()
        if not a:
            raise HTTPException(404, "助手不存在")
        try:
            return resolve_assistant(a)
        except ValueError as e:
            raise HTTPException(400, str(e))
    base_url = (settings.llm_api_base or "").strip()
    model_name = (req.model or settings.llm_model or "").strip()
    api_key = (settings.llm_api_key or "").strip()
    if not base_url or not model_name:
        raise HTTPException(400, "未配置默认 LLM 参数（请在 .env 或新建助手）")
    if not api_key and not is_local_endpoint(base_url):
        raise HTTPException(400, "默认远程 LLM 缺少 api_key")
    return {"base_url": base_url, "model_name": model_name, "api_key": api_key}


@router.post("/diagnose")
def diagnose(req: DiagnoseRequest, db: Session = Depends(get_db)):
    cfg = _resolve_call_params(req, db)
    prompt = DIAGNOSIS_PROMPT.replace("{case_description}", req.case_description)
    try:
        text = chat_completion(
            base_url=cfg["base_url"],
            model_name=cfg["model_name"],
            api_key=cfg["api_key"],
            prompt=prompt,
            temperature=0.3,
            max_tokens=2000,
        )
        return {"success": True, "result": {"raw": text}}
    except Exception as e:
        raise HTTPException(500, f"LLM 调用失败：{e}")


@router.get("/similar-cases")
def similar_cases(query: str, limit: int = 5, db: Session = Depends(get_db)):
    """Keyword-based case retrieval. Splits the query into Chinese n-grams
    (3-grams preferred, fallback 2-grams) and ranks documents by how many
    distinct keywords match in title or content. Whitespace-splitting (the
    old behavior) was useless for Chinese clinical text — Chinese words
    aren't space-separated, so the entire query collapsed into one string
    that almost never matched a title verbatim.
    """
    if not query or not query.strip():
        return []

    # Strip everything that isn't a Chinese character (keeps the search focused
    # on disease/symptom terms rather than punctuation/digits/English noise).
    text = re.sub(r"[^一-鿿]+", "", query[:500])
    if len(text) < 2:
        return []

    # Build candidate keywords: prefer longer n-grams (more specific), then
    # fall back to bigrams. Skip a few overly-common medical filler words.
    STOPGRAMS = {
        "患者", "病人", "目前", "进行", "出现", "情况", "考虑", "可能", "建议", "诊断",
        "治疗", "检查", "症状", "病史", "既往", "入院", "出院", "复查", "正常", "轻度",
        "本院", "门诊", "住院", "因为", "由于", "无明", "诉述", "诉发", "无明显",
    }
    candidates = []
    seen = set()
    for n in (3, 2):
        for i in range(len(text) - n + 1):
            kw = text[i:i + n]
            if kw in seen or kw in STOPGRAMS:
                continue
            seen.add(kw)
            candidates.append(kw)
            if len(candidates) >= 24:
                break
        if len(candidates) >= 24:
            break

    if not candidates:
        return []

    # Score each doc by # of distinct keywords matched (in title OR content).
    # Each keyword fetches up to 30 docs; we union and rank.
    doc_scores = {}
    for kw in candidates[:12]:
        rows = db.query(Document.id).filter(
            Document.type == "case_report",
            or_(Document.title.contains(kw), Document.content.contains(kw)),
        ).limit(40).all()
        for (doc_id,) in rows:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

    if not doc_scores:
        return []

    top_ids = sorted(doc_scores.items(), key=lambda x: -x[1])[:limit]
    docs = db.query(Document).filter(
        Document.id.in_([i for i, _ in top_ids])
    ).all()
    by_id = {d.id: d for d in docs}

    results = []
    for did, score in top_ids:
        d = by_id.get(did)
        if not d:
            continue
        snippet = (d.content or "").strip()
        # Anchor snippet on the first matched keyword for a more useful preview.
        if snippet and candidates:
            for kw in candidates:
                idx = snippet.find(kw)
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(snippet), idx + 140)
                    snippet = ("…" if start > 0 else "") + snippet[start:end] + ("…" if end < len(d.content or "") else "")
                    break
            else:
                snippet = snippet[:200] + ("…" if len(snippet) > 200 else "")
        results.append({
            "id": d.id,
            "title": d.title,
            "type": d.type,
            "score": score,
            "matched_keywords": [k for k in candidates if (d.title and k in d.title) or (d.content and k in d.content)][:5],
            "snippet": snippet,
        })
    return results
