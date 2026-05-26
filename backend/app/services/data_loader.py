import os
import glob
from pathlib import Path
from sqlalchemy.orm import Session
from ..models.models import Document
from ..config import settings


def _upsert_document(db: Session, doc_type: str, title: str, path: str, content: str, metadata: dict):
    existing = db.query(Document).filter(Document.source_path == path).first()
    if existing:
        return existing
    doc = Document(type=doc_type, title=title, source_path=path, content=content, doc_metadata=metadata)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def load_guidelines(db: Session) -> int:
    base = Path(settings.guideline_data_path)
    if not base.exists():
        return 0
    count = 0
    for md_file in base.rglob("*.md"):
        rel = str(md_file.relative_to(base))
        title = md_file.stem
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        parts = rel.split(os.sep)
        specialty = parts[0] if len(parts) > 1 else "未分类"
        _upsert_document(db, "guideline", title, str(md_file), content, {"specialty": specialty, "filename": md_file.name})
        count += 1
    return count


def load_cases(db: Session) -> int:
    import pandas as pd
    base = Path(settings.crawler_data_path)
    if not base.exists():
        return 0
    count = 0
    for xlsx in base.glob("*_cleaned.xlsx"):
        try:
            df = pd.read_excel(xlsx, engine="openpyxl")
        except Exception:
            continue
        for _, row in df.iterrows():
            title = str(row.get("标题", ""))[:490] or "未命名病例"
            content = str(row.get("正文内容", ""))
            if not content or content == "nan":
                continue
            path = f"{xlsx}::{title}"
            _upsert_document(db, "case_report", title, path, content, {
                "keywords": str(row.get("关键词", "")),
                "published_at": str(row.get("发布时间", "")),
                "source_url": str(row.get("链接", "")),
            })
            count += 1
    return count
