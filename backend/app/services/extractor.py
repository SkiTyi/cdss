"""Knowledge extraction runner.

Phase 1 (Step 1.1): unified output to DiagnosticInstance.

Three task types map to upstream document types:
  * case_extract     — case report → 1 instance per generated QA / reasoning sample
  * guideline_synth  — guideline doc → 1 instance per generated QA pair
                       (Step 1.2 will replace this with N-virtual-patient synthesis)
  * case_reasoning   — case report → 1 instance per clinical-reasoning scenario sample

Prompts here are placeholder versions kept from before the refactor — Step 1.2
will rewrite them to produce presentation/answer pairs natively. For now we
adapt their output into the DiagnosticInstance shape.
"""
import json
import re
import httpx
from sqlalchemy.orm import Session
from ..models.models import ExtractionJob, DiagnosticInstance, Document
from ..config import settings
from datetime import datetime


CASE_PROMPT = """你是一位医学知识提取专家。请从以下临床病例报告中提取结构化医学知识，以JSON格式返回。

病例内容：
{content}

请返回如下JSON（不要包含任何其他文字）：
{{
  "chief_complaint": "主诉",
  "symptoms": ["症状1", "症状2"],
  "diagnosis": "最终诊断（具体疾病名称）",
  "differential_diagnosis": ["鉴别诊断1"],
  "treatment": "治疗方案",
  "reasoning_chain": "诊断推理过程",
  "qa_pairs": [
    {{"question": "以[具体症状]为主诉的患者，最可能的诊断是什么？", "answer": "诊断为[具体疾病名称]，依据是..."}},
    {{"question": "[具体疾病名称]的标准治疗方案是什么？", "answer": "治疗方案包括..."}}
  ]
}}

重要要求：
1. qa_pairs 中的 question 和 answer 必须包含具体的疾病名称、症状或检查结果，禁止使用"该疾病"、"该患者"、"此病"等模糊指代词
2. 每个问题应当是独立可理解的，即脱离原文也能明白问的是什么
3. 至少生成3个qa_pairs，覆盖诊断、治疗、鉴别诊断等不同角度"""

GUIDELINE_PROMPT = """你是一位医学知识提取专家。请从以下临床指南/共识中提取结构化医学知识，以JSON格式返回。

指南内容：
{content}

请返回如下JSON（不要包含任何其他文字）：
{{
  "disease": "疾病名称",
  "diagnostic_criteria": ["诊断标准1", "诊断标准2"],
  "clinical_features": ["临床特征1"],
  "treatment_protocols": ["治疗方案1"],
  "key_points": ["关键知识点1"],
  "qa_pairs": [
    {{"question": "[具体疾病名称]的诊断标准是什么？", "answer": "[具体疾病名称]的诊断标准包括..."}},
    {{"question": "[具体疾病名称]的一线治疗方案是什么？", "answer": "[具体疾病名称]的治疗包括..."}}
  ]
}}

重要要求：
1. 首先从指南内容中识别出具体疾病名称，然后在所有 qa_pairs 的 question 和 answer 中用该具体名称替换占位符
2. 禁止使用"该疾病"、"此病"、"本病"等模糊指代词，每个问答必须独立可理解
3. 至少生成4个qa_pairs，覆盖诊断标准、临床特征、治疗方案、预防或预后等不同角度"""


CLINICAL_REASONING_PROMPT = """你是一位资深临床医师与医学AI数据合成专家，正在为训练临床辅助诊断大模型构建高质量监督学习数据集。请根据以下完整的真实临床病例报告，合成 2-4 条**脱敏的、结构化的、临床推理风格**的训练样本，以 JSON 格式返回。

【病例原文】
{content}

【脱敏铁律 — 全部样本中均严格遵守】
1. 删除所有具体姓名（患者、家属、医师、签名等），统一替换为：患者 / 家属 / 主管医师 / 主治医师 / 会诊医师。
2. 删除医院、科室、地区等机构信息，必要时统一为"某三甲医院"、"某专科门诊"。
3. 删除身份证号、住院号、电话、住址、邮箱等任何唯一标识。
4. 删除绝对日期，替换为相对时间（如"入院前 3 天"、"治疗第 5 周"、"近 2 月"）；保留患者性别、年龄、职业类别等流行病学相关信息。
5. 保留所有医学相关内容：症状、体征、生命体征、辅助检查（实验室/影像/病理）、既往史、家族史、用药史、过敏史、手术史、治疗经过与转归。

【输出 JSON 严格格式（不要输出任何其他文字、不要 Markdown 代码块）】
{{
  "case_summary": "对该病例的一句话脱敏总结（疾病/关键体征/治疗结局），≤80字",
  "primary_diagnosis": "该病例最主要的诊断（具体疾病名称）",
  "training_samples": [
    {{
      "scenario_type": "diagnosis_reasoning",
      "question": "脱敏后的患者就诊场景描述（含主诉、现病史、既往史、关键体征与必要检查结果），≥150字",
      "answer": "结构化推理过程：\\n1. 病情归纳：…\\n2. 鉴别诊断：列出至少3种可能的疾病并说明依据\\n3. 关键鉴别要点：…\\n4. 倾向诊断：…（明确写出具体疾病名）\\n5. 进一步建议：所需补充的检查或观察"
    }},
    {{
      "scenario_type": "differential_diagnosis",
      "question": "已知患者初步表现及部分检查（脱敏），请进行鉴别诊断分析。≥120字",
      "answer": "针对每个候选诊断分别分析：支持点 / 反对点 / 关键鉴别检查；最后给出最可能诊断及理由。"
    }},
    {{
      "scenario_type": "treatment_planning",
      "question": "已确诊为[具体疾病名]的患者（脱敏简述病情、合并症、过敏史、关键化验值），请给出个体化治疗方案。",
      "answer": "1. 一线治疗：药物名/剂量范围/用法（避免极端精确剂量，可写常用剂量区间）\\n2. 合并症与禁忌处理\\n3. 监测与随访指标\\n4. 健康教育与生活方式建议\\n5. 预后与复查计划"
    }}
  ]
}}

【硬性质量要求】
A. 每条 question 都必须是独立可理解的完整临床场景，不能出现"如上"、"该患者"在没有上下文时单独使用。
B. 答案必须基于原病例事实推理，不要捏造原文未提及的检查值或既往史。
C. 推理链必须显式列出"依据→结论"的因果关系。
D. 至少生成 2 条样本，最多 4 条；信息不足以支持某 scenario_type 时可省略而非编造。
E. 严格符合 JSON 语法，所有字符串使用双引号；不要在 JSON 外输出任何文字。"""


# ─────────────────────────── helpers ──────────────────────────────────────

def _is_local_endpoint(url: str) -> bool:
    if not url:
        return False
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def _normalize_label(s: str) -> str:
    """Lowercase + strip whitespace/punctuation for diagnosis_label.

    Used as a sampling key only. Different surface forms ("急性心梗" vs
    "急性心肌梗死") will NOT be unified by this function — Step 1.2 will add
    LLM-based canonicalization. For now we just collapse whitespace and
    strip trailing periods/quotes so that "急性心梗。" and " 急性心梗" go to
    the same bucket.
    """
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.strip(".。'\"`,，;；:：")
    return s[:200]


def _call_llm(prompt: str, model: str, base_url: str, api_key: str,
              max_tokens: int = 2000) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{base_url.rstrip('/')}/chat/completions",
                           headers=headers, json=payload)
        resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


# ─────────────────────────── per-doc processors ───────────────────────────

def _process_case_extract(doc, prompt_template, job, model_name, base_url, api_key, db):
    template = prompt_template or CASE_PROMPT
    content = (doc.content or "")[:3000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key)

    diagnosis_label = _normalize_label(extracted.get("diagnosis", ""))
    specialty = (doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None

    qa_pairs = extracted.get("qa_pairs", []) or []
    saved = 0
    for qa in qa_pairs:
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q or not a:
            continue
        db.add(DiagnosticInstance(
            presentation=q,
            answer=a,
            diagnosis_label=diagnosis_label,
            specialty=specialty,
            synthesis_strategy="case_direct",
            source_doc_id=doc.id,
            job_id=job.id,
        ))
        saved += 1
    if saved == 0:
        raise ValueError("LLM 未返回有效 qa_pairs")


def _process_guideline_synth(doc, prompt_template, job, model_name, base_url, api_key, db):
    template = prompt_template or GUIDELINE_PROMPT
    content = (doc.content or "")[:3000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key)

    diagnosis_label = _normalize_label(extracted.get("disease", ""))
    specialty = (doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None

    qa_pairs = extracted.get("qa_pairs", []) or []
    saved = 0
    for qa in qa_pairs:
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q or not a:
            continue
        db.add(DiagnosticInstance(
            presentation=q,
            answer=a,
            diagnosis_label=diagnosis_label,
            specialty=specialty,
            synthesis_strategy="guideline_synth",
            source_doc_id=doc.id,
            job_id=job.id,
        ))
        saved += 1
    if saved == 0:
        raise ValueError("LLM 未返回有效 qa_pairs")


def _process_case_reasoning(doc, prompt_template, job, model_name, base_url, api_key, db):
    template = prompt_template or CLINICAL_REASONING_PROMPT
    content = (doc.content or "")[:8000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key, max_tokens=4000)

    samples = extracted.get("training_samples", []) or []
    if not isinstance(samples, list) or not samples:
        raise ValueError("LLM 返回中缺少有效的 training_samples 数组")

    diagnosis_label = _normalize_label(extracted.get("primary_diagnosis", ""))
    specialty = (doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None

    saved = 0
    for sample in samples:
        q = (sample.get("question") or "").strip()
        a = (sample.get("answer") or "").strip()
        if not q or not a:
            continue
        db.add(DiagnosticInstance(
            presentation=q,
            answer=a,
            diagnosis_label=diagnosis_label,
            specialty=specialty,
            synthesis_strategy="case_direct",
            source_doc_id=doc.id,
            job_id=job.id,
        ))
        saved += 1
    if saved == 0:
        raise ValueError("LLM 未返回任何含 question+answer 的样本")


_PROCESSORS = {
    "case_extract":    _process_case_extract,
    "guideline_synth": _process_guideline_synth,
    "case_reasoning":  _process_case_reasoning,
}


# ─────────────────────────── job orchestrator ─────────────────────────────

def run_extraction_job(job_id: int, db: Session):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        return

    job.status = "running"
    job.started_at = datetime.utcnow()
    db.commit()

    # ── resolve LLM parameters ────────────────────────────────────────────
    base_url = ""
    model_name = ""
    api_key = ""
    if job.assistant_id:
        from ..models.models import LLMAssistant
        from .llm_client import resolve_assistant
        assistant = db.query(LLMAssistant).filter_by(id=job.assistant_id).first()
        try:
            cfg = resolve_assistant(assistant)
            base_url, model_name, api_key = cfg["base_url"], cfg["model_name"], cfg["api_key"]
        except ValueError as e:
            job.status = "failed"
            job.error_message = f"指定助手不可用：{e}"
            job.completed_at = datetime.utcnow()
            db.commit()
            return
    else:
        base_url = (job.base_url or settings.llm_api_base or "").strip()
        model_name = (job.model or settings.llm_model or "").strip()
        api_key = (job.api_key or settings.llm_api_key or "").strip()

    if not base_url:
        _fail(db, job, "未配置 LLM base_url")
        return
    if not model_name:
        _fail(db, job, "未配置 LLM model name")
        return
    if not api_key and not _is_local_endpoint(base_url):
        _fail(db, job, "远程 LLM 服务必须提供 api_key（仅 localhost 可省略）")
        return

    task_type = (job.task_type or "case_extract").strip()
    processor = _PROCESSORS.get(task_type)
    if not processor:
        _fail(db, job, f"未知 task_type: {task_type}")
        return

    # ── document selection ────────────────────────────────────────────────
    try:
        query = db.query(Document)
        # task_type narrows the document pool to its natural source type.
        if task_type == "guideline_synth":
            query = query.filter(Document.type == "guideline")
        elif task_type in ("case_extract", "case_reasoning"):
            query = query.filter(Document.type == "case_report")
        elif job.document_type and job.document_type != "all":
            query = query.filter(Document.type == job.document_type)

        if job.doc_limit and job.doc_limit > 0:
            docs = query.limit(job.doc_limit).all()
        else:
            docs = query.all()

        job.total_docs = len(docs)
        db.commit()

        for doc in docs:
            db.refresh(job)
            if job.is_cancelled:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                db.commit()
                return

            try:
                processor(doc, job.prompt_template, job, model_name, base_url, api_key, db)
                doc.status = "extracted"
                job.processed_docs += 1
            except Exception:
                job.failed_docs += 1

            db.commit()

        job.status = "completed"
        job.completed_at = datetime.utcnow()
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)

    db.commit()


def _fail(db, job, msg: str):
    job.status = "failed"
    job.error_message = msg
    job.completed_at = datetime.utcnow()
    db.commit()
