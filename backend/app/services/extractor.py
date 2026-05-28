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


CASE_PROMPT = """你是一位资深临床医师与医学AI训练数据合成专家。请基于下述真实临床病例报告，产出**1 条**面向单轮诊断训练的 DiagnosticInstance 样本，以 JSON 返回。

【病例原文】
{content}

【脱敏铁律】
1. 删除所有具体姓名（患者/家属/医师），统一替换为：患者/家属/主管医师/主治医师/会诊医师。
2. 删除医院、科室、地区等机构信息（必要时统一为"某三甲医院"）。
3. 删除身份证号、住院号、电话、住址等唯一标识。
4. 绝对日期改为相对时间（"入院前 3 天"、"治疗第 5 周"）；保留性别、年龄、职业类别。
5. 保留全部医学相关内容：症状、体征、生命体征、辅助检查、既往史、家族史、用药史、过敏史、手术史、治疗经过与转归。

【输出 JSON 严格格式（不要 Markdown、不要任何 JSON 外文字）】
{{
  "presentation": "脱敏后完整的患者就诊场景描述。必须包含：主诉 + 现病史 + 关键既往史 + 关键体征 + 已完成的辅助检查结果。这段文字将直接作为模型的输入，要求脱离原文也能完整理解病情。≥200 字。",
  "diagnosis": "该病例最终诊断的具体疾病名称（如：急性ST段抬高型心肌梗死）；不要使用模糊指代（"心脏病"、"该病"）。",
  "specialty": "学科分类（心血管 / 呼吸 / 消化 / 神经 / 内分泌 / 肾脏 / 血液 / 风湿免疫 / 感染 / 妇产 / 儿科 / 急诊 / 肿瘤 / 其他）。",
  "answer": "结构化诊断推理过程，作为模型的输出。必须严格按以下编号格式：\\n1. 病情归纳：…（提炼关键症状/体征/检查异常）\\n2. 鉴别诊断：列出至少 3 个候选疾病，每个简述支持/反对依据\\n3. 关键鉴别要点：哪些检查/体征能区分这些候选\\n4. 倾向诊断：[与 diagnosis 字段一致的具体疾病名]\\n5. 进一步建议：所需补充的检查或治疗考虑"
}}

【硬性质量要求】
A. presentation 必须是完整可独立理解的临床场景，不得出现"如上"、"该患者"作为脱离上下文的引用。
B. answer 必须基于原病例事实推理，不得编造原文未提及的检查值。
C. 鉴别诊断必须列出至少 3 项，覆盖至少 1 个常见的混淆诊断。
D. 严格 JSON，所有字符串用双引号。"""


GUIDELINE_PROMPT = """你是一位资深临床医师与医学AI训练数据合成专家。请基于下述临床指南/共识，合成 **{n_patients} 个不同的虚拟患者就诊场景**作为 DiagnosticInstance 训练样本，以 JSON 返回。

【指南内容】
{content}

【合成原则 — 必须遵守】
1. 先从指南中提炼出诊断标准、临床特征、典型表现，再据此构造**满足该诊断的真实感虚拟患者**。
2. 多个患者之间必须在以下维度有显著差异（每个维度至少覆盖 2 种）：
   - 年龄（青年 / 中年 / 老年）
   - 性别
   - 严重度（轻度 / 中度 / 重度 / 危重）
   - 合并症（无 / 高血压 / 糖尿病 / 慢阻肺 / 慢性肾病等）
   - 病程（急性发作 / 亚急性 / 慢性）
3. 患者描述必须自洽：症状、体征、检查结果之间符合医学逻辑，避免出现不能并存的描述。
4. 不要使用真实姓名/医院，参照"患者"、"某三甲医院"格式。

【输出 JSON 严格格式（不要 Markdown、不要任何 JSON 外文字）】
{{
  "disease": "指南所涉及的具体疾病名称",
  "specialty": "学科分类（心血管 / 呼吸 / 消化 / 神经 / 内分泌 / 肾脏 / 血液 / 风湿免疫 / 感染 / 妇产 / 儿科 / 急诊 / 肿瘤 / 其他）",
  "patients": [
    {{
      "presentation": "完整虚拟患者就诊场景：年龄/性别 + 主诉 + 现病史 + 既往史 + 关键体征 + 已完成的检查结果。≥180 字。",
      "severity": "轻 / 中 / 重 / 危重",
      "answer": "结构化诊断推理：\\n1. 病情归纳：…\\n2. 鉴别诊断：列出 ≥3 个候选并说明依据\\n3. 关键鉴别要点：…\\n4. 倾向诊断：[与 disease 字段一致]\\n5. 进一步建议：…（结合指南推荐的检查与治疗）"
    }}
  ]
}}

【硬性质量要求】
A. patients 数组长度必须 = {n_patients}（信息不足以支持时可少 1~2 个，但不得编造）。
B. 每个 presentation 都必须可独立理解，不得引用其他患者。
C. answer 必须 5 步全部出现，且"4. 倾向诊断"的疾病名要与 disease 字段一致。
D. patients 之间的 presentation 不得高度雷同，必须体现合成原则 #2 中的差异维度。
E. 严格 JSON，所有字符串用双引号。"""


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
  "specialty": "学科分类（同 case_extract 取值集）",
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
    """Case report → exactly ONE DiagnosticInstance (full presentation + reasoning answer).

    Replaces the old `qa_pairs` flavor. The new CASE_PROMPT returns a single
    {presentation, diagnosis, specialty, answer} object that maps 1:1 onto
    DiagnosticInstance, which is the shape the model is tested on.
    """
    template = prompt_template or CASE_PROMPT
    content = (doc.content or "")[:6000]   # cases can be long; allow more context
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key, max_tokens=2500)

    presentation = (extracted.get("presentation") or "").strip()
    answer = (extracted.get("answer") or "").strip()
    if not presentation or not answer:
        raise ValueError("LLM 返回缺少 presentation 或 answer")

    diagnosis_label = _normalize_label(extracted.get("diagnosis", ""))
    specialty = (extracted.get("specialty")
                 or ((doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None))

    db.add(DiagnosticInstance(
        presentation=presentation,
        answer=answer,
        diagnosis_label=diagnosis_label,
        specialty=specialty,
        synthesis_strategy="case_direct",
        source_doc_id=doc.id,
        job_id=job.id,
    ))


def _process_guideline_synth(doc, prompt_template, job, model_name, base_url, api_key, db):
    """Guideline → N virtual-patient DiagnosticInstances satisfying the guideline's criteria.

    N is taken from job.config.n_per_doc (default 8). Patients differ in age,
    sex, severity, comorbidity per the prompt's synthesis rules.
    """
    cfg = (job.config or {}) if isinstance(job.config, dict) else {}
    n_per_doc = int(cfg.get("n_per_doc", 8))
    n_per_doc = max(2, min(30, n_per_doc))    # clamp to a sane range

    template = prompt_template or GUIDELINE_PROMPT
    content = (doc.content or "")[:5000]
    prompt = (template
              .replace("{n_patients}", str(n_per_doc))
              .replace("{content}", content))
    # Output can be very long when N is large; budget tokens accordingly.
    max_tok = 1500 + 800 * n_per_doc
    extracted = _call_llm(prompt, model_name, base_url, api_key, max_tokens=max_tok)

    disease = extracted.get("disease", "")
    diagnosis_label = _normalize_label(disease)
    specialty = (extracted.get("specialty")
                 or ((doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None))

    patients = extracted.get("patients", []) or []
    if not isinstance(patients, list) or not patients:
        raise ValueError("LLM 返回中缺少有效的 patients 数组")

    saved = 0
    for p in patients:
        presentation = (p.get("presentation") or "").strip()
        answer = (p.get("answer") or "").strip()
        if not presentation or not answer:
            continue
        db.add(DiagnosticInstance(
            presentation=presentation,
            answer=answer,
            diagnosis_label=diagnosis_label,
            specialty=specialty,
            synthesis_strategy="guideline_synth",
            source_doc_id=doc.id,
            job_id=job.id,
        ))
        saved += 1
    if saved == 0:
        raise ValueError("LLM 返回的 patients 数组没有任何含 presentation+answer 的有效项")


def _process_case_reasoning(doc, prompt_template, job, model_name, base_url, api_key, db):
    """Case report → 2~4 high-diversity scenario DiagnosticInstances (richer than case_extract)."""
    template = prompt_template or CLINICAL_REASONING_PROMPT
    content = (doc.content or "")[:8000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key, max_tokens=4000)

    samples = extracted.get("training_samples", []) or []
    if not isinstance(samples, list) or not samples:
        raise ValueError("LLM 返回中缺少有效的 training_samples 数组")

    diagnosis_label = _normalize_label(extracted.get("primary_diagnosis", ""))
    specialty = (extracted.get("specialty")
                 or ((doc.doc_metadata or {}).get("specialty") if doc.doc_metadata else None))

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

    # augment jobs operate on instances, not documents — different runner.
    if (job.task_type or "") == "augment":
        from .augmenter import run_augment_job
        run_augment_job(job_id, db)
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
