import json
import httpx
from sqlalchemy.orm import Session
from ..models.models import ExtractionJob, KnowledgeItem, Document
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
    }},
    {{
      "scenario_type": "examination_decision",
      "question": "面对该患者目前的临床表现（脱敏），需要做哪些进一步辅助检查以明确诊断？请说明优先级与目的。",
      "answer": "按优先级列出：检查项目 → 目的 → 预期结果 → 对决策的影响。覆盖至少 3 项。"
    }}
  ]
}}

【硬性质量要求】
A. 每条 question 都必须是独立可理解的完整临床场景，不能出现"如上"、"该患者"在没有上下文时单独使用。
B. 答案必须基于原病例事实推理，不要捏造原文未提及的检查值或既往史。
C. 推理链必须显式列出"依据→结论"的因果关系，鼓励出现"考虑/支持/不支持/需鉴别"等临床思维表达。
D. 至少生成 2 条样本，最多 4 条；如果原病例信息不足以支持某 scenario_type，可省略该条而非编造。
E. 严格符合 JSON 语法，所有字符串使用双引号；不要在 JSON 外输出任何文字。"""


def _is_local_endpoint(url: str) -> bool:
    if not url:
        return False
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def _call_llm(prompt: str, model: str, base_url: str, api_key: str,
              max_tokens: int = 2000) -> dict:
    headers = {"Content-Type": "application/json"}
    # Local endpoints (e.g. ollama, vllm) typically don't require auth.
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
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        # clean trailing fence
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


def _process_qa_extraction(doc, prompt_template, job, model_name, base_url, api_key, db):
    """Original behavior: extract structured medical knowledge + qa_pairs."""
    template = prompt_template or (
        CASE_PROMPT if doc.type == "case_report" else GUIDELINE_PROMPT
    )
    content = (doc.content or "")[:3000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key)

    qa_pairs = extracted.pop("qa_pairs", [])
    for qa in qa_pairs:
        db.add(KnowledgeItem(
            job_id=job.id,
            document_id=doc.id,
            knowledge_type="qa_pair",
            content=qa,
        ))
    # store the structured extraction as well
    db.add(KnowledgeItem(
        job_id=job.id,
        document_id=doc.id,
        knowledge_type="case_analysis" if doc.type == "case_report" else "guideline_summary",
        content=extracted,
    ))


def _process_clinical_reasoning(doc, prompt_template, job, model_name, base_url, api_key, db):
    """Synthesize de-identified clinical reasoning training samples."""
    template = prompt_template or CLINICAL_REASONING_PROMPT
    # Reasoning synthesis needs the full case body, not just first 3000 chars,
    # but cap to keep within typical context; also the answer is longer so
    # we lift max_tokens.
    content = (doc.content or "")[:8000]
    prompt = template.replace("{content}", content)
    extracted = _call_llm(prompt, model_name, base_url, api_key, max_tokens=4000)

    samples = extracted.get("training_samples", []) or []
    if not isinstance(samples, list) or not samples:
        raise ValueError("LLM 返回中缺少有效的 training_samples 数组")

    case_summary = extracted.get("case_summary", "")
    for sample in samples:
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        if not question or not answer:
            continue
        db.add(KnowledgeItem(
            job_id=job.id,
            document_id=doc.id,
            knowledge_type="clinical_reasoning",
            content={
                "scenario_type": sample.get("scenario_type", "clinical_reasoning"),
                "question": question,
                "answer": answer,
                "case_summary": case_summary,
            },
        ))


def run_extraction_job(job_id: int, db: Session):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        return

    job.status = "running"
    job.started_at = datetime.utcnow()
    db.commit()

    # ── resolve LLM parameters ────────────────────────────────────────────
    # Priority: configured assistant > job-level overrides > global .env
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
        job.status = "failed"
        job.error_message = "未配置 LLM base_url"
        job.completed_at = datetime.utcnow()
        db.commit()
        return
    if not model_name:
        job.status = "failed"
        job.error_message = "未配置 LLM model name"
        job.completed_at = datetime.utcnow()
        db.commit()
        return
    if not api_key and not _is_local_endpoint(base_url):
        job.status = "failed"
        job.error_message = "远程 LLM 服务必须提供 api_key（仅 localhost 可省略）"
        job.completed_at = datetime.utcnow()
        db.commit()
        return

    task_type = (job.task_type or "qa_extraction").strip()

    try:
        query = db.query(Document)
        if task_type == "clinical_reasoning_synthesis":
            # Reasoning synthesis is meaningful only on case reports.
            query = query.filter(Document.type == "case_report")
        elif job.document_type != "all":
            query = query.filter(Document.type == job.document_type)

        # 支持限制数量
        if job.doc_limit and job.doc_limit > 0:
            docs = query.limit(job.doc_limit).all()
        else:
            docs = query.all()

        job.total_docs = len(docs)
        db.commit()

        for doc in docs:
            # 每次处理前检查是否被取消
            db.refresh(job)
            if job.is_cancelled:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                db.commit()
                return

            try:
                if task_type == "clinical_reasoning_synthesis":
                    _process_clinical_reasoning(
                        doc, job.prompt_template, job, model_name, base_url, api_key, db)
                else:
                    _process_qa_extraction(
                        doc, job.prompt_template, job, model_name, base_url, api_key, db)
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
