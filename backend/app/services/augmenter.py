"""Augmentation strategies for DiagnosticInstance.

Each strategy takes a *base* DiagnosticInstance and produces one or more
*variant* DiagnosticInstances. Variants are persisted with:
  * parent_instance_id pointing back at the base
  * synthesis_strategy = 'aug_<strategy_name>'

Default strategies enabled for Phase 1 are paraphrase, distractor, and
cot_enrich — they directly improve generalization on the single-turn
diagnostic test set. hardneg is implemented but reserved as seed data for
Phase 2 (DPO); it's harmless to enable in SFT data but contributes less
than the other three.

Each strategy = independent LLM call. Failures of one strategy on one
instance do not block the rest.
"""
from __future__ import annotations

from datetime import datetime
import json
import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models.models import DiagnosticInstance, ExtractionJob, LLMAssistant


# ─────────────────────────── prompt library ───────────────────────────────

_PARAPHRASE = """你是一位医学文档改写专家。请改写下面的临床场景描述，**严格保持所有医学事实不变**（症状、体征、检查值、既往史、年龄、性别等），但用不同的语言风格表达，例如：
  - 从医生病程记录风格 → 改为患者自述风格
  - 从分点描述 → 改为段落叙述
  - 调整描述顺序
  - 使用同义词替换非关键词

不要改动答案。不要添加或删除任何医学事实。

原 presentation：
{presentation}

请直接输出改写后的 presentation 文本（≥150 字），不要 JSON、不要解释、不要 Markdown 包裹。"""


_DISTRACTOR = """你是一位医学训练数据增强专家。请在下面的临床场景描述中，**插入 1~2 条与最终诊断无关的既往史 / 家族史 / 既往手术 / 用药史**作为干扰信息，模拟真实病历中常见的无关信息噪音。

要求：
1. 插入的内容必须是真实可能的（如：30 年前阑尾切除史、远亲患糖尿病、青霉素过敏史等）
2. 必须不影响最终诊断（不能加入会改变诊断方向的信息）
3. 保持原描述的所有医学事实不变
4. 干扰信息应自然融入描述，而非生硬罗列

原 presentation：
{presentation}

原 answer 中的诊断方向（仅供你判断"无关"的标准，不要在输出中体现）：
{diagnosis}

请直接输出改写后的 presentation 文本（≥160 字），不要 JSON、不要解释、不要 Markdown 包裹。"""


_COT_ENRICH = """你是一位资深临床医师。下面这条诊断 answer 太简短，缺乏显式推理步骤。请扩展为包含完整推理链的答案，**严格保持最终诊断与原 answer 一致**。

要求按以下编号格式输出（5 步必须齐全）：
1. 病情归纳：从 presentation 中提炼关键症状/体征/检查异常
2. 鉴别诊断：列出至少 3 个候选疾病，每个简述支持/反对依据
3. 关键鉴别要点：哪些检查或体征能区分这些候选
4. 倾向诊断：明确写出诊断名称（必须与原 answer 一致）
5. 进一步建议：所需补充的检查或治疗考虑

原 presentation：
{presentation}

原 answer（最终诊断务必保持一致）：
{answer}

请直接输出扩展后的 answer 文本，不要 JSON、不要解释、不要 Markdown 包裹。"""


_HARDNEG = """你是一位医学训练数据合成专家。下面有一条临床诊断样本，诊断是 {diagnosis}。请基于**与之容易混淆的鉴别诊断方向**，构造一条**修改后的 presentation 与对应 answer**，使新场景应当被诊断为另一个具体疾病（而非原 diagnosis）。

要求：
1. 选择一个与原 diagnosis 在症状或体征上有重叠、易混淆的鉴别疾病作为新诊断
2. 在原 presentation 基础上**最小化修改**：仅调整少量关键鉴别点（如：典型检查异常、特定体征、关键化验值），使其指向新诊断
3. 不要把原 presentation 完全推翻，要保留大部分原始描述
4. 新 answer 必须包含 5 步推理结构，且最后倾向诊断写新疾病名

原 presentation：
{presentation}

原 answer：
{answer}

返回 JSON 严格格式（不要 Markdown）：
{{
  "new_diagnosis": "新的具体疾病名称（区别于 {diagnosis}）",
  "presentation": "修改后的 presentation（≥150 字）",
  "answer": "对应新诊断的 5 步推理 answer"
}}"""


_COMORBIDITY = """你是一位资深临床医师。请在下面的临床场景中**叠加一个常见合并症**（如高血压、2 型糖尿病、慢性肾病 3 期、慢阻肺、冠心病等），并相应地调整 answer 中的治疗与监测建议。

要求：
1. 选择一个在临床上常与原 diagnosis 并存的合并症（保持诊断方向不变）
2. 在 presentation 中加入合并症相关的既往史、用药史、当前控制状态
3. 在 answer 的 5 步推理中：
   - 病情归纳要把合并症纳入
   - 进一步建议要考虑合并症对治疗选择/药物禁忌/监测的影响
   - 倾向诊断仍为原疾病名（不变）

原 presentation：
{presentation}

原 answer：
{answer}

原 diagnosis：{diagnosis}

返回 JSON 严格格式：
{{
  "comorbidity": "添加的合并症名称",
  "presentation": "叠加合并症后的 presentation（≥180 字）",
  "answer": "考虑合并症调整后的 5 步推理 answer"
}}"""


# ─────────────────────────── strategy registry ───────────────────────────

# Each strategy returns a list of (presentation, answer, extra_label_overrides).
# extra_label_overrides may include {"diagnosis_label": ..., "specialty": ...}
# to override the inherited values (used by hardneg which changes the diagnosis).

def _strategy_paraphrase(base, llm):
    text = _llm_text(_PARAPHRASE.format(presentation=base.presentation), llm, max_tokens=1500)
    return [(text, base.answer, {})]


def _strategy_distractor(base, llm):
    text = _llm_text(
        _DISTRACTOR.format(presentation=base.presentation,
                           diagnosis=base.diagnosis_label or "<未提供>"),
        llm, max_tokens=1500,
    )
    return [(text, base.answer, {})]


def _strategy_cot_enrich(base, llm):
    text = _llm_text(
        _COT_ENRICH.format(presentation=base.presentation, answer=base.answer),
        llm, max_tokens=2000,
    )
    return [(base.presentation, text, {})]


def _strategy_hardneg(base, llm):
    obj = _llm_json(
        _HARDNEG.format(presentation=base.presentation,
                        answer=base.answer,
                        diagnosis=base.diagnosis_label or "<未提供>"),
        llm, max_tokens=2500,
    )
    pres = (obj.get("presentation") or "").strip()
    ans = (obj.get("answer") or "").strip()
    new_label = _normalize_label(obj.get("new_diagnosis", ""))
    if not pres or not ans:
        raise ValueError("hardneg 输出缺少 presentation 或 answer")
    return [(pres, ans, {"diagnosis_label": new_label})]


def _strategy_comorbidity(base, llm):
    obj = _llm_json(
        _COMORBIDITY.format(presentation=base.presentation,
                            answer=base.answer,
                            diagnosis=base.diagnosis_label or "<未提供>"),
        llm, max_tokens=2500,
    )
    pres = (obj.get("presentation") or "").strip()
    ans = (obj.get("answer") or "").strip()
    if not pres or not ans:
        raise ValueError("comorbidity 输出缺少 presentation 或 answer")
    return [(pres, ans, {})]


STRATEGIES = {
    "aug_paraphrase":  _strategy_paraphrase,
    "aug_distractor":  _strategy_distractor,
    "aug_cot":         _strategy_cot_enrich,
    "aug_hardneg":     _strategy_hardneg,
    "aug_comorbidity": _strategy_comorbidity,
}

# Default set chosen for Step 1.3 — paraphrase + distractor + cot directly
# improve generalization on the single-turn diagnostic metric; hardneg is
# reserved as Phase 2 DPO seed; comorbidity is opt-in (changes answer).
DEFAULT_STRATEGIES = ["aug_paraphrase", "aug_distractor", "aug_cot"]


# ─────────────────────────── helpers ──────────────────────────────────────

def _normalize_label(s: str) -> str:
    """Same normalization as extractor._normalize_label — duplicated here to
    avoid circular import."""
    import re
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.strip(".。'\"`,，;；:：")
    return s[:200]


def _llm_text(prompt: str, llm: dict, max_tokens: int = 1500) -> str:
    """Call /chat/completions and return the raw assistant text (stripped)."""
    headers = {"Content-Type": "application/json"}
    if llm["api_key"]:
        headers["Authorization"] = f"Bearer {llm['api_key']}"
    payload = {
        "model": llm["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,    # a bit of variation for augmentation
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{llm['base_url'].rstrip('/')}/chat/completions",
                           headers=headers, json=payload)
        resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip occasional code-fence wrapping the model adds despite instructions
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json") or text.startswith("text"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    return text


def _llm_json(prompt: str, llm: dict, max_tokens: int = 2000) -> dict:
    text = _llm_text(prompt, llm, max_tokens=max_tokens)
    return json.loads(text)


def _resolve_llm(job: ExtractionJob, db: Session) -> dict:
    """Resolve {base_url, model_name, api_key} from assistant or inline overrides."""
    if job.assistant_id:
        from .llm_client import resolve_assistant
        assistant = db.query(LLMAssistant).filter_by(id=job.assistant_id).first()
        cfg = resolve_assistant(assistant)
        return {"base_url": cfg["base_url"], "model_name": cfg["model_name"], "api_key": cfg["api_key"]}
    return {
        "base_url": (job.base_url or settings.llm_api_base or "").strip(),
        "model_name": (job.model or settings.llm_model or "").strip(),
        "api_key": (job.api_key or settings.llm_api_key or "").strip(),
    }


# ─────────────────────────── job runner ───────────────────────────────────

def run_augment_job(job_id: int, db: Session):
    """Iterate every base instance of `source_job_id`, run each requested
    strategy, persist variants. Cancel-checked per source instance.
    """
    job = db.query(ExtractionJob).filter_by(id=job_id).first()
    if not job:
        return

    job.status = "running"
    job.started_at = datetime.utcnow()
    db.commit()

    try:
        llm = _resolve_llm(job, db)
    except ValueError as e:
        _fail(db, job, f"LLM 配置无效：{e}")
        return
    if not llm["base_url"] or not llm["model_name"]:
        _fail(db, job, "augment 任务需要可用的 LLM 配置（base_url + model_name）")
        return

    strategies = [s for s in (job.augment_strategies or []) if s in STRATEGIES]
    if not strategies:
        _fail(db, job, f"未指定有效的 augment_strategies；合法值：{sorted(STRATEGIES.keys())}")
        return

    cfg = (job.config or {}) if isinstance(job.config, dict) else {}
    max_source = int(cfg.get("max_source_instances", 200))
    variants_per = max(1, int(cfg.get("variants_per_strategy", 1)))

    # Source instances = base outputs of the upstream job. We only augment
    # base outputs (synthesis_strategy ∈ {case_direct, guideline_synth}),
    # never variants — avoids exponential blowup if user augments an
    # already-augmented job.
    source_q = (db.query(DiagnosticInstance)
                .filter(DiagnosticInstance.job_id == job.source_job_id)
                .filter(DiagnosticInstance.synthesis_strategy.in_(["case_direct", "guideline_synth"]))
                .order_by(DiagnosticInstance.id.asc())
                .limit(max_source))
    bases = source_q.all()
    job.total_docs = len(bases)
    db.commit()

    for base in bases:
        db.refresh(job)
        if job.is_cancelled:
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            db.commit()
            return

        any_success = False
        for strategy_name in strategies:
            fn = STRATEGIES[strategy_name]
            for _ in range(variants_per):
                try:
                    variants = fn(base, llm)
                    for presentation, answer, overrides in variants:
                        if not presentation or not answer:
                            continue
                        db.add(DiagnosticInstance(
                            presentation=presentation,
                            answer=answer,
                            diagnosis_label=overrides.get("diagnosis_label", base.diagnosis_label),
                            specialty=overrides.get("specialty", base.specialty),
                            synthesis_strategy=strategy_name,
                            parent_instance_id=base.id,
                            source_doc_id=base.source_doc_id,
                            job_id=job.id,
                        ))
                        any_success = True
                except Exception:
                    # Failure of one strategy on one base doesn't block others.
                    pass

        if any_success:
            job.processed_docs += 1
        else:
            job.failed_docs += 1
        db.commit()

    job.status = "completed"
    job.completed_at = datetime.utcnow()
    db.commit()


def _fail(db, job, msg):
    job.status = "failed"
    job.error_message = msg
    job.completed_at = datetime.utcnow()
    db.commit()
