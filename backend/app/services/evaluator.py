"""
LLM-as-Judge evaluation runner — two-phase pipeline.

Phase 1 (generating):  candidate generates response for ALL items.
                       If a baseline is configured, baseline generates
                       in parallel — meaning candidate AND baseline must
                       be running simultaneously during this phase.
                       Judge is NOT consulted; it can be offline.

Phase 2 (judging):     judge scores every candidate (and baseline) response.
                       Judge MUST be running. Candidate / baseline can be
                       stopped (we only read their cached responses from DB).

This split lets the operator alternate which vllm model is loaded:
generation phase needs candidate (+ baseline) up; judging phase needs
judge up. Item rows persist intermediate state, so a re-run/續跑 picks
up at the right phase automatically.

Concurrency model for phase 1:
  - We submit ALL pending (item, role) calls to a ThreadPoolExecutor at
    once, so vllm's continuous batching can keep both GPUs saturated
    instead of stalling between items. DB writes are funneled back to
    the main thread via `as_completed`. Concurrency is configurable
    via env var `CDSS_EVAL_GEN_CONCURRENCY` (default 8).
"""
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..models.models import (
    Dataset, DatasetItem, EvaluationRun, EvaluationItem, LLMAssistant,
)
from .llm_client import resolve_assistant, chat_completion, parse_judge_json


# Concurrent in-flight requests during generation phase. vllm has its own
# continuous batching so values up to ~32 are reasonable on a single GPU;
# with two vllm servers (candidate + baseline) the requests are split between
# them automatically by virtue of the assistant routing.
_GEN_CONCURRENCY = int(os.getenv("CDSS_EVAL_GEN_CONCURRENCY", "8"))


JUDGE_PROMPT = """你是一位资深临床医师，正在对 AI 临床助手的回答进行严格评分。请按下面的标准对 AI 回答与参考答案进行比较。

【评分标准 0-5 分】
- 5 分：医学正确性、推理过程、关键诊断/治疗要点全部覆盖且无错误
- 4 分：核心结论正确，仅有少量非关键遗漏
- 3 分：结论方向正确但论证不足或漏掉重要鉴别/治疗要点（视为合格）
- 2 分：部分正确但有明显错误或重要遗漏
- 1 分：大部分错误，仅有少量正确点
- 0 分：完全错误，或包含医学性危险错误

【评分维度】关注：诊断准确度 / 鉴别诊断完整性 / 治疗合理性 / 推理严谨性。文风/句式/词汇差异不扣分。

【题目】
{instruction}

【参考答案】
{expected}

【AI 回答】
{response}

请严格只输出如下 JSON（不要多余文字、不要 markdown 代码块）：
{{"score": <0~5 的整数>, "reasoning": "<≤120字的评分理由，指出关键正确点与错误点>"}}"""


# ─────────────────────────────── helpers ─────────────────────────────────

def _agg_stats(items, role: str):
    """Compute mean score + pass-rate from EvaluationItem list."""
    scores = [getattr(i, f"{role}_score") for i in items
              if getattr(i, f"{role}_score") is not None]
    if not scores:
        return None, None
    mean = sum(scores) / len(scores)
    passes = sum(1 for s in scores if s >= 3)
    return round(mean, 4), round(passes / len(scores), 4)


def _build_prompt(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\n\n{input_text}"
    return instruction


def _set_phase(db, run, phase: str, *, reset_progress: bool = True,
               progress_total: int = 0):
    run.phase = phase
    if reset_progress:
        run.progress_done = 0
        run.progress_total = progress_total
    db.commit()


def _refresh_run_cancelled(db, run) -> bool:
    db.refresh(run)
    return bool(run.is_cancelled)


# ─────────────────────────────── ensure items ───────────────────────────

def _ensure_eval_items(db, run, dataset_items):
    """Create EvaluationItem rows for the (possibly new) run.

    For re-runs that already have items, leave them in place so we can
    resume from where we left off.
    """
    existing = (db.query(EvaluationItem)
                .filter(EvaluationItem.run_id == run.id)
                .order_by(EvaluationItem.id).all())
    if existing:
        return existing

    rows = []
    for di in dataset_items:
        ev = EvaluationItem(
            run_id=run.id,
            dataset_item_id=di.id,
            instruction=di.instruction,
            expected_output=di.output,
        )
        db.add(ev)
        rows.append(ev)
    db.commit()
    # Re-load with populated ids
    return (db.query(EvaluationItem)
            .filter(EvaluationItem.run_id == run.id)
            .order_by(EvaluationItem.id).all())


# ─────────────────────────────── phase 1: generation ────────────────────

def _phase_generate(db, run, items, dataset_items_by_id, candidate_cfg, baseline_cfg):
    """Phase 1: generate candidate (and optional baseline) responses.

    Cross-sample parallelism: every (item, role) call is submitted to a
    ThreadPoolExecutor up-front so vllm's continuous batching can pack
    requests on both GPUs simultaneously. DB writes are serialized in
    the main thread via `as_completed`.

    `candidate_cfg` / `baseline_cfg` are PLAIN DICTS pre-resolved by the
    caller in the main thread. We deliberately do NOT pass ORM objects
    into worker threads — SQLAlchemy sessions are not thread-safe and
    the resulting attribute reads in `resolve_assistant()` would race,
    occasionally producing the wrong base_url / model_name (e.g. both
    candidate and baseline calls hitting the same vllm port).
    """
    has_baseline = baseline_cfg is not None

    def _item_done(it):
        if not it.candidate_response:
            return False
        if has_baseline and not it.baseline_response:
            return False
        return True

    # Build flat task list (one entry per missing role per item).
    tasks = []  # (item_id, role, cfg, prompt)
    for it in items:
        di = dataset_items_by_id.get(it.dataset_item_id)
        if not di:
            it.error_message = (it.error_message or "") + "[gen] 原始 dataset_item 已被删除\n"
            continue
        prompt = _build_prompt(di.instruction, di.input)
        if not it.candidate_response:
            tasks.append((it.id, "candidate", candidate_cfg, prompt))
        if has_baseline and not it.baseline_response:
            tasks.append((it.id, "baseline", baseline_cfg, prompt))
    db.commit()  # flush any error_message updates

    _set_phase(db, run, "generating", progress_total=len(items))
    item_done_set = {it.id for it in items if _item_done(it)}
    run.progress_done = len(item_done_set)
    db.commit()

    if not tasks:
        return True

    def _work(role, cfg, prompt):
        """Runs in worker thread — receives an immutable dict, never an ORM object."""
        try:
            text = chat_completion(
                base_url=cfg["base_url"],
                model_name=cfg["model_name"],
                api_key=cfg["api_key"],
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
            )
            return (role, text, None)
        except Exception as e:
            # Include cfg fingerprint so a 404/etc reveals exactly what was sent
            # (model name + endpoint). Critical for diagnosing "model not found"
            # errors where the wrong model_name reaches a vllm server.
            sent = f"endpoint={cfg.get('base_url')!r} model={cfg.get('model_name')!r}"
            return (role, None, f"{e!r} | sent {sent}")

    cancelled = False
    with ThreadPoolExecutor(max_workers=_GEN_CONCURRENCY) as pool:
        futures_map = {}
        for it_id, role, cfg, prompt in tasks:
            fut = pool.submit(_work, role, cfg, prompt)
            futures_map[fut] = it_id

        try:
            for fut in as_completed(futures_map):
                it_id = futures_map[fut]
                role_name, text, err = fut.result()

                # Single-thread DB write here.
                it = db.query(EvaluationItem).filter_by(id=it_id).first()
                if not it:
                    continue
                if text is not None:
                    setattr(it, f"{role_name}_response", text)
                else:
                    it.error_message = (it.error_message or "") + f"[{role_name}] {err}\n"

                # Update item-completion counter for the progress bar.
                if it.id not in item_done_set and _item_done(it):
                    item_done_set.add(it.id)
                    run.progress_done = len(item_done_set)
                db.commit()

                if _refresh_run_cancelled(db, run):
                    cancelled = True
                    # Cancel queued futures (running ones in vllm finish).
                    for f in futures_map:
                        f.cancel()
                    break
        except Exception:
            for f in futures_map:
                f.cancel()
            raise

    if cancelled:
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
        db.commit()
        return False
    return True


# ─────────────────────────────── phase 2: judge ─────────────────────────

def _phase_judge(db, run, items, judge_cfg, has_baseline: bool):
    """Phase 2: judge scores both candidate and baseline responses.

    Like phase 1, the caller pre-resolves judge to a plain dict in the
    main thread and we pass that dict everywhere — no ORM access here.
    """
    _set_phase(db, run, "judging", progress_total=len(items))

    # An item is "done" for judging when every role with a response has a score.
    def _is_done(it):
        if it.candidate_response and it.candidate_score is None:
            return False
        if has_baseline and it.baseline_response and it.baseline_score is None:
            return False
        return True

    pending = [it for it in items if not _is_done(it)]
    run.progress_done = len(items) - len(pending)
    db.commit()

    for it in pending:
        if _refresh_run_cancelled(db, run):
            run.status = "cancelled"
            run.completed_at = datetime.utcnow()
            db.commit()
            return False

        # candidate
        if it.candidate_response and it.candidate_score is None:
            try:
                score, reasoning = _judge_with_cfg(
                    judge_cfg, it.instruction,
                    it.expected_output, it.candidate_response,
                )
                it.candidate_score = score
                it.candidate_reasoning = reasoning
            except Exception as e:
                it.error_message = (it.error_message or "") + f"[judge-candidate] {e}\n"

        # baseline (optional)
        if has_baseline and it.baseline_response and it.baseline_score is None:
            try:
                score_b, reasoning_b = _judge_with_cfg(
                    judge_cfg, it.instruction,
                    it.expected_output, it.baseline_response,
                )
                it.baseline_score = score_b
                it.baseline_reasoning = reasoning_b
            except Exception as e:
                it.error_message = (it.error_message or "") + f"[judge-baseline] {e}\n"

        run.progress_done += 1

        # Refresh aggregated stats periodically so the UI mean ticks up live.
        if run.progress_done % 5 == 0 or run.progress_done == len(items):
            all_items = db.query(EvaluationItem).filter_by(run_id=run.id).all()
            run.candidate_score, run.candidate_pass_rate = _agg_stats(all_items, "candidate")
            if has_baseline:
                run.baseline_score, run.baseline_pass_rate = _agg_stats(all_items, "baseline")
        db.commit()
    return True


def _judge_with_cfg(judge_cfg, instruction, expected, response):
    prompt = JUDGE_PROMPT.format(
        instruction=(instruction or "")[:2000],
        expected=(expected or "")[:2000],
        response=(response or "")[:2000],
    )
    raw = chat_completion(
        base_url=judge_cfg["base_url"],
        model_name=judge_cfg["model_name"],
        api_key=judge_cfg["api_key"],
        prompt=prompt,
        temperature=0.3,
        max_tokens=400,
    )
    obj = parse_judge_json(raw)
    if not obj:
        return None, f"评分输出无法解析为 JSON：{raw[:300]}"
    try:
        score = float(obj.get("score"))
    except (TypeError, ValueError):
        return None, f"评分值无效：{obj.get('score')}"
    score = max(0.0, min(5.0, score))
    return score, obj.get("reasoning", "")


# ─────────────────────────────── orchestration ───────────────────────────

def run_evaluation(run_id: int, db_factory):
    """Background entry. Runs phase 1 (generation) then phase 2 (judging).

    `run.phase` is the source of truth for which phase the run is in:
      - 'pending' or 'generating' → phase 1 may still need to execute
      - 'judging' → phase 1 is done (including any partial failures);
                     don't re-validate or re-execute it on resume
      - 'done' → both phases finished

    This means: if phase 1 attempted every item and we then crash on
    judge-validation, the run stays at phase='judging' (status='failed')
    so that resuming after starting judge does NOT re-require candidate
    or baseline to be online.
    """
    db = db_factory()
    try:
        run = db.query(EvaluationRun).filter_by(id=run_id).first()
        if not run:
            return
        run.status = "running"
        run.is_cancelled = False
        run.started_at = run.started_at or datetime.utcnow()
        db.commit()

        candidate = db.query(LLMAssistant).filter_by(id=run.candidate_assistant_id).first()
        baseline = (db.query(LLMAssistant).filter_by(id=run.baseline_assistant_id).first()
                    if run.baseline_assistant_id else None)
        judge = db.query(LLMAssistant).filter_by(id=run.judge_assistant_id).first()
        dataset = db.query(Dataset).filter_by(id=run.dataset_id).first()

        if not candidate or not judge or not dataset:
            run.status = "failed"
            run.phase = run.phase or "pending"
            run.error_message = "candidate / judge 助手或 dataset 不存在"
            run.completed_at = datetime.utcnow()
            db.commit()
            return

        items_q = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id)
        all_dataset_items = items_q.all()
        if not all_dataset_items:
            run.status = "failed"
            run.error_message = "数据集为空"
            run.completed_at = datetime.utcnow()
            db.commit()
            return
        if run.sample_limit and run.sample_limit > 0:
            all_dataset_items = all_dataset_items[: run.sample_limit]
        dataset_items_by_id = {di.id: di for di in all_dataset_items}

        items = _ensure_eval_items(db, run, all_dataset_items)
        current_phase = run.phase or "pending"

        # ─── auto-recovery for stuck/legacy runs ─────────────────────
        # An item is considered "attempted" for a role when it has either a
        # response OR an error message mentioning that role. If every item
        # has been attempted for every required role, phase 1 has done all
        # it can — advance to 'judging' even if phase still says 'generating'
        # (which can happen due to:
        #   1) legacy runs created before the orchestrator advanced phase
        #      explicitly between phase 1 success and judge validation;
        #   2) phase 1 completed with partial errors and the operator wants
        #      to score whatever was generated rather than re-run candidate.)
        def _was_attempted(it, role):
            if getattr(it, f"{role}_response"):
                return True
            if it.error_message and f"[{role}]" in it.error_message:
                return True
            return False

        if current_phase in ("pending", "generating"):
            all_attempted = all(
                _was_attempted(it, "candidate")
                and (not baseline or _was_attempted(it, "baseline"))
                for it in items
            )
            if all_attempted and items:
                current_phase = "judging"
                run.phase = "judging"
                db.commit()

        # ─── Phase 1: generation ─────────────────────────────────────
        # Only execute if we haven't already moved past it. Once phase
        # advances to 'judging', a resume will skip phase 1 entirely
        # regardless of whether some items have missing responses (the
        # judge phase tolerates missing responses by simply not scoring
        # that role for that item).
        if current_phase in ("pending", "generating"):
            # Decide which models actually need to be online based on what
            # work remains (e.g., if baseline was already fully generated
            # in a prior attempt, we don't require it now).
            needs_candidate = any(not it.candidate_response for it in items)
            needs_baseline = baseline and any(
                not it.baseline_response for it in items)
            phase1_has_work = needs_candidate or needs_baseline

            if phase1_has_work:
                # Resolve helper configs to plain dicts in THIS thread, before
                # handing them to the worker pool. Passing ORM objects across
                # threads triggers SQLAlchemy session races and can yield
                # wrong base_url / model_name (e.g. both candidate and baseline
                # ending up calling the same vllm port).
                candidate_cfg = None
                baseline_cfg = None
                if needs_candidate:
                    try:
                        candidate_cfg = resolve_assistant(candidate)
                    except ValueError as e:
                        run.status = "failed"
                        run.error_message = (
                            f"candidate 助手不可用：{e}（生成阶段需要候选模型在运行）")
                        run.completed_at = datetime.utcnow()
                        db.commit()
                        return
                if needs_baseline:
                    try:
                        baseline_cfg = resolve_assistant(baseline)
                    except ValueError as e:
                        run.status = "failed"
                        run.error_message = (
                            f"baseline 助手不可用：{e}（与 candidate 同时运行才能并行生成；"
                            f"如不需要对比，请删除 baseline_assistant_id 重建）")
                        run.completed_at = datetime.utcnow()
                        db.commit()
                        return

                # Diagnostic: print resolved cfgs so a misconfiguration (e.g. a
                # model_name that doesn't match what vllm actually serves) shows
                # up clearly in the backend log before any LLM call goes out.
                print(
                    f"[eval run {run.id}] phase 1 cfgs:"
                    + (f" candidate={{base_url={candidate_cfg['base_url']!r}, "
                       f"model_name={candidate_cfg['model_name']!r}}}"
                       if candidate_cfg else "")
                    + (f"  baseline={{base_url={baseline_cfg['base_url']!r}, "
                       f"model_name={baseline_cfg['model_name']!r}}}"
                       if baseline_cfg else ""),
                    flush=True,
                )

                if not _phase_generate(
                        db, run, items, dataset_items_by_id,
                        candidate_cfg, baseline_cfg):
                    return

            # Whether we did work or not, advance to 'judging' so that any
            # subsequent failure (e.g., judge offline) doesn't trick a
            # resume into re-running phase 1.
            run.phase = "judging"
            run.progress_total = len(items)
            run.progress_done = 0
            db.commit()

        # ─── Phase 2: judging ────────────────────────────────────────
        def _needs_judge(it):
            if it.candidate_response and it.candidate_score is None:
                return True
            if baseline and it.baseline_response and it.baseline_score is None:
                return True
            return False

        if any(_needs_judge(it) for it in items):
            try:
                judge_cfg = resolve_assistant(judge)
            except ValueError as e:
                run.status = "failed"
                run.error_message = (
                    f"judge 助手不可用：{e}（生成已完成，启动 judge 后点续跑即可继续评分）")
                run.completed_at = datetime.utcnow()
                db.commit()
                return
            if not _phase_judge(db, run, items, judge_cfg, baseline is not None):
                return

        # ─── Done ────────────────────────────────────────────────────
        all_done = db.query(EvaluationItem).filter_by(run_id=run.id).all()
        run.candidate_score, run.candidate_pass_rate = _agg_stats(all_done, "candidate")
        if baseline:
            run.baseline_score, run.baseline_pass_rate = _agg_stats(all_done, "baseline")
        run.status = "completed"
        run.phase = "done"
        run.completed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        try:
            run = db.query(EvaluationRun).filter_by(id=run_id).first()
            if run:
                run.status = "failed"
                run.error_message = f"{e}\n{traceback.format_exc()[-1500:]}"
                run.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def start_evaluation_thread(run_id: int, db_factory):
    """Spawn a background thread; FastAPI's BackgroundTasks doesn't survive request lifetime well for long runs."""
    t = threading.Thread(target=run_evaluation, args=(run_id, db_factory), daemon=True)
    t.start()
    return t
