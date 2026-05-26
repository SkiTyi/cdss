# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CDSS — a medical knowledge distillation + model training monitor + clinical reasoning demo platform. Six workflows the codebase is organized around:

```
data ingestion → knowledge extraction → dataset construction
                                       ↓
                continued pre-training (CPT) → SFT fine-tuning → clinical assistant / evaluation
```

CPT is optional but recommended: it injects medical-domain knowledge into the base model via causal-LM training on raw documents *before* SFT teaches it the QA format.

Stack: FastAPI + SQLAlchemy (SQLite) backend, React 19 + Vite + Tailwind CSS 4 + Recharts frontend.

## Commands

### Backend
```bash
cd backend
# conda env on the GPU server is `cdss`; vllm runs from a separate `vllm` env
uvicorn app.main:app --reload --port 8001     # Vite proxy targets 8001 — do not change the port
```
Config in `backend/.env` (copy from `.env.example`): `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`, `CRAWLER_DATA_PATH`, `GUIDELINE_DATA_PATH`.

### Frontend
```bash
cd frontend
npm run dev          # Vite dev server, proxies /api → http://localhost:8001
npm run build
npm run lint
```

### Local model serving (vllm)
```bash
# Reference command for what the in-app vllm manager invokes:
vllm serve /path/to/model --served-model-name <name> --host 127.0.0.1 --port <auto>
# Anything else (--max-model-len, --gpu-memory-utilization, --disable-log-requests, etc.)
# must be added via the assistant's `extra_vllm_args` field — see `services/vllm_manager.py`.
```

### Dev environment caveats
- The dev box (Windows / WSL) has **no GPU**. All training / vllm / evaluation tests must run on the GPU server. Do not try to install heavy ML deps locally.
- `cdss.db` (~125 MB) lives at `backend/cdss.db`; it's the source of truth for everything in the UI. Never delete it casually — there's no backup mechanism.

## Architecture

### Backend module map
```
backend/app/
├── main.py            # FastAPI bootstrap, mounts all api/ routers, calls run_migrations()
├── database.py        # engine + SessionLocal + Base + run_migrations()
├── config.py          # pydantic-settings, reads .env
├── models/models.py   # ALL ORM models in one file
├── api/               # FastAPI routers, one per top-level resource
│   ├── documents.py   ├── extraction.py    ├── datasets.py
│   ├── training.py    ├── evaluations.py   ├── assistants.py  (CRUD for LLMAssistant)
│   └── assistant.py   (the clinical demo endpoint, singular — separate file)
└── services/
    ├── data_loader.py    # scans crawler/*.xlsx + guideline/**/*.md → documents table
    ├── extractor.py      # LLM extraction job runner (qa_extraction & clinical_reasoning_synthesis)
    ├── trainer.py        # subprocess + monitor thread for train_script.py (SFT)
    ├── train_script.py   # standalone HF Transformers + PEFT + TRL fine-tune script
    ├── pretrainer.py     # subprocess + monitor thread for pretrain_script.py (CPT)
    ├── pretrain_script.py # standalone HF Trainer + DataCollatorForLanguageModeling causal-LM script
    ├── evaluator.py      # LLM-as-Judge runner (3-phase serial)
    ├── vllm_manager.py   # subprocess + health-poll thread for local vllm assistants
    └── llm_client.py     # resolve_assistant() + chat_completion() shared helpers
```

### LLMAssistant — the unifying abstraction
All LLM calls (extraction / evaluation / clinical demo) go through `services/llm_client.resolve_assistant()` which yields `{base_url, model_name, api_key}`.

- **Remote** assistants: stored fields used directly.
- **Local** assistants: `vllm_manager.start()` spawns vllm, assigns a port (8011-8099), and writes `base_url=http://127.0.0.1:<port>/v1` once `/v1/models` confirms the served-model-name. Status flow: `stopped → starting → running` (or `failed`).

Downstream tables reference assistants by id (`extraction_jobs.assistant_id`, `evaluation_runs.candidate/baseline/judge_assistant_id`). UI components show a dropdown of `running` assistants.

**Never hardcode vllm CLI flags** beyond model / served-model-name / host / port / max-model-len / LoRA / `--tensor-parallel-size` (auto-added for multi-GPU). vllm versions break on flags like `--disable-log-requests` — those must be user-supplied via `extra_vllm_args`.

`gpu_ids` semantics (mirrors training): `None`=auto / `[n]`=single GPU via `CUDA_VISIBLE_DEVICES=n` / `[n,m,...]`=multi-GPU + auto `--tensor-parallel-size`.

### Background work patterns
| Module | Mechanism | File |
|---|---|---|
| extraction | `BackgroundTasks` (per request) | `api/extraction.py` |
| training | `subprocess.Popen` + monitor thread | `services/trainer.py` |
| evaluation | `threading.Thread` (long-lived) | `services/evaluator.py` |
| vllm | `subprocess.Popen` + monitor thread + health poll | `services/vllm_manager.py` |

### Training pipeline (SFT)
1. `api/training.py` exports the linked dataset to `training_runs/<exp_id>/{train,val}.jsonl`.
2. `services/trainer.py` builds env (`CUDA_VISIBLE_DEVICES` from `gpu_ids`) and spawns `python train_script.py --config ...` (or `torchrun ...` for multi-GPU).
3. `train_script.py` runs as **rank-0 only writer**: training metrics + events emitted as one JSON object per line to `training_runs/<exp_id>/metrics.jsonl`. The `metrics.jsonl` path is the single source of truth for both single-GPU and DDP — do not parse stdout for metrics.
4. `trainer.py._tail_metrics_file()` tails this file in the main monitor thread and writes to `training_metrics` / updates `experiment.config["baseline_eval"]` / `["final_eval"]`.
5. `trainer.py._read_stdout()` runs in a parallel thread and stores **non-JSON** stdout lines as logs (avoids double-counting metrics that also went through stdout).
6. Pre-training baseline eval and post-training final eval are both done inside `train_script.py` via `trainer.get_eval_dataloader()` (do not hand a raw text Dataset to the data collator — it expects tokenized inputs).

### CPT pipeline (continued pre-training)
Same shape as SFT, separate file lineage:
1. `api/pretraining.py.start_pretraining` queries documents via `CorpusFilter` (types / min_content_length / doc_limit / eval_ratio) and dumps `pretrain_runs/<exp_id>/corpus_{train,eval}.jsonl` (one `{"text": ...}` per line).
2. `services/pretrainer.py` spawns `pretrain_script.py` with the same gpu_ids / DDP semantics as SFT.
3. `pretrain_script.py` tokenizes all docs into a single stream (separated by EOS), splits into fixed-size blocks (`block_size`, default 4096), and trains via HF `Trainer` + `DataCollatorForLanguageModeling(mlm=False)` for next-token prediction.
4. Same `metrics.jsonl` channel; emits `perplexity` (= exp(eval_loss)) alongside loss.
5. **CPT output → SFT input**: the saved adapter / model dir from a completed CPT experiment can be pasted directly into the SFT modal's `base_model` field. The two pipelines do not auto-link in the DB; this is intentional to keep them decoupled.

### Evaluation pipeline (LLM-as-Judge, 3-phase serial)
`services/evaluator.py:run_evaluation()` deliberately runs phases sequentially across the **entire** dataset before swapping models:
1. `phase=generating_candidate` — candidate model generates response for every item
2. `phase=generating_baseline` — baseline model (if any) generates response for every item
3. `phase=judging` — judge model scores candidate (and baseline if present) for every item

This shape exists because the GPU server typically can't host >1 model at once. The operator stops one vllm assistant and starts the next between phases, then clicks **续跑 (resume)** to continue from where it left off.

`EvaluationItem` rows persist partial state (`candidate_response`, `candidate_score`, `baseline_response`, `baseline_score`). The phase functions skip items whose target field is already populated, so resume-after-restart is automatic. `restart_mode=fresh` deletes all items and re-scores from scratch.

### Database / ORM dual-update rule
Schema changes need **two** edits in lockstep:
1. Add `Column(...)` in `models/models.py`
2. Add `(table_name, column_name, "TYPE [DEFAULT ...]")` to the list in `database.py:run_migrations()`

Forgetting #1 → `'X' is an invalid keyword argument for Y` when constructing the model.
Forgetting #2 → existing `cdss.db` lacks the column; queries fail.

`run_migrations()` is idempotent (catches "duplicate column" errors per ALTER) and runs on every backend startup. There is no Alembic.

It also resets `llm_assistants.status` from `running`/`starting` → `stopped` on startup, since the previous backend process owned those vllm subprocess handles and they're orphaned now.

### Frontend conventions
- API client: `frontend/src/api/client.js` — one named export per resource (`documents`, `extraction`, `datasets`, `training`, `assistant`, `assistants`, `evaluations`, `pretraining`).
- Pages: one file per top-level route in `pages/`. Most pages are tabbed:
  - `Training.jsx`: 增量预训练 CPT / 微调训练 SFT / 模型评估 (CPT tab is a separate `Pretraining.jsx` component embedded as a Tab)
  - `Assistant.jsx`: 临床演示 / 助手管理
- Markdown rendering uses the dependency-free `components/MarkdownView.jsx` — do not pull `react-markdown`. The clinical demo (`Assistant.jsx` DemoTab) uses it to render LLM responses.
- Status polling: pages start `setInterval` only while at least one job is `running` and tear it down otherwise (see `Extraction.jsx`, `Training.jsx`, `Assistant.jsx` for the pattern).
- Recharts gotcha: `<Line>` components must be **direct children** of `<LineChart>` — wrapping them in `<>...</>` Fragment makes Recharts silently drop them. Already debugged in `Training.jsx` MetricsPanel.

### Data sources
- `crawler/` — Yiigle scraper output, `*_cleaned.xlsx` (~5000 case reports) plus the cleaning scripts. `data_loader.load_cases()` reads the `_cleaned.xlsx` file.
- `guideline/` — ~9942 MD files from MinerU, organized by specialty under several subdirectories (中文诊断和治疗划分 / 医学内容切分_NICE / 卫健委指南切分 / 指南清洗第二阶段). `data_loader.load_guidelines()` walks the tree.

`POST /api/documents/load` triggers ingestion as a background task; safe to re-run (existing rows by source_path are skipped).
