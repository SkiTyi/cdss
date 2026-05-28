from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from ..database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50))  # case_report | guideline
    title = Column(String(500))
    source_path = Column(String(1000))
    content = Column(Text)
    doc_metadata = Column(JSON, default={})
    status = Column(String(50), default="pending")  # pending | extracted
    created_at = Column(DateTime, default=datetime.utcnow)

    instances = relationship("DiagnosticInstance", back_populates="source_doc")


class ExtractionJob(Base):
    __tablename__ = "extraction_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    document_type = Column(String(50))  # case_report | guideline | all
    # Phase 1 task types:
    #   case_extract     — extract DiagnosticInstance(s) from a case report (Q→A from full case)
    #   guideline_synth  — synthesize N virtual-patient DiagnosticInstances from a guideline doc
    #   augment          — produce variants from an existing job's instances (paraphrase / distractor / hardneg / ...)
    task_type = Column(String(50), default="case_extract")
    prompt_template = Column(Text)
    model = Column(String(100))
    base_url = Column(String(500), nullable=True)     # LLM API base url override
    api_key = Column(String(500), nullable=True)      # LLM API key override (optional)
    # Optional reference to a configured LLM assistant. When set, the job uses
    # the assistant's resolved {base_url, model_name, api_key} and the inline
    # base_url/model/api_key fields above are ignored.
    assistant_id = Column(Integer, ForeignKey("llm_assistants.id"), nullable=True)
    doc_limit = Column(Integer, nullable=True)       # 限制处理文档数量，None 表示全部
    # For task_type='augment': which upstream job's instances to operate on,
    # and which augmentation strategies to apply (list of strategy names).
    source_job_id = Column(Integer, ForeignKey("extraction_jobs.id"), nullable=True)
    augment_strategies = Column(JSON, nullable=True)
    is_cancelled = Column(Boolean, default=False)
    status = Column(String(50), default="pending")  # pending | running | paused | completed | failed | cancelled
    total_docs = Column(Integer, default=0)
    processed_docs = Column(Integer, default=0)
    failed_docs = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    instances = relationship("DiagnosticInstance", back_populates="job",
                             foreign_keys="DiagnosticInstance.job_id")


class DiagnosticInstance(Base):
    """A single (presentation → answer) training-shaped sample.

    This is the *only* unit knowledge is stored in. Anything that doesn't fit
    this shape — entities, relations, summaries, declarative facts — should
    be transformed into this shape before persisting. Justification: the
    downstream task (and the test set) is single-turn diagnostic
    `(presentation) → (answer)`, so the storage format mirrors training
    samples 1:1 instead of going through an intermediate human-readable
    representation.

    Created in three ways:
      * case_extract     — one or more from a case report
      * guideline_synth  — virtual patients constructed from a guideline
      * augment          — variants of an existing instance (paraphrase /
                           distractor injection / hard-negative / cot enrich)
    """
    __tablename__ = "diagnostic_instances"

    id = Column(Integer, primary_key=True, index=True)
    # Training fields (the model sees these)
    presentation = Column(Text)             # full clinical scenario
    answer = Column(Text)                   # diagnosis + reasoning (CoT)

    # Metadata used ONLY for sampling, balancing, analytics — not for training
    diagnosis_label = Column(String(200), index=True)  # normalized (lowercased, stripped) primary diagnosis
    specialty = Column(String(100), nullable=True)
    difficulty = Column(Float, nullable=True)           # 0~1; populated by Step 1.3 LLM judge

    # Lineage
    synthesis_strategy = Column(String(50))             # case_direct | guideline_synth | aug_paraphrase | aug_distractor | aug_hardneg | aug_cot | aug_comorbidity
    parent_instance_id = Column(Integer, ForeignKey("diagnostic_instances.id"), nullable=True)
    source_doc_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    job_id = Column(Integer, ForeignKey("extraction_jobs.id"), nullable=True)

    # Curation
    quality_score = Column(Float, nullable=True)
    is_approved = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    source_doc = relationship("Document", back_populates="instances")
    job = relationship("ExtractionJob", back_populates="instances",
                       foreign_keys=[job_id])
    parent = relationship("DiagnosticInstance", remote_side=[id])
    dataset_items = relationship("DatasetItem", back_populates="instance")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    description = Column(Text, nullable=True)
    format = Column(String(50), default="alpaca")  # alpaca | sharegpt | custom
    item_count = Column(Integer, default=0)
    status = Column(String(50), default="building")  # building | ready | exported
    created_at = Column(DateTime, default=datetime.utcnow)

    items = relationship("DatasetItem", back_populates="dataset")
    experiments = relationship("TrainingExperiment", back_populates="dataset")


class DatasetItem(Base):
    __tablename__ = "dataset_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    instance_id = Column(Integer, ForeignKey("diagnostic_instances.id"), nullable=True)
    instruction = Column(Text)
    input = Column(Text, default="")
    output = Column(Text)
    system_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="items")
    instance = relationship("DiagnosticInstance", back_populates="dataset_items")


class TrainingExperiment(Base):
    __tablename__ = "training_experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    base_model = Column(String(200))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    config = Column(JSON, default={})
    status = Column(String(50), default="pending")  # pending | running | completed | failed | stopped
    best_eval_loss = Column(Float, nullable=True)
    process_pid = Column(Integer, nullable=True)
    log_file = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    total_steps = Column(Integer, nullable=True)
    current_step = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="experiments")
    metrics = relationship("TrainingMetric", back_populates="experiment")
    logs = relationship("TrainingLog", back_populates="experiment")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("training_experiments.id"))
    step = Column(Integer)
    epoch = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    extra_metrics = Column(JSON, default={})
    recorded_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("TrainingExperiment", back_populates="metrics")


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("training_experiments.id"))
    level = Column(String(20), default="info")  # info | warning | error | metrics
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("TrainingExperiment", back_populates="logs")


# ─────────────────────────────── Continued Pre-Training (CPT) ──────────────
# Domain-adaptive continued pre-training: take raw documents (case reports,
# guidelines), tokenize as flat text, and train the base model with causal
# language modeling (next-token prediction). The output is a
# domain-adapted base model that can later be plugged into an SFT
# experiment as `base_model`.

class PretrainExperiment(Base):
    __tablename__ = "pretrain_experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    base_model = Column(String(500))           # local path or HF id
    # corpus_filter: {document_types: ["case_report"|"guideline"|"all"],
    #                 min_content_length: int, doc_limit: int|None,
    #                 eval_ratio: float (0~0.2 typical)}
    corpus_filter = Column(JSON, default={})
    # config: lr, num_epochs, batch_size, gradient_accumulation_steps,
    #         block_size, lora_*, use_4bit, use_bf16, gpu_ids, etc.
    config = Column(JSON, default={})
    status = Column(String(50), default="pending")
    best_eval_loss = Column(Float, nullable=True)
    process_pid = Column(Integer, nullable=True)
    log_file = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    total_steps = Column(Integer, nullable=True)
    current_step = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    metrics = relationship("PretrainMetric", back_populates="experiment")
    logs = relationship("PretrainLog", back_populates="experiment")


class PretrainMetric(Base):
    __tablename__ = "pretrain_metrics"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("pretrain_experiments.id"))
    step = Column(Integer)
    epoch = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    extra_metrics = Column(JSON, default={})
    recorded_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("PretrainExperiment", back_populates="metrics")


class PretrainLog(Base):
    __tablename__ = "pretrain_logs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("pretrain_experiments.id"))
    level = Column(String(20), default="info")
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("PretrainExperiment", back_populates="logs")


class LLMAssistant(Base):
    """Reusable LLM endpoint config — used by extraction, evaluation, demo."""
    __tablename__ = "llm_assistants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True)
    type = Column(String(20))                # 'local' | 'remote'
    description = Column(Text, nullable=True)

    # Common (computed for local once running, set directly for remote)
    base_url = Column(String(500), nullable=True)      # e.g. http://127.0.0.1:8011/v1
    model_name = Column(String(200), nullable=True)    # served-model-name (local) or remote model id
    api_key = Column(String(500), nullable=True)       # remote: required (unless base_url is local); local: usually empty

    # Local-only (vllm serve config)
    model_path = Column(String(1000), nullable=True)   # filesystem path on the server
    max_model_len = Column(Integer, nullable=True)     # vllm --max-model-len
    extra_vllm_args = Column(JSON, default=[])         # additional vllm CLI flags as list[str]
    # Extra env vars merged into the vllm subprocess env (dict[str, str]).
    # Use this for runtime knobs that aren't CLI flags — e.g. on V100 hosts
    # with heterogeneous PCIe topology, set NCCL_P2P_DISABLE=1 here to avoid
    # NCCL's P2P probe deadlock during multi-GPU startup.
    extra_env_vars = Column(JSON, default={})
    lora_adapter_path = Column(String(1000), nullable=True)  # optional adapter dir; enables --enable-lora
    # gpu_ids semantics (mirrors training):
    #   None / missing → auto: don't override CUDA_VISIBLE_DEVICES
    #   [0]            → single GPU (CUDA_VISIBLE_DEVICES=0)
    #   [0, 1, ...]    → multi-GPU; we also auto-add --tensor-parallel-size=N
    gpu_ids = Column(JSON, nullable=True)
    port = Column(Integer, nullable=True)              # assigned at start-time
    process_pid = Column(Integer, nullable=True)
    status = Column(String(20), default="stopped")     # stopped | starting | running | failed
    error_message = Column(Text, nullable=True)
    log_file = Column(String(500), nullable=True)

    # Optional provenance link
    source_experiment_id = Column(Integer, ForeignKey("training_experiments.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationRun(Base):
    """An LLM-as-Judge evaluation comparing candidate vs (optional) baseline on a dataset."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    candidate_assistant_id = Column(Integer, ForeignKey("llm_assistants.id"))
    baseline_assistant_id = Column(Integer, ForeignKey("llm_assistants.id"), nullable=True)
    judge_assistant_id = Column(Integer, ForeignKey("llm_assistants.id"))

    sample_limit = Column(Integer, nullable=True)      # cap items to evaluate, None = full dataset
    status = Column(String(20), default="pending")     # pending|running|completed|failed|cancelled
    # Sub-state of status='running' indicating which generation/judging stage we're in.
    # Allowed: pending | generating_candidate | generating_baseline | judging | done
    phase = Column(String(30), default="pending")
    is_cancelled = Column(Boolean, default=False)

    # Aggregated 0-5 mean scores (filled in once enough items are scored)
    candidate_score = Column(Float, nullable=True)
    baseline_score = Column(Float, nullable=True)
    candidate_pass_rate = Column(Float, nullable=True)  # fraction with score >= 3
    baseline_pass_rate = Column(Float, nullable=True)

    progress_total = Column(Integer, default=0)
    progress_done = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset")
    candidate = relationship("LLMAssistant", foreign_keys=[candidate_assistant_id])
    baseline = relationship("LLMAssistant", foreign_keys=[baseline_assistant_id])
    judge = relationship("LLMAssistant", foreign_keys=[judge_assistant_id])
    items = relationship("EvaluationItem", back_populates="run")


class EvaluationItem(Base):
    __tablename__ = "evaluation_items"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    dataset_item_id = Column(Integer, ForeignKey("dataset_items.id"))

    instruction = Column(Text)
    expected_output = Column(Text)

    candidate_response = Column(Text, nullable=True)
    candidate_score = Column(Float, nullable=True)       # 0-5
    candidate_reasoning = Column(Text, nullable=True)

    baseline_response = Column(Text, nullable=True)
    baseline_score = Column(Float, nullable=True)
    baseline_reasoning = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvaluationRun", back_populates="items")
