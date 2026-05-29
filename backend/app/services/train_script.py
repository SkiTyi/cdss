#!/usr/bin/env python3
"""
Standalone LLM fine-tuning script using HuggingFace Transformers + PEFT + TRL.

Supports single-GPU and multi-GPU (torchrun) training.
Only rank-0 process writes output; metrics are written to both stdout and
a metrics_file so the backend can monitor them regardless of launch mode.

Output format (one JSON per line):
  {"type": "progress",  "message": "..."}
  {"type": "info",      "total_steps": N, "train_samples": N, "val_samples": N}
  {"type": "metrics",   "step": N, "epoch": F, "train_loss": F, "eval_loss": F,
                        "learning_rate": F, "grad_norm": F}
  {"type": "error",     "message": "..."}
  {"type": "completed", "output_dir": "..."}
"""
import argparse
import json
import os
import sys

# Distributed training rank (set by torchrun; 0 for single-GPU)
_RANK = int(os.environ.get("RANK", "0"))
_metrics_file: str = ""   # Set from config after parsing


def emit(data: dict):
    """Write a JSON event; no-op on non-primary ranks."""
    if _RANK != 0:
        return
    line = json.dumps(data, ensure_ascii=False)
    print(line, flush=True)
    if _metrics_file:
        with open(_metrics_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main():
    global _metrics_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    # Metrics file is the authoritative output channel (works for single + multi-GPU)
    _metrics_file = config.get("metrics_file", "")

    # ------------------------------------------------------------------ config
    model_path = config["model_path"]
    output_dir = config["output_dir"]
    train_file = config["train_file"]
    val_file = config.get("val_file")

    lr = float(config.get("learning_rate", 2e-4))
    num_epochs = int(config.get("num_epochs", 3))
    batch_size = int(config.get("batch_size", 4))
    grad_accum = int(config.get("gradient_accumulation_steps", 4))
    max_seq_len = int(config.get("max_seq_length", 2048))
    warmup_ratio = float(config.get("warmup_ratio", 0.05))
    weight_decay = float(config.get("weight_decay", 0.01))
    logging_steps = int(config.get("logging_steps", 10))
    eval_steps = int(config.get("eval_steps", 50))
    save_steps = int(config.get("save_steps", 100))

    use_lora = bool(config.get("use_lora", True))
    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))
    lora_target = config.get("lora_target_modules", "all-linear")

    use_4bit = bool(config.get("use_4bit", False))
    use_bf16 = bool(config.get("use_bf16", False))

    # ------------------------------------------------------------------ imports
    emit({"type": "progress", "message": "正在导入深度学习库..."})
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            TrainerCallback,
            TrainerState,
            TrainerControl,
        )
        from datasets import Dataset as HFDataset
    except ImportError as e:
        emit({"type": "error", "message": f"缺少必要的库: {e}。请安装: pip install transformers datasets torch"})
        sys.exit(1)

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError as e:
        emit({"type": "error", "message": f"缺少 PEFT 库: {e}。请安装: pip install peft"})
        sys.exit(1)

    try:
        from trl import SFTTrainer
    except ImportError as e:
        emit({"type": "error", "message": f"缺少 trl 库: {e}。请安装: pip install trl"})
        sys.exit(1)

    # ── trl / transformers cross-version compatibility ─────────────────
    # SFTConfig was introduced in trl >= 0.12. On older trl (V100 hosts
    # often pinned to transformers ~4.46 because vllm 0.6.2 demands it,
    # and the matching trl ~0.11.x doesn't ship SFTConfig), fall back to
    # plain TrainingArguments and pass SFT-specific kwargs to the trainer
    # itself. Which kwargs are accepted is detected by inspect below.
    try:
        from trl import SFTConfig as _ArgsClass
        _ARGS_HAVE_SFT_FIELDS = True
    except ImportError:
        _ArgsClass = TrainingArguments
        _ARGS_HAVE_SFT_FIELDS = False

    # ------------------------------------------------------------------ device
    # LOCAL_RANK / WORLD_SIZE are set by torchrun for DDP; default to 0/1 for single-proc.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(f"GPU{i} {props.name} ({props.total_memory // 1024**3}GB)")
    emit({"type": "progress",
          "message": f"使用设备: {device}"
                     + (f" (LOCAL_RANK={local_rank}/{world_size})" if world_size > 1 else "")
                     + (f" — 可见: {', '.join(gpu_info)}" if gpu_info else "")})

    # ------------------------------------------------------------------ load data
    emit({"type": "progress", "message": "正在加载数据集..."})

    def load_jsonl(path: str):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return HFDataset.from_list(rows)

    train_dataset = load_jsonl(train_file)
    eval_dataset = load_jsonl(val_file) if val_file else None

    total_steps = max(1, len(train_dataset) * num_epochs // (batch_size * grad_accum))
    emit({
        "type": "info",
        "total_steps": total_steps,
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset) if eval_dataset else 0,
    })
    emit({"type": "progress",
          "message": f"训练集 {len(train_dataset)} 条，"
                     + (f"验证集 {len(eval_dataset)} 条，" if eval_dataset else "无验证集，")
                     + f"预计 {total_steps} 步"})

    # ------------------------------------------------------------------ formatting
    def format_alpaca(example: dict) -> dict:
        sys_prompt = example.get("system_prompt") or ""
        instruction = example.get("instruction") or ""
        inp = example.get("input") or ""
        output = example.get("output") or ""
        text = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n" if sys_prompt else "")
        text += f"<|im_start|>user\n{instruction}"
        if inp:
            text += f"\n{inp}"
        text += f"<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return {"text": text}

    def format_sharegpt(example: dict) -> dict:
        text = ""
        for turn in example.get("conversations", []):
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            if role in ("system",):
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role in ("user", "human"):
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ("assistant", "gpt"):
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return {"text": text}

    sample = train_dataset[0]
    is_sharegpt = "conversations" in sample
    fmt_fn = format_sharegpt if is_sharegpt else format_alpaca

    train_dataset = train_dataset.map(fmt_fn, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(fmt_fn, remove_columns=eval_dataset.column_names)

    # ------------------------------------------------------------------ model
    emit({"type": "progress", "message": f"正在加载模型: {model_path} ..."})

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        # Explicit — avoids the legacy path that materializes the whole model
        # in host RAM before moving to GPU (OOM-killed silently on small hosts).
        "low_cpu_mem_usage": True,
    }
    # Stream weights directly into the assigned GPU rather than going through
    # host RAM. Without this, transformers materializes every shard in CPU
    # memory first, which often gets OOM-killed on large (>= 7B) models even
    # when there is plenty of free GPU memory. For DDP, each rank loads its
    # own copy onto its own GPU.
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": local_rank}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            emit({"type": "progress", "message": "bitsandbytes 未安装，跳过 4-bit 量化"})

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    if use_lora:
        emit({"type": "progress", "message": f"应用 LoRA (r={lora_r}, alpha={lora_alpha})..."})
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        if isinstance(lora_target, list):
            target_modules = lora_target
        elif lora_target == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",") if m.strip()]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        emit({"type": "progress",
              "message": f"可训练参数: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)"})

    # ------------------------------------------------------------------ callback
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            epoch = round(state.epoch or 0, 4)
            m: dict = {"type": "metrics", "step": step, "epoch": epoch}
            if "loss" in logs:
                m["train_loss"] = round(logs["loss"], 6)
            if "eval_loss" in logs:
                m["eval_loss"] = round(logs["eval_loss"], 6)
            if "learning_rate" in logs:
                m["learning_rate"] = logs["learning_rate"]
            if "grad_norm" in logs:
                m["grad_norm"] = round(float(logs["grad_norm"]), 6)
            emit(m)

        def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            emit({"type": "progress",
                  "message": f"开始 Epoch {int((state.epoch or 0) + 1)} / {args.num_train_epochs}"})

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            emit({"type": "progress", "message": "训练正式开始！"})

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            emit({"type": "progress", "message": "训练循环结束，正在保存模型..."})

    # ------------------------------------------------------------------ training args
    fp16 = (not use_bf16) and torch.cuda.is_available()

    # transformers enforces save_steps % eval_steps == 0 when
    # load_best_model_at_end is True. Auto-snap save_steps up to the next
    # multiple of eval_steps to avoid that ValueError on first validation.
    if eval_dataset is not None and save_steps % eval_steps != 0:
        new_save = ((save_steps // eval_steps) + 1) * eval_steps
        emit({"type": "progress",
              "message": f"save_steps={save_steps} 不是 eval_steps={eval_steps} 的整数倍，已自动调整为 {new_save}"})
        save_steps = new_save

    # Build training-args kwargs through inspection, so the same script
    # runs on transformers 4.46 (V100/vllm0.6) and 5.x (5090/vllm-latest)
    # without two code paths. raw[] enumerates everything we'd LIKE to set;
    # only kwargs the chosen class actually accepts are kept.
    import inspect as _inspect
    _arg_params = _inspect.signature(_ArgsClass.__init__).parameters

    raw = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "fp16": fp16,
        "bf16": use_bf16,
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": eval_dataset is not None,
        "report_to": "none",
        # In-training eval only needs eval_loss (post-train token accuracy
        # bypasses Trainer.evaluate). Keeping logits off saves ~B×T×V×4
        # bytes of fp32 logits per eval batch — without it a 7B + vocab
        # 150K + seq 2048 OOMs on 32 GB during evaluation.
        "prediction_loss_only": True,
    }
    # eval_strategy was renamed from evaluation_strategy in transformers 4.39+
    if "eval_strategy" in _arg_params:
        raw["eval_strategy"] = "steps" if eval_dataset else "no"
    elif "evaluation_strategy" in _arg_params:
        raw["evaluation_strategy"] = "steps" if eval_dataset else "no"
    if eval_dataset is not None:
        raw["eval_steps"] = eval_steps
    # eval_do_concat_batches added in transformers 4.40+
    if "eval_do_concat_batches" in _arg_params:
        raw["eval_do_concat_batches"] = False

    # SFT-specific knobs live on SFTConfig (new trl). On old trl we fall
    # back to TrainingArguments — these would crash there, so we route them
    # to the SFTTrainer constructor instead (see below).
    if _ARGS_HAVE_SFT_FIELDS:
        if "max_length" in _arg_params:
            raw["max_length"] = max_seq_len          # trl >= 0.26
        elif "max_seq_length" in _arg_params:
            raw["max_seq_length"] = max_seq_len      # trl 0.12 ~ 0.25
        if "dataset_text_field" in _arg_params:
            raw["dataset_text_field"] = "text"

    final_kwargs = {k: v for k, v in raw.items() if k in _arg_params}
    training_args = _ArgsClass(**final_kwargs)

    # SFTTrainer kwargs — also varies across versions.
    _sft_params = _inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "callbacks": [MetricsCallback()],
    }
    if "processing_class" in _sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Old trl: SFT-specific kwargs land on the trainer constructor.
    if not _ARGS_HAVE_SFT_FIELDS:
        if "max_seq_length" in _sft_params:
            trainer_kwargs["max_seq_length"] = max_seq_len
        if "dataset_text_field" in _sft_params:
            trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------ eval helpers
    # Reused for both baseline (pre-train) and final (post-train) evaluation.
    import torch as _torch
    from contextlib import nullcontext as _nullcontext

    def _baseline_ctx():
        """Bypass LoRA adapters when present so we measure the bare base model."""
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                return model.disable_adapter()
        except ImportError:
            pass
        return _nullcontext()

    def _token_accuracy():
        """Return (accuracy, total_tokens) over trainer's eval dataloader."""
        model.eval()
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        eval_loader = trainer.get_eval_dataloader()
        correct = 0
        total = 0
        with _torch.no_grad():
            for batch in eval_loader:
                batch = {k: (v.to(target_device) if hasattr(v, "to") else v)
                         for k, v in batch.items()}
                labels = batch.get("labels")
                if labels is None:
                    continue
                out = model(**{k: v for k, v in batch.items() if k != "labels"})
                logits = out.logits
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels.ne(-100)
                if (hasattr(tokenizer, "pad_token_id")
                        and tokenizer.pad_token_id is not None):
                    mask = mask & shift_labels.ne(tokenizer.pad_token_id)
                correct += ((preds == shift_labels) & mask).sum().item()
                total += mask.sum().item()
        return ((correct / total) if total > 0 else None), total

    def _run_eval(label: str, baseline: bool):
        """Run trainer.evaluate() + token accuracy and return a result dict."""
        eval_loss = None
        accuracy = None
        tokens = 0
        ctx = _baseline_ctx() if baseline else _nullcontext()
        try:
            with ctx:
                ev = trainer.evaluate()
                eval_loss = ev.get("eval_loss")
        except Exception as e:
            import traceback as _tb
            emit({"type": "progress",
                  "message": f"{label} trainer.evaluate() 失败：{e}\n{_tb.format_exc()}"})
        # Need a separate `with` because disable_adapter is single-use.
        ctx2 = _baseline_ctx() if baseline else _nullcontext()
        try:
            with ctx2:
                accuracy, tokens = _token_accuracy()
        except Exception as e:
            import traceback as _tb
            emit({"type": "progress",
                  "message": f"{label} token 准确率失败：{e}\n{_tb.format_exc()}"})
        return {
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "token_accuracy": float(accuracy) if accuracy is not None else None,
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_tokens": int(tokens),
        }

    # ------------------------------------------------------------------ baseline eval (pre-train)
    if eval_dataset is not None:
        emit({"type": "progress", "message": "正在测量基座模型在验证集上的 baseline 准确率..."})
        baseline = _run_eval("baseline", baseline=True)
        emit({"type": "baseline_eval", **baseline})
        if baseline["token_accuracy"] is not None:
            emit({"type": "progress",
                  "message": f"基座 baseline：token 准确率 = {baseline['token_accuracy']*100:.2f}%"
                             + (f"  |  eval_loss = {baseline['eval_loss']:.4f}"
                                if baseline['eval_loss'] is not None else "")})

    # ------------------------------------------------------------------ train
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    emit({"type": "progress", "message": f"模型已保存至: {output_dir}"})

    # ------------------------------------------------------------------ final eval (post-train)
    if eval_dataset is not None:
        emit({"type": "progress", "message": "正在计算微调后模型在验证集上的最终准确率..."})
        final = _run_eval("final", baseline=False)
        emit({"type": "final_eval", **final})

        msg_parts = []
        if final["token_accuracy"] is not None:
            msg_parts.append(f"微调后 token 准确率 = {final['token_accuracy']*100:.2f}%")
        if final["eval_loss"] is not None:
            msg_parts.append(f"eval_loss = {final['eval_loss']:.4f}")
        if msg_parts:
            emit({"type": "progress", "message": "  |  ".join(msg_parts)})

    emit({"type": "completed", "output_dir": output_dir})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as _tb
        emit({
            "type": "error",
            "message": f"训练进程异常退出：{type(e).__name__}: {e}\n{_tb.format_exc()}",
        })
        sys.exit(1)
