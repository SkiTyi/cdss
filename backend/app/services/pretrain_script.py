#!/usr/bin/env python3
"""
Standalone causal-LM continued pre-training script.

Reads a JSONL corpus produced by the backend (one {"text": "..."} per line),
tokenizes + concatenates into fixed-size blocks, and runs HF Trainer with
DataCollatorForLanguageModeling(mlm=False) for next-token prediction.

Output schema (one JSON per line, mirrors train_script.py so trainer.py and
pretrainer.py can both tail the file):
  {"type": "progress",  "message": "..."}
  {"type": "info",      "total_steps": N, "train_blocks": N, "val_blocks": N,
                        "total_tokens": N}
  {"type": "metrics",   "step": N, "epoch": F, "train_loss": F, "eval_loss": F,
                        "perplexity": F, "learning_rate": F, "grad_norm": F}
  {"type": "error",     "message": "..."}
  {"type": "completed", "output_dir": "..."}

Single-GPU and multi-GPU (torchrun) both supported; only rank-0 emits.
"""
import argparse
import json
import math
import os
import sys

# Note: LD_LIBRARY_PATH for CUDA libs (libnvJitLink.so.13 etc.) MUST be set
# by the parent before spawning this script — see services/_subprocess_env.py.
# Patching os.environ here is too late: ctypes resolves library paths at
# process startup, so this child sees the linker state the parent gave it.

_RANK = int(os.environ.get("RANK", "0"))
_metrics_file: str = ""


def emit(data: dict):
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
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    _metrics_file = config.get("metrics_file", "")

    # ─── config ────────────────────────────────────────────────────────────
    model_path = config["model_path"]
    output_dir = config["output_dir"]
    corpus_file = config["corpus_file"]                     # JSONL
    eval_corpus_file = config.get("eval_corpus_file")       # optional JSONL

    lr = float(config.get("learning_rate", 5e-5))           # CPT typically lower than SFT
    num_epochs = float(config.get("num_epochs", 1))
    batch_size = int(config.get("batch_size", 2))
    grad_accum = int(config.get("gradient_accumulation_steps", 8))
    block_size = int(config.get("block_size", 4096))        # context window for CPT
    warmup_ratio = float(config.get("warmup_ratio", 0.03))
    weight_decay = float(config.get("weight_decay", 0.01))
    logging_steps = int(config.get("logging_steps", 10))
    eval_steps = int(config.get("eval_steps", 200))
    save_steps = int(config.get("save_steps", 500))

    use_lora = bool(config.get("use_lora", True))
    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))
    lora_target = config.get("lora_target_modules", "all-linear")

    use_4bit = bool(config.get("use_4bit", False))
    use_bf16 = bool(config.get("use_bf16", False))

    # ─── imports ───────────────────────────────────────────────────────────
    emit({"type": "progress", "message": "正在导入深度学习库..."})
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer, TrainingArguments, TrainerCallback,
            TrainerState, TrainerControl,
        )
        from datasets import Dataset as HFDataset
    except ImportError as e:
        emit({"type": "error", "message": f"缺少必要库: {e}"})
        sys.exit(1)

    if use_lora or use_4bit:
        try:
            from peft import (LoraConfig, get_peft_model,
                              prepare_model_for_kbit_training, TaskType)
        except ImportError as e:
            emit({"type": "error", "message": f"缺少 PEFT 库: {e}"})
            sys.exit(1)

    # ─── device ────────────────────────────────────────────────────────────
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

    # ─── tokenizer ─────────────────────────────────────────────────────────
    emit({"type": "progress", "message": f"正在加载 tokenizer: {model_path} ..."})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # ─── load + tokenize + chunk ───────────────────────────────────────────
    def _load_jsonl(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        if d.get("text"):
                            rows.append(d["text"])
                    except json.JSONDecodeError:
                        continue
        return rows

    def _build_blocks(texts):
        """Tokenize all docs, separate with EOS, split into block_size chunks."""
        emit({"type": "progress",
              "message": f"开始 tokenize {len(texts)} 篇文档..."})
        all_ids = []
        report_every = max(1, len(texts) // 20)
        for idx, t in enumerate(texts):
            ids = tokenizer.encode(t, add_special_tokens=False)
            all_ids.extend(ids)
            if eos_id is not None:
                all_ids.append(eos_id)
            if (idx + 1) % report_every == 0:
                emit({"type": "progress",
                      "message": f"已 tokenize {idx+1}/{len(texts)} ({len(all_ids):,} tokens)"})
        # chunk
        n_blocks = len(all_ids) // block_size
        all_ids = all_ids[: n_blocks * block_size]
        blocks = [all_ids[i * block_size:(i + 1) * block_size]
                  for i in range(n_blocks)]
        return blocks, len(all_ids)

    train_texts = _load_jsonl(corpus_file)
    if not train_texts:
        emit({"type": "error", "message": "训练 corpus 为空"})
        sys.exit(1)
    train_blocks, train_token_count = _build_blocks(train_texts)
    if not train_blocks:
        emit({"type": "error",
              "message": f"corpus tokenize 后不足一个 block ({block_size} tokens)，请减小 block_size 或扩充语料"})
        sys.exit(1)
    train_ds = HFDataset.from_dict({"input_ids": train_blocks})

    eval_ds = None
    eval_token_count = 0
    if eval_corpus_file and os.path.exists(eval_corpus_file):
        eval_texts = _load_jsonl(eval_corpus_file)
        if eval_texts:
            eval_blocks, eval_token_count = _build_blocks(eval_texts)
            if eval_blocks:
                eval_ds = HFDataset.from_dict({"input_ids": eval_blocks})

    total_steps = max(1, math.ceil(len(train_blocks) * num_epochs / (batch_size * grad_accum)))
    emit({
        "type": "info",
        "total_steps": total_steps,
        "train_blocks": len(train_blocks),
        "val_blocks": len(eval_ds) if eval_ds else 0,
        "total_tokens": train_token_count + eval_token_count,
        "block_size": block_size,
    })
    emit({"type": "progress",
          "message": f"训练块 {len(train_blocks)} 条 × {block_size} tokens"
                     + (f"，验证块 {len(eval_ds)} 条" if eval_ds else "，无验证集")
                     + f"，预计 {total_steps} 步"})

    # ─── model ─────────────────────────────────────────────────────────────
    emit({"type": "progress", "message": f"正在加载模型: {model_path} ..."})
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "low_cpu_mem_usage": True,
    }
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
            emit({"type": "progress", "message": "bitsandbytes 未安装，跳过 4-bit"})

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    if use_lora:
        emit({"type": "progress", "message": f"应用 LoRA (r={lora_r}, alpha={lora_alpha})"})
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        if isinstance(lora_target, list):
            target_modules = lora_target
        elif lora_target == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM, target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        emit({"type": "progress",
              "message": f"可训练参数: {trainable:,} / {total_params:,} "
                         f"({100*trainable/total_params:.2f}%)"})

    # ─── callbacks ─────────────────────────────────────────────────────────
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if not logs:
                return
            m = {"type": "metrics", "step": state.global_step, "epoch": round(state.epoch or 0, 4)}
            if "loss" in logs:
                m["train_loss"] = round(logs["loss"], 6)
            if "eval_loss" in logs:
                m["eval_loss"] = round(logs["eval_loss"], 6)
                # perplexity = exp(eval_loss); cap to avoid math overflow on broken runs
                try:
                    m["perplexity"] = round(math.exp(min(logs["eval_loss"], 30)), 4)
                except Exception:
                    pass
            if "learning_rate" in logs:
                m["learning_rate"] = logs["learning_rate"]
            if "grad_norm" in logs:
                m["grad_norm"] = round(float(logs["grad_norm"]), 6)
            emit(m)

        def on_epoch_begin(self, args, state, control, **kwargs):
            emit({"type": "progress",
                  "message": f"开始 Epoch {int((state.epoch or 0) + 1)} / {args.num_train_epochs}"})

        def on_train_begin(self, args, state, control, **kwargs):
            emit({"type": "progress", "message": "增量预训练正式开始"})

    # ─── training args ─────────────────────────────────────────────────────
    fp16 = (not use_bf16) and torch.cuda.is_available()

    # transformers enforces save_steps % eval_steps == 0 when
    # load_best_model_at_end is True. Auto-snap save_steps up to the next
    # multiple of eval_steps to avoid that ValueError on first validation.
    if eval_ds is not None and save_steps % eval_steps != 0:
        new_save = ((save_steps // eval_steps) + 1) * eval_steps
        emit({"type": "progress",
              "message": f"save_steps={save_steps} 不是 eval_steps={eval_steps} 的整数倍，已自动调整为 {new_save}"})
        save_steps = new_save

    # Build TrainingArguments through inspect — kwarg names drift across
    # transformers versions (eval_strategy renamed from evaluation_strategy
    # in 4.39; eval_do_concat_batches added in 4.40). V100 hosts often pin
    # to ~4.46 for vllm 0.6.2 compat, so this script must run on both.
    import inspect as _inspect
    _ta_params = _inspect.signature(TrainingArguments.__init__).parameters

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
        "load_best_model_at_end": eval_ds is not None,
        "report_to": "none",
        "gradient_checkpointing": True,    # save memory for long blocks
        # CPT only needs eval_loss / perplexity; without prediction_loss_only
        # Trainer caches (B, seq_len, vocab) fp32 logits across eval batches
        # → 18+ GB OOM on a 7B + vocab 152K + block 4096 setup.
        "prediction_loss_only": True,
    }
    if "eval_strategy" in _ta_params:
        raw["eval_strategy"] = "steps" if eval_ds else "no"
    elif "evaluation_strategy" in _ta_params:
        raw["evaluation_strategy"] = "steps" if eval_ds else "no"
    if eval_ds is not None:
        raw["eval_steps"] = eval_steps
    if "eval_do_concat_batches" in _ta_params:
        raw["eval_do_concat_batches"] = False

    training_args = TrainingArguments(**{k: v for k, v in raw.items() if k in _ta_params})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[MetricsCallback()],
    )

    # ─── train ─────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    emit({"type": "progress", "message": f"模型已保存至: {output_dir}"})

    # ─── final eval ────────────────────────────────────────────────────────
    if eval_ds is not None:
        try:
            emit({"type": "progress", "message": "正在计算最终验证集 loss / perplexity..."})
            res = trainer.evaluate()
            eval_loss = res.get("eval_loss")
            ppl = None
            if eval_loss is not None:
                try:
                    ppl = round(math.exp(min(eval_loss, 30)), 4)
                except Exception:
                    pass
            emit({"type": "final_eval",
                  "eval_loss": float(eval_loss) if eval_loss is not None else None,
                  "perplexity": ppl,
                  "val_blocks": len(eval_ds)})
            if eval_loss is not None:
                emit({"type": "progress",
                      "message": f"最终 eval_loss = {eval_loss:.4f}"
                                 + (f"，perplexity = {ppl}" if ppl else "")})
        except Exception as e:
            import traceback as _tb
            emit({"type": "progress",
                  "message": f"最终评估失败：{e}\n{_tb.format_exc()}"})

    emit({"type": "completed", "output_dir": output_dir})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as _tb
        emit({"type": "error",
              "message": f"预训练进程异常退出：{type(e).__name__}: {e}\n{_tb.format_exc()}"})
        sys.exit(1)
