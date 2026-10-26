#!/usr/bin/env python3
"""Train a non-thinking Qwen3 QLoRA adapter, with completion-only SFT and resumable checkpoints."""

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

from common import (SYSTEM_PROMPT, load_frozen_split, model_signature, read_json, sha256_file,
                    tokenize_example, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--drop-overlong", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.max_length, args.epochs, args.learning_rate, args.gradient_accumulation,
           args.lora_rank, args.checkpoint_steps) <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("Training sizes/rates must be positive; --max-steps is -1 or positive.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This script targets one GPU; do not launch it with distributed torchrun.")
    train_rows = load_frozen_split(args.data_dir / "train.jsonl", "train")
    dev_rows = load_frozen_split(args.data_dir / "dev.jsonl", "dev")
    if {row["schema_id"] for row in train_rows} & {row["schema_id"] for row in dev_rows}:
        raise ValueError("Training and development schemas overlap.")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key != "resume_from_checkpoint"}
    config.update(train_sha256=sha256_file(args.data_dir / "train.jsonl"),
                  dev_sha256=sha256_file(args.data_dir / "dev.jsonl"), system_prompt=SYSTEM_PROMPT,
                  model_signature=model_signature(args.model_dir))
    config["versions"] = {name: version(name) for name in ("transformers", "trl", "peft", "datasets", "torch", "bitsandbytes")}
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.resolve().parent != args.output_dir.resolve():
            raise ValueError("Resume checkpoint must be a checkpoint-N directory directly under output-dir.")
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                         "adapter_config.json", "adapter_model.safetensors"):
            if not (args.resume_from_checkpoint / filename).is_file():
                raise ValueError(f"Not a full resumable checkpoint; missing {filename}")
        if read_json(args.output_dir / "run_config.json") != config:
            raise ValueError("Resume model, data, environment or training configuration changed.")
    elif args.output_dir.exists():
        raise FileExistsError("Use a new output-dir, or resume an existing full checkpoint.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires the cloud CUDA GPU environment.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    datasets, excluded = {}, []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        records = []
        for row in rows:
            encoded = tokenize_example(tokenizer, row)
            if len(encoded["input_ids"]) > args.max_length:
                excluded.append({"id": row["id"], "split": split, "tokens": len(encoded["input_ids"])})
            else:
                records.append(encoded)
        if not records:
            raise ValueError(f"No {split} samples fit the token limit.")
        datasets[split] = Dataset.from_list(records)
    if excluded and not args.drop_overlong:
        raise ValueError(f"{len(excluded)} samples exceed max-length; increase it or explicitly use --drop-overlong. Examples: {excluded[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "tokenization_report.json", {
        "retained": {name: len(value) for name, value in datasets.items()}, "excluded": excluded,
    })
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )
    model.config.use_cache = False
    settings = SFTConfig(
        output_dir=str(args.output_dir), max_length=args.max_length, packing=False,
        completion_only_loss=True, assistant_only_loss=False, bf16=bf16, fp16=not bf16,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps,
        optim="adamw_torch", warmup_ratio=0.03, lr_scheduler_type="cosine",
        eval_strategy="steps", eval_steps=args.checkpoint_steps, save_strategy="steps",
        save_steps=args.checkpoint_steps, save_total_limit=2, save_only_model=False,
        logging_steps=5, logging_dir=str(args.output_dir / "logs"), report_to="none",
        seed=42, data_seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=settings, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["dev"],
        peft_config=LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                               r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05),
    )
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")
    trainer.save_state()
    result.metrics["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    result.metrics["gpu"] = torch.cuda.get_device_name()
    trainer.save_metrics("train", result.metrics)
    write_json(args.output_dir / "evaluation_loss.json", trainer.evaluate())
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
