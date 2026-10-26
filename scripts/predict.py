#!/usr/bin/env python3
"""Generate reproducible baseline/adapter predictions from a frozen split on one GPU."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from common import (SYSTEM_PROMPT, load_frozen_split, load_jsonl, messages_for, model_signature, read_json,
                    sha256_file, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--prompt-style", choices=("json", "xiyan"), default="json")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_input_tokens, args.max_new_tokens) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Token limits and --limit must be positive")
    rows = load_frozen_split(args.data_file)
    if args.prompt_style == "xiyan":
        rows = [row for row in rows if row["expected"]["action"] == "sql"]
    if args.limit:
        rows = rows[:args.limit]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires the planned CUDA GPU environment.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metadata = {
        "model_dir": str(args.model_dir.resolve()), "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
        "data_sha256": sha256_file(args.data_file), "prompt_style": args.prompt_style,
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "inference_code_sha256": sha256_file(Path(__file__)),
        "generation_profile": {
            "do_sample": False, "load_in_4bit": True, "quant_type": "nf4", "double_quant": True,
            "dtype": str(dtype), "attn_implementation": "sdpa", "use_cache": True,
            "versions": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")},
            "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        },
    }
    if args.adapter_dir:
        metadata["adapter_sha256"] = sha256_file(args.adapter_dir / "adapter_model.safetensors")
        metadata["adapter_config_sha256"] = sha256_file(args.adapter_dir / "adapter_config.json")
    metadata["model_signature"] = model_signature(args.model_dir)
    metadata_file = args.output_file.with_suffix(args.output_file.suffix + ".meta.json")
    completed = set()
    if args.output_file.exists():
        if not args.resume or read_json(metadata_file) != metadata:
            raise ValueError("Prediction output exists or resume configuration changed; use a new output path.")
        previous = load_jsonl(args.output_file)
        if any(row["id"] not in {item["id"] for item in rows} for row in previous):
            raise ValueError("Previous predictions do not belong to this selected split/subset.")
        completed = {row["id"] for row in previous}
    elif args.resume:
        raise FileNotFoundError("--resume needs an existing prediction file and metadata.")
    else:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(metadata_file, metadata)
    if completed == {row["id"] for row in rows}:
        print("All selected examples already have predictions.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    with args.output_file.open("a" if args.resume else "x", encoding="utf-8") as stream:
        for row in rows:
            if row["id"] in completed:
                continue
            started = time.perf_counter()
            record = {"id": row["id"], "response": "", "response_style": "json" if args.prompt_style == "json" else "sql"}
            try:
                if args.prompt_style == "json":
                    prompt = tokenizer.apply_chat_template(messages_for(row), tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt = ("你是一名SQLite专家，根据数据库schema和参考信息生成只读SQL。\n〖用户问题〗\n"
                              + row["question"] + "\n〖数据库schema〗\n" + row["context"]["schema"]
                              + "\n〖参考信息〗\n" + json.dumps({k: v for k, v in row["context"].items() if k != "schema"}, ensure_ascii=False)
                              + "\n〖用户问题〗\n" + row["question"] + "\n```sql\n")
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                length = inputs["input_ids"].shape[1]
                if length > args.max_input_tokens:
                    raise ValueError(f"Prompt length {length} exceeds {args.max_input_tokens}; no truncation was applied.")
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                record["response"] = tokenizer.decode(output[0, length:], skip_special_tokens=True)
                if args.prompt_style == "xiyan":
                    record["response"] = record["response"].split("```", 1)[0].strip()
            except torch.cuda.OutOfMemoryError:
                raise
            except ValueError as error:
                record["error"] = str(error)
            record["seconds"] = time.perf_counter() - started
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            completed.add(row["id"])
            if len(completed) % 10 == 0:
                print(f"Predicted {len(completed)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
