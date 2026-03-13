#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference/evaluation for Qwen/Qwen3-VL-2B-Instruct on one or more JSON datasets."
    )
    parser.add_argument(
        "--input-jsons",
        nargs="+",
        required=True,
        help="One or more dataset JSON files (same format as your generated files).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write predictions and summary statistics.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model path or HF model id.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter/checkpoint path. Omit for baseline inference.",
    )
    parser.add_argument("--split-name", default="eval", help="Name stored in summary.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto", help='Transformers device_map, e.g. "auto", "cuda:0".')
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],
        help="Optional attention implementation.",
    )
    return parser.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return "auto"


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_first_float(text: str) -> Optional[float]:
    match = FLOAT_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_user_prompt(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.replace("<image>", "").strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return " ".join(text_parts).strip()
    return ""


def extract_assistant_answer(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                return " ".join(text_parts).strip()
    return ""


def load_samples(paths: List[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            print(f"[WARN] Input file not found, skipping: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Dataset file must contain a list: {path}")

        parent = path.parent
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            images = copied.get("images", [])
            if isinstance(images, list):
                fixed_images = []
                for img in images:
                    img_path = Path(str(img))
                    if not img_path.is_absolute():
                        img_path = (parent / img_path).resolve()
                    fixed_images.append(str(img_path))
                copied["images"] = fixed_images
            copied["_source_file"] = str(path)
            copied["_source_index"] = idx
            merged.append(copied)
    return merged


def build_user_message(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = extract_user_prompt(sample.get("messages", []))
    images = sample.get("images", [])
    content: List[Dict[str, Any]] = []
    for img_path in images:
        content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model_and_processor(args: argparse.Namespace):
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError("`peft` is required when --adapter-path is used.") from e
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, processor


def safe_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def run_inference(
    model,
    processor,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    exact_matches = 0
    numeric_pairs = 0
    abs_errors: List[float] = []
    sq_errors: List[float] = []
    signed_errors: List[float] = []
    total_time = 0.0

    if not samples:
        return [], {
            "split_name": args.split_name,
            "num_samples": 0,
            "num_exact_match": 0,
            "exact_match_rate": 0.0,
            "num_numeric_pairs": 0,
            "numeric_coverage": 0.0,
            "mae": None,
            "rmse": None,
            "mean_signed_error": None,
            "avg_inference_seconds": None,
        }

    for idx, sample in enumerate(samples):
        gt_answer = extract_assistant_answer(sample.get("messages", []))
        chat_messages = build_user_message(sample)

        inputs = processor.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = safe_to_device(inputs, model.device)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
        }
        if args.do_sample:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p

        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generate_kwargs)
        elapsed = time.time() - start
        total_time += elapsed

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        pred_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        is_exact = normalize_text(pred_text) == normalize_text(gt_answer)
        if is_exact:
            exact_matches += 1

        gt_value = extract_first_float(gt_answer)
        pred_value = extract_first_float(pred_text)
        abs_error = None
        if gt_value is not None and pred_value is not None:
            numeric_pairs += 1
            error = pred_value - gt_value
            abs_error = abs(error)
            abs_errors.append(abs_error)
            sq_errors.append(error * error)
            signed_errors.append(error)

        predictions.append(
            {
                "sample_index": idx,
                "source_file": sample.get("_source_file"),
                "source_index": sample.get("_source_index"),
                "images": sample.get("images", []),
                "user_prompt": extract_user_prompt(sample.get("messages", [])),
                "ground_truth": gt_answer,
                "prediction": pred_text,
                "ground_truth_value": gt_value,
                "prediction_value": pred_value,
                "abs_error": abs_error,
                "exact_match": is_exact,
                "inference_seconds": elapsed,
            }
        )

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(samples)} samples")

    num_samples = len(samples)
    summary = {
        "split_name": args.split_name,
        "num_samples": num_samples,
        "num_exact_match": exact_matches,
        "exact_match_rate": exact_matches / num_samples,
        "num_numeric_pairs": numeric_pairs,
        "numeric_coverage": numeric_pairs / num_samples,
        "mae": (sum(abs_errors) / numeric_pairs) if numeric_pairs > 0 else None,
        "rmse": math.sqrt(sum(sq_errors) / numeric_pairs) if numeric_pairs > 0 else None,
        "mean_signed_error": (sum(signed_errors) / numeric_pairs) if numeric_pairs > 0 else None,
        "avg_inference_seconds": total_time / num_samples,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "generated_at_unix": int(time.time()),
    }
    return predictions, summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    samples = load_samples(args.input_jsons)
    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"[INFO] Loaded {len(samples)} total samples")
    model, processor = load_model_and_processor(args)
    predictions, summary = run_inference(model, processor, samples, args)

    predictions_path = Path(args.output_dir) / "predictions.jsonl"
    summary_path = Path(args.output_dir) / "summary.json"

    with predictions_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Predictions written to: {predictions_path}")
    print(f"[INFO] Summary written to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
