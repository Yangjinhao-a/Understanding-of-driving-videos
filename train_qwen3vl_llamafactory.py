#!/usr/bin/env python3
import argparse
import glob
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def bool_to_str(value: bool) -> str:
    return "true" if value else "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare data splits (optional) and run LLaMA-Factory SFT for Qwen3.5 models, "
            "with optional post-training inference."
        )
    )
    parser.add_argument(
        "--input-jsons",
        nargs="+",
        default=None,
        help=(
            "One or more input JSON dataset files. "
            "If omitted, training will use --dataset-dir/--dataset directly."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default="data",
        help=(
            "Existing LLaMA-Factory dataset directory. "
            "If omitted and --input-jsons is provided, auto-generated prepared data is used."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="data_two,data_one",
        help="LLaMA-Factory dataset name(s) passed to --dataset.",
    )
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help="Optional LLaMA-Factory eval dataset name(s) passed to --eval_dataset.",
    )
    parser.add_argument(
        "--work-dir",
        default="runs/qwen3_5_llamafactory",
        help="Working directory for prepared data, outputs, and evaluation artifacts.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="/root/autodl-tmp/Qwen/Qwen3.5-9B-Base",
        help="Base model path or HF model id.",
    )
    parser.add_argument(
        "--llamafactory-cli",
        default=None,
        help=(
            "Optional LLaMA-Factory CLI executable. "
            "If not set, auto-detects `llamafactory-cli` or falls back to `python -m llamafactory.cli`."
        ),
    )
    parser.add_argument(
        "--template",
        default="qwen3_5",
        help="LLaMA-Factory template name for the model.",
    )
    parser.add_argument(
        "--finetuning-type",
        default="lora",
        choices=["lora", "full", "freeze"],
        help="Fine-tuning mode for LLaMA-Factory.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=150000)
    parser.add_argument("--preprocessing-num-workers", type=int, default=24)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--bf16",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable bf16 (true/false).",
    )
    parser.add_argument("--flash-attn", default="auto", help='LLaMA-Factory --flash_attn value, e.g. "auto".')
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target",
        default="all",
        help="LLaMA-Factory LoRA target modules (use 'all' for auto).",
    )
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["no", "epoch", "steps"],
        help="LLaMA-Factory save strategy.",
    )
    parser.add_argument(
        "--eval-strategy",
        default="no",
        choices=["no", "epoch", "steps"],
        help="LLaMA-Factory eval strategy.",
    )
    parser.add_argument(
        "--packing",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable packing (true/false).",
    )
    parser.add_argument(
        "--enable-thinking",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable thinking mode for supported models (true/false).",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--plot-loss",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to save loss curves (true/false).",
    )
    parser.add_argument(
        "--trust-remote-code",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Trust remote model code (true/false).",
    )
    parser.add_argument("--ddp-timeout", type=int, default=180000000)
    parser.add_argument(
        "--include-num-input-tokens-seen",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Track seen input tokens (true/false).",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument(
        "--overwrite-cache",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Overwrite cache files (true/false).",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Overwrite output dir (true/false).",
    )
    parser.add_argument(
        "--freeze-vision-tower",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Freeze vision tower when present (true/false).",
    )
    parser.add_argument(
        "--freeze-multi-modal-projector",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Freeze multimodal projector when present (true/false).",
    )
    parser.add_argument("--image-max-pixels", type=int, default=126464)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--video-max-pixels", type=int, default=65536)
    parser.add_argument("--video-min-pixels", type=int, default=256)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name; default is timestamp-based.",
    )
    parser.add_argument(
        "--inference-script",
        default="run_qwen3vl_inference.py",
        help="Path to standalone inference script.",
    )
    parser.add_argument("--inference-max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-post-inference", action="store_true")
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run baseline (no adapter) on val/test splits.",
    )
    return parser.parse_args()


def load_and_merge(paths: Sequence[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser().resolve()
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
                normalized_images = []
                for img in images:
                    img_path = Path(str(img))
                    if not img_path.is_absolute():
                        img_path = (parent / img_path).resolve()
                    normalized_images.append(str(img_path))
                copied["images"] = normalized_images
            copied["_source_file"] = str(path)
            copied["_source_index"] = idx
            merged.append(copied)
    return merged


def split_data(data: List[Dict[str, Any]], seed: int):
    if not data:
        raise ValueError("No valid samples were loaded from --input-jsons.")

    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    n_test = n - n_train - n_val

    # Ensure at least 1 sample per split when possible.
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        while (n_train + n_val + n_test) > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        while (n_train + n_val + n_test) < n:
            n_train += 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def strip_internal_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for row in rows:
        copied = {k: v for k, v in row.items() if not k.startswith("_")}
        cleaned.append(copied)
    return cleaned


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_command(cmd: List[str], cwd: Path) -> None:
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_llamafactory_cmd(cli_override: str = None) -> List[str]:
    if cli_override:
        return [cli_override]

    cli_path = shutil.which("llamafactory-cli")
    if cli_path:
        return [cli_path]

    if importlib.util.find_spec("llamafactory") is not None:
        return [sys.executable, "-m", "llamafactory.cli"]

    raise FileNotFoundError(
        "Cannot find LLaMA-Factory CLI. Install it first, for example:\n"
        "  pip install -U llamafactory\n"
        "Then re-run, or pass --llamafactory-cli /path/to/llamafactory-cli."
    )


def list_checkpoints(output_dir: Path) -> List[Path]:
    candidates = []
    for ckpt in glob.glob(str(output_dir / "checkpoint-*")):
        p = Path(ckpt)
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, p))
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


def run_inference(
    inference_script: Path,
    split_json: Path,
    output_dir: Path,
    model_name_or_path: str,
    max_new_tokens: int,
    split_name: str,
    adapter_path: str = "",
) -> None:
    cmd = [
        sys.executable,
        str(inference_script),
        "--input-jsons",
        str(split_json),
        "--output-dir",
        str(output_dir),
        "--model-name-or-path",
        model_name_or_path,
        "--split-name",
        split_name,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    run_command(cmd, cwd=Path.cwd())


def main() -> None:
    args = parse_args()
    start_time = time.time()

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_dir = work_dir / "prepared_data"
    output_dir = work_dir / run_name / "llamafactory_output"
    eval_dir = work_dir / run_name / "evaluation"
    inference_script = Path(args.inference_script).expanduser().resolve()

    if not inference_script.exists():
        print(
            f"[WARN] Inference script not found: {inference_script}. "
            "Post inference will be skipped."
        )
        args.skip_post_inference = True

    train_json: Optional[Path] = None
    val_json: Optional[Path] = None
    test_json: Optional[Path] = None
    split_stats_path = work_dir / run_name / "split_stats.json"

    if args.input_jsons:
        merged = load_and_merge(args.input_jsons)
        if args.max_samples is not None:
            merged = merged[: args.max_samples]
        print(f"[INFO] Loaded {len(merged)} samples from {len(args.input_jsons)} input files")

        train_rows, val_rows, test_rows = split_data(merged, args.seed)
        train_clean = strip_internal_fields(train_rows)
        val_clean = strip_internal_fields(val_rows)
        test_clean = strip_internal_fields(test_rows)

        train_json = data_dir / "train.json"
        val_json = data_dir / "validation.json"
        test_json = data_dir / "test.json"
        dataset_info_json = data_dir / "dataset_info.json"

        write_json(train_json, train_clean)
        write_json(val_json, val_clean)
        write_json(test_json, test_clean)
        write_json(
            dataset_info_json,
            {
                "qwen3vl_train": {
                    "file_name": "train.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                        "system_tag": "system",
                    },
                },
                "qwen3vl_val": {
                    "file_name": "validation.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                        "system_tag": "system",
                    },
                },
                "qwen3vl_test": {
                    "file_name": "test.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                        "system_tag": "system",
                    },
                },
            },
        )

        print("[INFO] --input-jsons mode uses prepared datasets `qwen3vl_train` and `qwen3vl_val`.")

        dataset_dir_for_training = data_dir
        dataset_name = "qwen3vl_train"
        eval_dataset_name = "qwen3vl_val"

        split_stats = {
            "mode": "prepared_from_input_jsons",
            "total_samples": len(merged),
            "train_samples": len(train_clean),
            "validation_samples": len(val_clean),
            "test_samples": len(test_clean),
            "ratios": {
                "train": len(train_clean) / len(merged),
                "validation": len(val_clean) / len(merged),
                "test": len(test_clean) / len(merged),
            },
            "seed": args.seed,
            "input_jsons": [str(Path(p).expanduser().resolve()) for p in args.input_jsons],
        }
        write_json(split_stats_path, split_stats)
    else:
        if args.dataset_dir is None:
            raise ValueError("Either --input-jsons or --dataset-dir must be provided.")

        dataset_dir_for_training = Path(args.dataset_dir).expanduser().resolve()
        if not dataset_dir_for_training.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir_for_training}")
        dataset_name = args.dataset
        eval_dataset_name = args.eval_dataset

        if not args.skip_post_inference:
            print("[WARN] Disabling post inference because --input-jsons was not provided.")
            args.skip_post_inference = True

        split_stats = {
            "mode": "existing_llamafactory_dataset",
            "dataset_dir": str(dataset_dir_for_training),
            "dataset": dataset_name,
            "eval_dataset": eval_dataset_name,
            "seed": args.seed,
        }
        write_json(split_stats_path, split_stats)

    training_command: List[str] = []
    checkpoints: List[Path] = []

    if not args.skip_training:
        llf_cmd_prefix = resolve_llamafactory_cmd(args.llamafactory_cli)
        training_command = [
            *llf_cmd_prefix,
            "train",
            "--stage",
            "sft",
            "--do_train",
            "true",
            "--model_name_or_path",
            args.model_name_or_path,
            "--template",
            args.template,
            "--dataset_dir",
            str(dataset_dir_for_training),
            "--dataset",
            dataset_name,
            "--cutoff_len",
            str(args.cutoff_len),
            "--preprocessing_num_workers",
            str(args.preprocessing_num_workers),
            "--output_dir",
            str(output_dir),
            "--logging_steps",
            str(args.logging_steps),
            "--save_strategy",
            args.save_strategy,
            "--eval_strategy",
            args.eval_strategy,
            "--per_device_train_batch_size",
            str(args.per_device_train_batch_size),
            "--per_device_eval_batch_size",
            str(args.per_device_eval_batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--num_train_epochs",
            str(args.num_train_epochs),
            "--lr_scheduler_type",
            args.lr_scheduler_type,
            "--warmup_steps",
            str(args.warmup_steps),
            "--max_grad_norm",
            str(args.max_grad_norm),
            "--plot_loss",
            bool_to_str(args.plot_loss),
            "--packing",
            bool_to_str(args.packing),
            "--enable_thinking",
            bool_to_str(args.enable_thinking),
            "--report_to",
            args.report_to,
            "--overwrite_cache",
            bool_to_str(args.overwrite_cache),
            "--overwrite_output_dir",
            bool_to_str(args.overwrite_output_dir),
            "--flash_attn",
            str(args.flash_attn),
            "--seed",
            str(args.seed),
            "--finetuning_type",
            args.finetuning_type,
            "--bf16",
            bool_to_str(args.bf16),
            "--trust_remote_code",
            bool_to_str(args.trust_remote_code),
            "--ddp_timeout",
            str(args.ddp_timeout),
            "--include_num_input_tokens_seen",
            bool_to_str(args.include_num_input_tokens_seen),
            "--optim",
            args.optim,
            "--freeze_vision_tower",
            bool_to_str(args.freeze_vision_tower),
            "--freeze_multi_modal_projector",
            bool_to_str(args.freeze_multi_modal_projector),
            "--image_max_pixels",
            str(args.image_max_pixels),
            "--image_min_pixels",
            str(args.image_min_pixels),
            "--video_max_pixels",
            str(args.video_max_pixels),
            "--video_min_pixels",
            str(args.video_min_pixels),
        ]
        if args.eval_strategy != "no" and eval_dataset_name:
            training_command.extend(["--eval_dataset", eval_dataset_name])
        if args.save_steps is not None:
            training_command.extend(["--save_steps", str(args.save_steps)])
        if args.save_total_limit is not None:
            training_command.extend(["--save_total_limit", str(args.save_total_limit)])
        if args.max_samples is not None:
            training_command.extend(["--max_samples", str(args.max_samples)])

        if args.finetuning_type == "lora":
            training_command.extend(
                [
                    "--lora_rank",
                    str(args.lora_rank),
                    "--lora_alpha",
                    str(args.lora_alpha),
                    "--lora_dropout",
                    str(args.lora_dropout),
                    "--lora_target",
                    args.lora_target,
                ]
            )

        run_command(training_command, cwd=Path.cwd())
        checkpoints = list_checkpoints(output_dir)
        print(f"[INFO] Found {len(checkpoints)} checkpoints in {output_dir}")
    else:
        checkpoints = list_checkpoints(output_dir)
        print("[INFO] --skip-training enabled, using existing checkpoints if available.")

    val_epoch1_out = eval_dir / "val_after_epoch1"
    test_final_out = eval_dir / "test_after_full_training"
    baseline_val_out = eval_dir / "baseline_val"
    baseline_test_out = eval_dir / "baseline_test"

    epoch1_adapter = ""
    final_adapter = ""
    if checkpoints:
        epoch1_adapter = str(checkpoints[0])
        final_adapter = str(checkpoints[-1])
    elif output_dir.exists():
        # LLaMA-Factory may save adapter files directly under output_dir.
        final_adapter = str(output_dir)
        epoch1_adapter = str(output_dir)

    if not args.skip_post_inference and "qwen3_vl" not in args.template.lower():
        print(
            "[WARN] Post inference uses run_qwen3vl_inference.py (VL-specific). "
            "Skipping because template is not qwen3_vl."
        )
        args.skip_post_inference = True

    if not args.skip_post_inference:
        if val_json is None or test_json is None:
            print("[WARN] Validation/test JSONs are unavailable; skipping post inference.")
            args.skip_post_inference = True

    if not args.skip_post_inference:
        if args.run_baseline:
            run_inference(
                inference_script=inference_script,
                split_json=val_json,
                output_dir=baseline_val_out,
                model_name_or_path=args.model_name_or_path,
                max_new_tokens=args.inference_max_new_tokens,
                split_name="baseline_validation",
            )
            run_inference(
                inference_script=inference_script,
                split_json=test_json,
                output_dir=baseline_test_out,
                model_name_or_path=args.model_name_or_path,
                max_new_tokens=args.inference_max_new_tokens,
                split_name="baseline_test",
            )

        if epoch1_adapter:
            run_inference(
                inference_script=inference_script,
                split_json=val_json,
                output_dir=val_epoch1_out,
                model_name_or_path=args.model_name_or_path,
                max_new_tokens=args.inference_max_new_tokens,
                split_name="validation_after_epoch1",
                adapter_path=epoch1_adapter,
            )
        else:
            print("[WARN] No checkpoint found for validation-after-epoch1 inference.")

        if final_adapter:
            run_inference(
                inference_script=inference_script,
                split_json=test_json,
                output_dir=test_final_out,
                model_name_or_path=args.model_name_or_path,
                max_new_tokens=args.inference_max_new_tokens,
                split_name="test_after_full_training",
                adapter_path=final_adapter,
            )
        else:
            print("[WARN] No checkpoint found for final test inference.")

    summary = {
        "run_name": run_name,
        "work_dir": str(work_dir),
        "prepared_data_dir": str(data_dir) if args.input_jsons else None,
        "dataset_dir_for_training": str(dataset_dir_for_training),
        "dataset": dataset_name,
        "eval_dataset": eval_dataset_name,
        "llamafactory_output_dir": str(output_dir),
        "evaluation_dir": str(eval_dir),
        "training_command": training_command,
        "checkpoints": [str(p) for p in checkpoints],
        "epoch1_adapter_used": epoch1_adapter or None,
        "final_adapter_used": final_adapter or None,
        "split_stats_path": str(split_stats_path),
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(work_dir / run_name / "run_summary.json", summary)

    print("[INFO] Run completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
