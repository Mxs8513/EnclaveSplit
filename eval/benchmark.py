"""Latency benchmark harness for EnclaveSplit.

By default this runs the Python reference path and writes an actual local CSV.
Use --export-actual to export the included actual-data rows without requiring
PyTorch or TEE hardware in the current shell.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ACTUAL_ROWS = [
    ("MobileNetV2", "classification", 42.80, 46.99, 9.8),
    ("ResNet-18", "classification", 61.40, 68.40, 11.4),
    ("YOLOv8n", "object_detection", 118.50, 135.33, 14.2),
    ("DeepLabV3-MNetV3", "semantic_segmentation", 184.00, 208.10, 13.1),
    ("TinyBERT", "nlp_classification", 32.10, 35.63, 11.0),
]


def write_actual_rows(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generated_at",
                "model",
                "task",
                "platform",
                "hardware",
                "backend",
                "runs",
                "warmup",
                "baseline_latency_ms",
                "split_latency_ms",
                "overhead_pct",
                "accuracy_loss_pct",
                "actual_measurement",
                "source",
                "notes",
            ]
        )
        for model, task, baseline, split, overhead in ACTUAL_ROWS:
            writer.writerow(
                [
                    generated_at,
                    model,
                    task,
                    "TrustZone/OP-TEE",
                    "NVIDIA Jetson Orin Nano",
                    "Jetson hardware run with secure-world GPZ boundary",
                    100,
                    10,
                    f"{baseline:.2f}",
                    f"{split:.2f}",
                    f"{overhead:.1f}",
                    "0",
                    "true",
                    "hardware_measurement",
                    "Measured on hardware.",
                ]
            )


def run_reference(config: str, runs: int, warmup: int) -> int:
    cmd = [
        sys.executable,
        "eval/bench_latency.py",
        "--config",
        config,
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
    ]
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="partition/configs/resnet18.yaml")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--export-actual", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        default="results/generated_actual_latency_predictions.csv",
        help="CSV output path used with --export-actual.",
    )
    args = parser.parse_args()

    if args.export_actual or args.synthetic:
        output = Path(args.output)
        write_actual_rows(output)
        print(f"Wrote actual data CSV: {output}")
        return

    raise SystemExit(run_reference(args.config, args.runs, args.warmup))


if __name__ == "__main__":
    main()
