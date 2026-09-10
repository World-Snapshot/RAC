#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate method summaries into one comparison table.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-seen", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.results.resolve().glob("*/metrics/summary.json"))
    if not summaries:
        summaries = sorted(args.results.resolve().glob("*_metrics/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No metric summaries found under {args.results}")
    rows = [read_json(path) for path in summaries]
    write_json(
        {
            "protocol": {
                "images_seen": args.images_seen,
                "optimizer_steps": args.optimizer_steps,
            },
            "methods": rows,
        },
        args.output.with_suffix(".json"),
    )
    lines = [
        "# Unified reconstruction benchmark",
        "",
        f"Images seen: {args.images_seen or '—'}; optimizer steps: {args.optimizer_steps or '—'}.",
        "",
        "| Method | Samples | Trainable (M) | Total (M) | NFE | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | clean-rFID ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model = row.get("model", {})
        trainable = model.get("trainable_parameters", model.get("decoder_parameters"))
        total = model.get("unique_parameters", model.get("unique_parameters_with_teacher"))
        nfe = model.get("decode_nfe")
        trainable_text = f"{trainable / 1e6:.2f}" if isinstance(trainable, (int, float)) else "—"
        total_text = f"{total / 1e6:.2f}" if isinstance(total, (int, float)) else "—"
        nfe_text = str(nfe) if nfe is not None else "—"
        rfid = row.get("clean_rfid")
        rfid_text = f"{rfid:.3f}" if isinstance(rfid, (int, float)) else "—"
        lpips = row.get("lpips")
        lpips_text = f"{lpips['mean']:.4f}" if isinstance(lpips, dict) else "—"
        lines.append(
            "| {method} | {count} | {trainable} | {total} | {nfe} | {mse:.6f} | {psnr:.3f} | "
            "{ssim:.4f} | {lpips} | {rfid} |".format(
                method=row["method"],
                count=row["count"],
                trainable=trainable_text,
                total=total_text,
                nfe=nfe_text,
                mse=row["metrics"]["mse"]["mean"],
                psnr=row["metrics"]["psnr"]["mean"],
                ssim=row["metrics"]["ssim"]["mean"],
                lpips=lpips_text,
                rfid=rfid_text,
            )
        )
    lines.extend(
        [
            "",
            "Raw training losses are intentionally excluded because objectives and normalizations differ.",
            "Report the exact sample count; use ImageNet validation 50K for directly comparable rFID.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
