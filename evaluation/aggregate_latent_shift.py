#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate latent-shift summaries.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.results.resolve().glob("*/summary.json"))
    payloads = []
    for path in summaries:
        payload = read_json(path)
        if "rows" in payload:
            payloads.append(payload)
    if not payloads:
        raise FileNotFoundError(f"No latent-shift */summary.json files under {args.results}.")
    signatures = {
        (payload["samples"], tuple(row["sigma"] for row in payload["rows"]))
        for payload in payloads
    }
    if len(signatures) != 1:
        raise RuntimeError(f"Methods do not share sample count and sigma grid: {signatures}")

    methods = []
    lines = [
        "# Unified latent-shift robustness",
        "",
        "| Method | sigma | PSNR ↑ | ΔPSNR vs. clean ↑ | SSIM ↑ | Output drift MSE ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        methods.append(payload)
        for row in payload["rows"]:
            metrics = row["metrics"]
            lines.append(
                f"| {payload['method']} | {row['sigma']:.2f} | "
                f"{metrics['psnr']['mean']:.3f} ± {metrics['psnr']['ci95']:.3f} | "
                f"{metrics['delta_psnr_vs_clean']['mean']:+.3f} | "
                f"{metrics['ssim']['mean']:.4f} | "
                f"{metrics['output_drift_mse']['mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "All methods use the same images, sigma grid, and paired standard-normal directions.",
            "SSDD reuses identical decoder noise for its clean and perturbed output.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    write_json({"methods": methods}, args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
