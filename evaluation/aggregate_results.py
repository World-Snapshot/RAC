#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate method summaries into one comparison table.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.results.resolve().glob("*/metrics/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No */metrics/summary.json files under {args.results}")
    rows = [read_json(path) for path in summaries]
    write_json({"methods": rows}, args.output.with_suffix(".json"))
    lines = [
        "# Unified reconstruction benchmark",
        "",
        "| Method | Samples | Params (M) | NFE | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | clean-rFID ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model = row.get("model", {})
        params = model.get("unique_parameters")
        nfe = model.get("decode_nfe")
        params_text = f"{params / 1e6:.2f}" if isinstance(params, (int, float)) else "—"
        nfe_text = str(nfe) if nfe is not None else "—"
        rfid = row.get("clean_rfid")
        rfid_text = f"{rfid:.3f}" if isinstance(rfid, (int, float)) else "—"
        lpips = row.get("lpips")
        lpips_text = f"{lpips['mean']:.4f}" if isinstance(lpips, dict) else "—"
        lines.append(
            "| {method} | {count} | {params} | {nfe} | {mse:.6f} | {psnr:.3f} | "
            "{ssim:.4f} | {lpips} | {rfid} |".format(
                method=row["method"],
                count=row["count"],
                params=params_text,
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
