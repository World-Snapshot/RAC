#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import image_paths, load_image, paired_metrics, read_json, summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paired reconstructions under one protocol.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reconstructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=["alex", "squeeze", "vgg"], default="alex")
    parser.add_argument("--fid-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs.resolve()
    reconstructions = args.reconstructions.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    previous = read_json(summary_path) if summary_path.exists() else {}
    input_paths = [path for path in image_paths(inputs) if path.suffix.lower() == ".png"]
    recon_paths = [path for path in image_paths(reconstructions) if path.suffix.lower() == ".png"]
    input_map = {path.name: path for path in input_paths}
    recon_map = {path.name: path for path in recon_paths}
    if input_map.keys() != recon_map.keys():
        missing = sorted(input_map.keys() - recon_map.keys())[:10]
        extra = sorted(recon_map.keys() - input_map.keys())[:10]
        raise RuntimeError(f"Pairing mismatch. Missing={missing}; extra={extra}")
    names = sorted(input_map)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_metric = None
    lpips_error = None
    if not args.skip_lpips:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=args.lpips_net,
                reduction="none",
                normalize=True,
            ).eval().to(device)
        except Exception as error:
            lpips_error = f"{type(error).__name__}: {error}"
    records: list[dict] = []
    for start in tqdm(range(0, len(names), args.batch_size), desc=args.method):
        batch_names = names[start : start + args.batch_size]
        target = torch.stack([load_image(input_map[name]) for name in batch_names]).to(device)
        prediction = torch.stack([load_image(recon_map[name]) for name in batch_names]).to(device)
        metrics = paired_metrics(prediction, target)
        lpips_values = lpips_metric(prediction, target).flatten() if lpips_metric else None
        for index, name in enumerate(batch_names):
            records.append(
                {
                    "image": name,
                    "mse": float(metrics["mse"][index].cpu()),
                    "psnr": float(metrics["psnr"][index].cpu()),
                    "ssim": float(metrics["ssim"][index].cpu()),
                    "lpips": float(lpips_values[index].cpu()) if lpips_values is not None else None,
                }
            )

    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mse", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(records)

    result = {
        "method": args.method,
        "count": len(records),
        "inputs": str(inputs),
        "reconstructions": str(reconstructions),
        "metrics": {
            metric: summary([row[metric] for row in records])
            for metric in ("mse", "psnr", "ssim")
        },
        "lpips": (
            summary([row["lpips"] for row in records])
            if lpips_metric is not None
            else None
        ),
        "lpips_net": args.lpips_net if lpips_metric is not None else None,
        "lpips_error": lpips_error,
        "clean_rfid": previous.get("clean_rfid") if args.skip_fid else None,
        "clean_rfid_error": previous.get("clean_rfid_error") if args.skip_fid else None,
    }
    metadata_path = reconstructions / "metadata.json"
    if metadata_path.exists():
        result["model"] = read_json(metadata_path)
    if not args.skip_fid:
        try:
            from cleanfid import fid

            result["clean_rfid"] = float(
                fid.compute_fid(
                    str(inputs),
                    str(reconstructions),
                    mode="clean",
                    num_workers=args.fid_workers,
                )
            )
        except Exception as error:
            result["clean_rfid_error"] = f"{type(error).__name__}: {error}"
    write_json(result, summary_path)

    metrics = result["metrics"]
    lines = [
        f"# {args.method}",
        "",
        f"Paired images: {len(records)}",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|---:|",
        f"| MSE ↓ | {metrics['mse']['mean']:.6f} | ±{metrics['mse']['ci95']:.6f} |",
        f"| PSNR ↑ | {metrics['psnr']['mean']:.3f} | ±{metrics['psnr']['ci95']:.3f} |",
        f"| SSIM ↑ | {metrics['ssim']['mean']:.4f} | ±{metrics['ssim']['ci95']:.4f} |",
        f"| LPIPS ↓ | {result['lpips']['mean']:.4f} | ±{result['lpips']['ci95']:.4f} |"
        if result["lpips"] is not None
        else "| LPIPS ↓ | unavailable | — |",
        f"| clean-rFID ↓ | {result['clean_rfid'] if result['clean_rfid'] is not None else 'unavailable'} | — |",
        "",
    ]
    if result["clean_rfid_error"]:
        lines.append(f"FID error: `{result['clean_rfid_error']}`")
    if result["lpips_error"]:
        lines.append(f"LPIPS error: `{result['lpips_error']}`")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
