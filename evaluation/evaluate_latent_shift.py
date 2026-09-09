#!/usr/bin/env python3
"""Paired latent-shift robustness for continuous F8C4 autoencoders."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm

from common import image_paths, load_image, paired_metrics, summary, write_json


DEFAULT_RAC_ROOT = Path("/research/cbim/vast/sf895/code/Rectified-Auto-Coder")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rac", "sdvae", "ssdd"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rac-root", type=Path, default=DEFAULT_RAC_ROOT)
    parser.add_argument("--rac-checkpoint", type=Path)
    parser.add_argument("--rac-config", type=Path)
    parser.add_argument("--rac-steps", type=int, default=4)
    parser.add_argument("--rac-encode-steps", type=int, default=4)
    parser.add_argument("--sdvae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ssdd-source", type=Path, default=root / "third_party" / "SSDD")
    parser.add_argument(
        "--ssdd-checkpoint",
        type=Path,
        default=root / "weights" / "F8C4" / "F8C4_S_256.safetensors",
    )
    parser.add_argument("--ssdd-steps", type=int, default=8)
    return parser.parse_args()


def load_rac(args, device):
    root = args.rac_root.resolve()
    sys.path[:0] = [str(root / "paper2"), str(root)]
    import eval_latent_perturbation as shared
    import eval_native_latent_perturbation as native
    import train_taesd_rac as rac

    checkpoint = (args.rac_checkpoint or shared.DEFAULT_CHECKPOINT).resolve()
    config_path = (args.rac_config or shared.DEFAULT_CONFIG).resolve()
    config = shared.parse_simple_yaml(config_path)
    model = shared.build_model(config, checkpoint, device)

    def encode(images01):
        images_m11 = images01.mul(2).sub(1)
        return native.encode_native_rac_latent(
            model,
            images_m11,
            teacher_latent_size=args.image_size // 8,
            config=config,
            encode_steps=args.rac_encode_steps,
            image_size=args.image_size,
            device=device,
        )

    def decode(latent, _noise):
        return shared.decode_rac(model, latent, args.rac_steps, args.image_size, device)

    return encode, decode, {
        "method": f"RAC-Tiny-native-K{args.rac_steps}",
        "checkpoint": str(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decode_nfe": args.rac_steps,
    }


def load_sdvae(args, device):
    from diffusers import AutoencoderKL

    model = AutoencoderKL.from_pretrained(args.sdvae_model).eval().to(device)

    def encode(images01):
        return model.encode(images01.mul(2).sub(1)).latent_dist.mode()

    def decode(latent, _noise):
        return model.decode(latent).sample.mul(0.5).add(0.5).clamp(0, 1)

    return encode, decode, {
        "method": "SD-VAE-ft-MSE",
        "checkpoint": args.sdvae_model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decode_nfe": 1,
    }


def load_ssdd(args, device):
    sys.path.insert(0, str(args.ssdd_source.resolve()))
    from ssdd import SSDD

    model = SSDD(
        encoder="f8c4",
        decoder="S",
        fm_sampler={"steps": args.ssdd_steps, "t_pow_shift": 2.0},
        checkpoint=str(args.ssdd_checkpoint.resolve()),
    ).eval().to(device)

    def encode(images01):
        return model.encode(images01.mul(2).sub(1)).mode()

    def decode(latent, noise):
        prediction = model.decode(latent, steps=args.ssdd_steps, noise=noise)
        return prediction.mul(0.5).add(0.5).clamp(0, 1)

    return encode, decode, {
        "method": f"SSDD-S-F8C4-K{args.ssdd_steps}",
        "checkpoint": str(args.ssdd_checkpoint.resolve()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decode_nfe": args.ssdd_steps,
    }


def build_method(
    args, device
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable, dict]:
    return {"rac": load_rac, "sdvae": load_sdvae, "ssdd": load_ssdd}[args.method](
        args, device
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if 0.0 not in args.sigmas:
        raise ValueError("--sigmas must include 0.0 to define the clean reference.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encode, decode, metadata = build_method(args, device)
    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    if not paths:
        raise ValueError(f"No PNG inputs found in {args.inputs}.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "fp16"
    latent_generator = torch.Generator(device=device).manual_seed(args.seed)
    decoder_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    records: list[dict] = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc=metadata["method"]):
            batch_paths = paths[start : start + args.batch_size]
            target = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent = encode(target)
                if latent.shape[1:] != (4, args.image_size // 8, args.image_size // 8):
                    raise RuntimeError(f"Expected F8C4 latent, received {tuple(latent.shape)}.")
                scale = latent.float().flatten(1).std(dim=1, unbiased=False)
                scale = scale.clamp_min(1e-6).view(-1, 1, 1, 1).to(latent.dtype)
                direction = torch.randn(
                    latent.shape,
                    generator=latent_generator,
                    device=device,
                    dtype=latent.dtype,
                )
                decoder_noise = None
                if args.method == "ssdd":
                    decoder_noise = torch.randn(
                        target.shape,
                        generator=decoder_generator,
                        device=device,
                        dtype=target.dtype,
                    )
                clean = decode(latent, decoder_noise).float().clamp(0, 1)

                for sigma in args.sigmas:
                    prediction = (
                        clean
                        if sigma == 0.0
                        else decode(latent + float(sigma) * scale * direction, decoder_noise)
                    )
                    quality = paired_metrics(prediction, target)
                    drift = (prediction.float() - clean).square().mean(dim=(1, 2, 3))
                    for index, path in enumerate(batch_paths):
                        records.append(
                            {
                                "image": path.name,
                                "sigma": float(sigma),
                                "mse": float(quality["mse"][index].cpu()),
                                "psnr": float(quality["psnr"][index].cpu()),
                                "ssim": float(quality["ssim"][index].cpu()),
                                "output_drift_mse": float(drift[index].cpu()),
                            }
                        )

    clean_by_image = {
        row["image"]: row for row in records if row["sigma"] == 0.0
    }
    for row in records:
        clean = clean_by_image[row["image"]]
        row["delta_psnr_vs_clean"] = row["psnr"] - clean["psnr"]
        row["delta_ssim_vs_clean"] = row["ssim"] - clean["ssim"]

    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    rows = []
    metrics = [
        "mse",
        "psnr",
        "ssim",
        "output_drift_mse",
        "delta_psnr_vs_clean",
        "delta_ssim_vs_clean",
    ]
    for sigma in args.sigmas:
        group = [row for row in records if row["sigma"] == sigma]
        rows.append(
            {
                "sigma": sigma,
                "count": len(group),
                "metrics": {
                    metric: summary([row[metric] for row in group]) for metric in metrics
                },
            }
        )
    payload = {
        **metadata,
        "inputs": str(args.inputs.resolve()),
        "samples": len(paths),
        "latent": f"4x{args.image_size // 8}x{args.image_size // 8}_continuous",
        "perturbation": "z_sigma = z + sigma * std_per_sample(z) * epsilon",
        "paired_direction_seed": args.seed,
        "decoder_noise_seed": args.seed + 1 if args.method == "ssdd" else None,
        "precision": args.precision,
        "rows": rows,
    }
    write_json(payload, output / "summary.json")

    lines = [
        f"# {metadata['method']} latent-shift robustness",
        "",
        "| sigma | PSNR ↑ | ΔPSNR ↑ | SSIM ↑ | ΔSSIM ↑ | output drift MSE ↓ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics_row = row["metrics"]
        lines.append(
            f"| {row['sigma']:.2f} | {metrics_row['psnr']['mean']:.3f} "
            f"± {metrics_row['psnr']['ci95']:.3f} | "
            f"{metrics_row['delta_psnr_vs_clean']['mean']:+.3f} | "
            f"{metrics_row['ssim']['mean']:.4f} | "
            f"{metrics_row['delta_ssim_vs_clean']['mean']:+.4f} | "
            f"{metrics_row['output_drift_mse']['mean']:.6f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
