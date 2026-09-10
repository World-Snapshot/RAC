#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import image_paths, load_image, save_batch, write_json


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export SSDD official reconstructions.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=root / "third_party" / "SSDD")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "weights" / "F8C4" / "F8C4_S_256.safetensors",
    )
    parser.add_argument("--decoder", default="S")
    parser.add_argument("--encoder", choices=["f8c4", "sdvae"], default="f8c4")
    parser.add_argument("--local-accelerate-checkpoint", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-batches", type=int, default=2)
    return parser.parse_args()


def event_time(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(args.source.resolve()))
    from ssdd import SSDD

    checkpoint = args.checkpoint.resolve()
    model = SSDD(
        encoder=args.encoder,
        decoder=args.decoder,
        fm_sampler={"steps": args.steps, "t_pow_shift": 2.0},
        checkpoint=None if args.local_accelerate_checkpoint else str(checkpoint),
    )
    if args.local_accelerate_checkpoint:
        from safetensors.torch import load_file

        filename = "model_1.safetensors" if args.use_ema else "model.safetensors"
        weights_path = checkpoint / filename if checkpoint.is_dir() else checkpoint
        state = load_file(str(weights_path), device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_nonencoder = [key for key in incompatible.missing_keys if not key.startswith("encoder.")]
        if unexpected or missing_nonencoder:
            raise RuntimeError(f"Incompatible local SSDD checkpoint: unexpected={unexpected[:10]}, missing_nonencoder={missing_nonencoder[:10]}")
    model = model.eval().to(device)
    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "fp16"
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    first = first.mul(2).sub(1)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent = model.encode(first).mode()
                model.decode(latent, steps=args.steps, noise=torch.zeros_like(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="SSDD"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            images = images.mul(2).sub(1)
            noise = torch.randn(
                images.shape,
                generator=noise_generator,
                dtype=torch.float32,
            ).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent, encode_ms = event_time(lambda: model.encode(images).mode())
                prediction, decode_ms = event_time(
                    lambda: model.decode(latent, steps=args.steps, noise=noise)
                )
            prediction = prediction.mul(0.5).add(0.5).clamp(0, 1)
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    encoder_module = model.encoder.sdvae[0] if hasattr(model.encoder, "sdvae") else model.encoder
    encoder_parameters = sum(parameter.numel() for parameter in encoder_module.parameters())
    unique_parameters = sum(parameter.numel() for parameter in model.parameters()) + encoder_parameters
    write_json(
        {
            "method": f"SSDD-{args.decoder}-{args.encoder.upper()}-K{args.steps}",
            "source": "https://github.com/facebookresearch/SSDD",
            "checkpoint": str(checkpoint),
            "checkpoint_format": ("accelerate_ema" if args.use_ema else "accelerate_current") if args.local_accelerate_checkpoint else "official",
            "latent": "4x32x32_continuous",
            "noise_seed": args.seed,
            "encoder_mode": "frozen_sdvae" if args.encoder == "sdvae" else "native",
            "encode_nfe": 1,
            "decode_nfe": args.steps,
            "unique_parameters": unique_parameters,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": sum(parameter.numel() for parameter in model.decoder.parameters()),
            "shared_encoder_decoder_parameters": False,
            "precision": args.precision,
            "mean_encoder_ms_per_image": statistics.mean(encode_times),
            "mean_decoder_ms_per_image": statistics.mean(decode_times),
            "mean_end_to_end_ms_per_image": statistics.mean(encode_times) + statistics.mean(decode_times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "samples": len(paths),
        },
        output / "metadata.json",
    )


if __name__ == "__main__":
    main()
