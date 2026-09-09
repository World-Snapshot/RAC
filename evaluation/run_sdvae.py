#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

from common import image_paths, load_image, save_batch, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standard SD-VAE reconstructions.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
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
    model = AutoencoderKL.from_pretrained(args.model).eval().to(device)
    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "fp16"

    def encode(images):
        return model.encode(images).latent_dist.mode()

    def decode(latents):
        return model.decode(latents).sample

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    first = first.mul(2).sub(1)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                decode(encode(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="SD-VAE"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            images = images.mul(2).sub(1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent, encode_ms = event_time(lambda: encode(images))
                prediction, decode_ms = event_time(lambda: decode(latent))
            prediction = prediction.mul(0.5).add(0.5).clamp(0, 1)
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    write_json(
        {
            "method": "SD-VAE-ft-MSE",
            "source": args.model,
            "latent": "4x32x32_continuous",
            "encoder_mode": "native",
            "encode_nfe": 1,
            "decode_nfe": 1,
            "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
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
