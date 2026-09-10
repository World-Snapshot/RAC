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
    parser = argparse.ArgumentParser(description="Export DiTo reconstructions from a local checkpoint.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=root / "third_party" / "dito")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-batches", type=int, default=0)
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import models
    from utils.geometry import make_coord_scale_grid

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = checkpoint["model"]
    model = models.make(model_spec, load_sd=True).eval().to(device)
    model.render_n_steps = args.steps

    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    if not paths:
        raise RuntimeError(f"No PNG inputs found in {args.inputs}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_bf16 = args.precision == "bf16"

    def coordinates(batch_size: int):
        coord, scale = make_coord_scale_grid(
            (args.image_size, args.image_size),
            device=device,
            batch_size=batch_size,
        )
        return coord.permute(0, 3, 1, 2), scale.permute(0, 3, 1, 2)

    def encode(images_m11: torch.Tensor):
        return model.decode(model.encode(images_m11))

    def render(z_dec: torch.Tensor):
        coord, scale = coordinates(z_dec.shape[0])
        return model.render(z_dec, coord, scale).mul(0.5).add(0.5).clamp(0, 1)

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    first = first.mul(2).sub(1)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                render(encode(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="DiTo"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            images = images.mul(2).sub(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                latent, encode_ms = event_time(lambda: encode(images))
                prediction, decode_ms = event_time(lambda: render(latent))
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    renderer_parameters = sum(parameter.numel() for parameter in model.renderer.parameters())
    write_json(
        {
            "method": f"DiTo-B-F8C4-K{args.steps}",
            "source": "https://github.com/facebookresearch/DiTo",
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint.get("iter"),
            "latent": "4x32x32_continuous",
            "noise_seed": args.seed,
            "encoder_mode": "native_jointly_trained",
            "encode_nfe": 1,
            "decode_nfe": args.steps,
            "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": renderer_parameters,
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
