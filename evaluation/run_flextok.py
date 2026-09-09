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
    parser = argparse.ArgumentParser(description="Export official FlexTok reconstructions.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=root / "third_party" / "ml-flextok")
    parser.add_argument("--vendor", type=Path, default=root / "vendor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "weights" / "flextok_d12_d12_in1k",
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-batches", type=int, default=1)
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
    if not 1 <= args.tokens <= 256:
        raise ValueError("--tokens must be between 1 and 256.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sys.path[:0] = [str(args.vendor.resolve()), str(args.source.resolve())]
    from flextok.flextok_wrapper import FlexTokFromHub

    model = FlexTokFromHub.from_pretrained(str(args.checkpoint.resolve())).eval().to(device)
    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    if not paths:
        raise ValueError(f"No PNG inputs found in {args.inputs}.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "bf16"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def encode(images: torch.Tensor) -> list[torch.Tensor]:
        return [tokens[:, : args.tokens] for tokens in model.tokenize(images)]

    def decode(tokens: list[torch.Tensor]) -> torch.Tensor:
        return model.detokenize(
            tokens,
            vae_image_sizes=args.image_size // model.downsample_factor,
            timesteps=args.steps,
            generator=generator,
            verbose=False,
            guidance_scale=args.guidance_scale,
            perform_norm_guidance=True,
        ).clamp(0, 1)

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                decode(encode(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="FlexTok"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                tokens, encode_ms = event_time(lambda: encode(images))
                prediction, decode_ms = event_time(lambda: decode(tokens))
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    write_json(
        {
            "method": f"FlexTok-d12-d12-L{args.tokens}-K{args.steps}",
            "source": "https://github.com/apple/ml-flextok",
            "checkpoint": str(args.checkpoint.resolve()),
            "latent": f"{args.tokens}x6_discrete_fsq",
            "noise_seed": args.seed,
            "encoder_mode": "native",
            "encode_nfe": 1,
            "decode_nfe": args.steps,
            "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
            "decoder_parameters": sum(parameter.numel() for parameter in model.decoder.parameters()),
            "shared_encoder_decoder_parameters": False,
            "guidance_scale": args.guidance_scale,
            "norm_guidance": True,
            "precision": args.precision,
            "mean_encoder_ms_per_image": statistics.mean(encode_times),
            "mean_decoder_ms_per_image": statistics.mean(decode_times),
            "mean_end_to_end_ms_per_image": (
                statistics.mean(encode_times) + statistics.mean(decode_times)
            ),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "samples": len(paths),
        },
        output / "metadata.json",
    )


if __name__ == "__main__":
    main()
