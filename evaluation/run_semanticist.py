#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from common import image_paths, load_image, save_batch, write_json


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export official SEMANTICIST reconstructions.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=root / "third_party" / "semanticist")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "weights" / "semanticist" / "semanticist_tok_L.pkl",
    )
    parser.add_argument("--vae", type=Path, default=root / "weights" / "mar-vae-kl16")
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
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
    if not 1 <= args.slots <= 256:
        raise ValueError("--slots must be between 1 and 256.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(args.source.resolve()))
    from semanticist.stage1.diffuse_slot import DiffuseSlot

    cfg = OmegaConf.load(args.source / "configs" / "tokenizer_l.yaml")
    model_cfg = OmegaConf.to_container(cfg.trainer.params.model.params, resolve=True)
    # REPA is only an auxiliary training loss; disabling it avoids loading DINOv2 at evaluation.
    model_cfg["use_repa"] = False
    model_cfg["vae"] = str(args.vae.resolve())
    model_cfg["num_sampling_steps"] = str(args.steps)
    model = DiffuseSlot(**model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = {key.replace("._orig_mod", ""): value for key, value in checkpoint.items()}
    incompatible = model.load_state_dict(checkpoint, strict=False)
    training_only_prefixes = ("repa_encoder.", "dit.projector.")
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(training_only_prefixes)
    ]
    expected_missing = [
        key
        for key in incompatible.missing_keys
        if not (key.startswith("vae.") or key in {"dit.pos_embed", "nested_sampler.arange"})
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected non-REPA checkpoint keys: {unexpected[:20]}")
    if expected_missing:
        raise RuntimeError(f"Missing learned checkpoint keys: {expected_missing[:20]}")
    model = model.eval().to(device)
    model.enable_nest = True

    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    if not paths:
        raise ValueError(f"No PNG inputs found in {args.inputs}.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "bf16"

    def encode(images: torch.Tensor) -> torch.Tensor:
        return model.encode_slots(images)

    def decode(slots: torch.Tensor) -> torch.Tensor:
        drop_mask = model.nested_sampler(
            slots.shape[0], slots.device, inference_with_n_slots=args.slots
        )
        return model.sample(slots, drop_mask=drop_mask, cfg=args.guidance_scale).clamp(0, 1)

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                decode(encode(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="SEMANTICIST"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                slots, encode_ms = event_time(lambda: encode(images))
                prediction, decode_ms = event_time(lambda: decode(slots))
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    write_json(
        {
            "method": f"SEMANTICIST-L-S{args.slots}-K{args.steps}",
            "source": "https://github.com/visual-gen/semanticist",
            "checkpoint": str(args.checkpoint.resolve()),
            "latent": f"{args.slots}x16_continuous_slots",
            "noise_seed": args.seed,
            "encoder_mode": "native",
            "encode_nfe": 1,
            "decode_nfe": args.steps,
            "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
            "decoder_parameters": sum(parameter.numel() for parameter in model.dit.parameters()),
            "shared_encoder_decoder_parameters": False,
            "guidance_scale": args.guidance_scale,
            "precision": args.precision,
            "mean_encoder_ms_per_image": statistics.mean(encode_times),
            "mean_decoder_ms_per_image": statistics.mean(decode_times),
            "mean_end_to_end_ms_per_image": (
                statistics.mean(encode_times) + statistics.mean(decode_times)
            ),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "samples": len(paths),
            "external_vae_keys": sum(key.startswith("vae.") for key in incompatible.missing_keys),
            "generated_position_embedding": "dit.pos_embed" in incompatible.missing_keys,
        },
        output / "metadata.json",
    )


if __name__ == "__main__":
    main()
