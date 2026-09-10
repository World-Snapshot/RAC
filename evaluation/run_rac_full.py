#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from common import image_paths, load_image, save_batch, write_json


DEFAULT_RAC_ROOT = Path("/research/cbim/vast/sf895/code/Rectified-Auto-Coder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-size RAC reconstructions.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rac-root", type=Path, default=DEFAULT_RAC_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
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

    root = args.rac_root.resolve()
    sys.path.insert(0, str(root))
    import train_REPA_rac_tuning as rac

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    teacher_source = config.get("teacher_pretrained") or "stabilityai/sd-vae-ft-mse"
    teacher, teacher_kind = rac.load_teacher(teacher_source, device)
    state_channels = int(config.get("state_channels", 4))
    sample_steps = int(args.steps)
    trained_sample_steps = int(config.get("sample_steps", sample_steps))
    model = rac.RACDecoder(
        state_channels=state_channels,
        decoder_config=rac.get_decoder_config(teacher),
        pretrained=None,
        use_pos_enc=bool(config.get("pos_enc", True)),
        pos_enc_scale=float(config.get("pos_enc_scale", 0.01)),
        use_time=trained_sample_steps > 1 or bool(config.get("mean_velocity", False)),
        use_extra_blocks=bool(config.get("post_block", False)),
        zero_init_conditioning=bool(config.get("zero_init_conditioning", False)),
        use_adapter=bool(config.get("adapter", False)),
        adapter_dim=int(config.get("adapter_dim", 88)),
    ).to(device).eval()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)

    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    if not paths:
        raise RuntimeError(f"No PNG inputs found in {args.inputs}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "fp16"

    def encode(images_m11: torch.Tensor) -> torch.Tensor:
        posterior = teacher.encode(images_m11).latent_dist
        latent = posterior.mode()
        if latent.shape[1] < state_channels:
            padding = torch.zeros(
                latent.shape[0],
                state_channels - latent.shape[1],
                *latent.shape[2:],
                device=latent.device,
                dtype=latent.dtype,
            )
            latent = torch.cat([latent, padding], dim=1)
        return rac.expand_latents(latent, args.image_size)

    def decode(state: torch.Tensor) -> torch.Tensor:
        decoded = rac.integrate_flow(
            model,
            state,
            sample_steps,
            device,
            "cuda",
            use_amp,
            full_size=args.image_size,
            reverse=False,
            random_time_grid=False,
        )
        return model.project(decoded).mul(0.5).add(0.5).clamp(0, 1)

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
        for start in tqdm(range(0, len(paths), args.batch_size), desc="RAC-1x"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            images = images.mul(2).sub(1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent, encode_ms = event_time(lambda: encode(images))
                prediction, decode_ms = event_time(lambda: decode(latent))
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    write_json(
        {
            "method": f"RAC-1x-teacher-K{sample_steps}",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
            "config": str(config_path),
            "teacher": teacher_source,
            "teacher_kind": teacher_kind,
            "latent": "4x32x32_continuous",
            "encoder_mode": "frozen_teacher",
            "encode_nfe": 1,
            "decode_nfe": sample_steps,
            "unique_parameters_with_teacher": model_parameters + teacher_parameters,
            "decoder_parameters": model_parameters,
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
