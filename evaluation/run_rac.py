#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import image_paths, load_image, save_batch, write_json


DEFAULT_FULL_ROOT = Path("/research/cbim/vast/sf895/code/Rectified-Auto-Coder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paired TAESD or RAC reconstructions.")
    parser.add_argument("--method", choices=["rac", "taesd"], default="rac")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rac-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--encode-steps", type=int, default=4)
    parser.add_argument("--encoder", choices=["native", "teacher"], default="native")
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
    for path in (root / "paper2", root):
        sys.path.insert(0, str(path))
    import eval_latent_perturbation as shared
    import eval_native_latent_perturbation as native
    import train_taesd_rac as rac

    checkpoint = (args.checkpoint or shared.DEFAULT_CHECKPOINT).resolve()
    config_path = (args.config or shared.DEFAULT_CONFIG).resolve()
    config = shared.parse_simple_yaml(config_path)
    model = shared.build_model(config, checkpoint, device)
    teacher, teacher_kind = rac.load_teacher(config.get("teacher_pretrained"), device)
    state_channels = int(config.get("state_channels", 4))
    paths = [path for path in image_paths(args.inputs.resolve()) if path.suffix.lower() == ".png"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    use_amp = args.precision == "fp16"

    def encode(images01: torch.Tensor) -> torch.Tensor:
        images_m11 = images01.mul(2).sub(1)
        if args.method == "taesd" or args.encoder == "teacher":
            return rac.build_latent_state(
                images_m11,
                images01,
                teacher,
                teacher_kind,
                state_channels=state_channels,
            )
        return native.encode_native_rac_latent(
            model,
            images_m11,
            teacher_latent_size=args.image_size // 8,
            config=config,
            encode_steps=args.encode_steps,
            image_size=args.image_size,
            device=device,
        )

    def decode(latent: torch.Tensor) -> torch.Tensor:
        if args.method == "taesd":
            return shared.decode_taesd(teacher, teacher_kind, latent)
        return shared.decode_rac(model, latent, args.steps, args.image_size, device)

    first = torch.stack([load_image(path, args.image_size) for path in paths[: args.batch_size]]).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                decode(encode(first))
        torch.cuda.synchronize()

    encode_times: list[float] = []
    decode_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc=args.method):
            batch_paths = paths[start : start + args.batch_size]
            images01 = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent, encode_ms = event_time(lambda: encode(images01))
                prediction, decode_ms = event_time(lambda: decode(latent))
            encode_times.append(encode_ms / len(batch_paths))
            decode_times.append(decode_ms / len(batch_paths))
            save_batch(prediction, [path.name for path in batch_paths], output)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    if args.method == "rac":
        unique_parameters = model_parameters if args.encoder == "native" else model_parameters + teacher_parameters
        name = f"RAC-Tiny-native-K{args.steps}" if args.encoder == "native" else f"RAC-Tiny-teacher-K{args.steps}"
        encoder_parameters = model_parameters if args.encoder == "native" else teacher_parameters
        decoder_parameters = model_parameters
        encode_nfe = args.encode_steps if args.encoder == "native" else 1
        decode_nfe = args.steps
    else:
        unique_parameters = teacher_parameters
        name = "TAESD"
        encoder_parameters = sum(parameter.numel() for parameter in teacher.encoder.parameters())
        decoder_parameters = sum(parameter.numel() for parameter in teacher.decoder.parameters())
        encode_nfe = 1
        decode_nfe = 1
    write_json(
        {
            "method": name,
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_payload.get("step"),
            "config": str(config_path),
            "teacher_kind": teacher_kind,
            "latent": "4x32x32_continuous",
            "encoder_mode": args.encoder if args.method == "rac" else "native",
            "encode_nfe": encode_nfe,
            "decode_nfe": decode_nfe,
            "unique_parameters": unique_parameters,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
            "shared_encoder_decoder_parameters": args.method == "rac" and args.encoder == "native",
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
