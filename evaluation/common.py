from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

LOCAL_VENDOR = Path(__file__).resolve().parent / "vendor"
if LOCAL_VENDOR.exists():
    sys.path.insert(0, str(LOCAL_VENDOR))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_paths(root: Path) -> list[Path]:
    paths = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    new_width = max(size, int(round(width * scale)))
    new_height = max(size, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((new_width, new_height), resampling)
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return image.crop((left, top, left + size, top + size))


def load_image(path: Path, size: int | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != (size, size):
            image = resize_center_crop(image, size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def save_batch(batch: torch.Tensor, names: Iterable[str], output_dir: Path) -> None:
    for tensor, name in zip(batch, names):
        save_image(tensor, output_dir / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(11, device=device, dtype=dtype) - 5
    kernel = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel[:, None] * kernel[None, :]
    return kernel_2d.expand(channels, 1, 11, 11).contiguous()


def paired_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    channels = prediction.shape[1]
    window = gaussian_window(channels, prediction.device, prediction.dtype)
    mu_x = F.conv2d(prediction, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x_sq = F.conv2d(prediction.square(), window, padding=5, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target.square(), window, padding=5, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=5, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim = (
        ((2 * mu_xy + c1) * (2 * sigma_xy + c2))
        / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    ).mean(dim=(1, 2, 3))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(count)) if count > 1 else 0.0,
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
