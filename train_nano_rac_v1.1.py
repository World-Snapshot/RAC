import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from diffusers import AutoencoderTiny
from diffusers.models.autoencoders.vae import DecoderTiny

from utils.taesd_dataloader import build_dataloader
from utils.gpu_auto import auto_select_gpu


# Constant for padding value in state channels
STATE_PAD_VALUE = 0.5
# Number of RGB channels
RGB_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Train rectified auto-coder (TAESD-style backbone).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/LAION-Aesthetics",
        help="Path to training images or a HF dataset clone.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split name when using a HF dataset clone.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gallery-every", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Check and update best checkpoint every N steps.",
    )
    parser.add_argument(
        "--save-mid-every",
        type=int,
        default=1000,
        help="Save intermediate checkpoints every N steps.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=4,
        help="Number of Euler steps for RAC encode/decode gallery sampling.",
    )
    parser.add_argument(
        "--sample-steps-random",
        action="store_true",
        help="Use a random number of steps (1..sample-steps) during training.",
    )
    parser.add_argument(
        "--mean-velocity",
        action="store_true",
        dest="mean_velocity",
        help="Enable mean-velocity training (adds mean-velocity losses).",
    )
    parser.add_argument(
        "--no-mean-velocity",
        action="store_false",
        dest="mean_velocity",
        help="Disable mean-velocity training (use integrate_flow losses only).",
    )
    parser.add_argument(
        "--no-train-random-time-grid",
        action="store_false",
        dest="train_random_time_grid",
        help="Disable random time points during training (use uniform linspace).",
    )
    parser.set_defaults(mean_velocity=False)
    parser.set_defaults(train_random_time_grid=True)
    parser.add_argument(
        "--inline-gallery",
        action="store_true",
        help="Update gallery inline when running in a notebook.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional local path for velocity model weights.",
    )
    parser.add_argument(
        "--teacher-pretrained",
        type=str,
        default=None,
        help="Optional HF repo id or local path for TAESD teacher weights.",
    )
    parser.add_argument(
        "--state-channels",
        type=int,
        default=4,
        help="State channel count for velocity field (default 4).",
    )
    parser.add_argument(
        "--no-pos-enc",
        action="store_true",
        help="Disable relative sinusoidal position encoding.",
    )
    parser.add_argument(
        "--pos-enc-scale",
        type=float,
        default=0.01,
        help="Scale factor for relative cosine position encoding.",
    )
    parser.add_argument(
        "--state-downsample-factor",
        type=int,
        default=8,
        help="Downsample factor for control input (size // factor).",
    )
    parser.add_argument(
        "--no-rearrange-control",
        action="store_false",
        dest="rearrange_control",
        help="Use avg-pooling for control input instead of pixel unshuffle.",
    )
    parser.add_argument(
        "--enc-latent-weight",
        type=float,
        default=1.0,
        help="Weight for latent MSE loss (RAC latent vs teacher latent).",
    )
    parser.add_argument(
        "--enc-teacher-weight",
        type=float,
        default=None,
        help="Deprecated: use --enc-pixel-weight.",
    )
    parser.add_argument(
        "--enc-pixel-weight",
        type=float,
        default=1.0,
        help="Weight for teacher-decoded pixel MSE loss.",
    )
    parser.add_argument(
        "--roundtrip-weight",
        type=float,
        default=1.0,
        help="Weight for RAC encode->decode roundtrip MSE loss.",
    )
    parser.add_argument(
        "--block-const-weight",
        type=float,
        default=1.0,
        help="Weight for block-consistency loss on encoded state (pool->expand).",
    )
    parser.add_argument(
        "--rac-roundtrip-pool",
        action="store_true",
        dest="rac_roundtrip_pool",
        help="Use latent pooling before RAC roundtrip decode (default: enabled).",
    )
    parser.add_argument(
        "--no-rac-roundtrip-pool",
        action="store_false",
        dest="rac_roundtrip_pool",
        help=(
            "Disable latent pooling before RAC roundtrip decode. When disabled, "
            "reconstruction is performed directly in the full-resolution state, "
            "which makes the behavior closer to an RAE-style autoencoder with "
            "little to no compression; this can yield strong reconstructions but "
            "is effectively a shortcut."
        ),
    )
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=0.05,
        help="Relative std for Gaussian noise added at each step during training.",
    )
    parser.add_argument(
        "--input-noise-min",
        type=float,
        default=0.0,
        help="Minimum absolute std for step noise.",
    )
    parser.add_argument(
        "--no-input-noise",
        action="store_true",
        help="Disable step noise for forward/encode training.",
    )
    parser.add_argument(
        "--enc-noise",
        action="store_true",
        dest="enc_noise",
        help="Enable step noise during reverse/encode integration.",
    )
    parser.add_argument(
        "--no-enc-noise",
        action="store_false",
        dest="enc_noise",
        help="Disable step noise during reverse/encode integration (default).",
    )
    parser.add_argument(
        "--enc-loss-weight",
        type=float,
        default=None,
        help="Deprecated: sets both enc-latent-weight and enc-pixel-weight.",
    )
    parser.add_argument(
        "--no-enc-loss",
        action="store_false",
        dest="enc_loss",
        help="Disable latent encoding loss during training.",
    )
    parser.set_defaults(enc_loss=True)
    parser.set_defaults(rac_roundtrip_pool=True)
    parser.set_defaults(rearrange_control=True)
    parser.set_defaults(enc_noise=True)
    parser.add_argument(
        "--latent-downsample",
        type=str,
        default="learned",
        help="Latent downsample mode when not using center crop: avg, random, or learned.",
    )
    parser.add_argument(
        "--latent-downsample-channels",
        type=int,
        default=40,
        help="Base channel width for learned latent downsampler (approx 100k params).",
    )
    parser.add_argument(
        "--latent-center-crop",
        action="store_true",
        help="Use center crop (HxW -> latent) instead of pooling for RAC latent.",
    )
    parser.add_argument(
        "--no-latent-center-crop",
        action="store_false",
        dest="latent_center_crop",
        help="Disable center crop for RAC latent (use pooling or random sampling).",
    )
    parser.set_defaults(latent_center_crop=False)
    parser.add_argument(
        "--fully-bidirectional",
        action="store_true",
        dest="fully_bidirectional",
        help="Use centered latent<->state mapping in both forward and reverse paths (default: enabled).",
    )
    parser.add_argument(
        "--no-fully-bidirectional",
        action="store_false",
        dest="fully_bidirectional",
        help="Disable centered latent<->state mapping and use the original expand/pool path.",
    )
    parser.set_defaults(fully_bidirectional=True)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP mixed-precision training (default: disabled).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP mixed-precision training.",
    )
    return parser.parse_args()


def _format_yaml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    text = text.replace("'", "''")
    return f"'{text}'"


def save_config_yaml(log_dir, args, extra=None, filename="config.yaml"):
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args))
    if extra:
        cfg.update(extra)
    path = log_dir / filename
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return path
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(cfg.keys()):
            f.write(f"{key}: {_format_yaml_value(cfg[key])}\n")
    return path


def save_stats(
    log_dir,
    steps,
    losses,
    recon_losses,
    recon_pixel_losses=None,
    enc_latent_losses=None,
    enc_pixel_losses=None,
    state_align_losses=None,
    roundtrip_state_losses=None,
    roundtrip_state_pixel_losses=None,
    decode_path_losses=None,
    encode_path_losses=None,
    roundtrip_losses=None,
    block_const_losses=None,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    stats_path = log_dir / "stats.npz"
    payload = {
        "step": np.array(steps, dtype=np.int64),
        "loss": np.array(losses, dtype=np.float32),
        "recon": np.array(recon_losses, dtype=np.float32),
    }
    if recon_pixel_losses is not None:
        payload["recon_pixel"] = np.array(recon_pixel_losses, dtype=np.float32)
    if enc_latent_losses is not None:
        payload["enc_latent"] = np.array(enc_latent_losses, dtype=np.float32)
    if enc_pixel_losses is not None:
        payload["enc_pixel"] = np.array(enc_pixel_losses, dtype=np.float32)
    if state_align_losses is not None:
        payload["state_align"] = np.array(state_align_losses, dtype=np.float32)
    if roundtrip_state_losses is not None:
        payload["roundtrip_state"] = np.array(roundtrip_state_losses, dtype=np.float32)
    if roundtrip_state_pixel_losses is not None:
        payload["roundtrip_state_pixel"] = np.array(roundtrip_state_pixel_losses, dtype=np.float32)
    if decode_path_losses is not None:
        payload["decode_path"] = np.array(decode_path_losses, dtype=np.float32)
    if encode_path_losses is not None:
        payload["encode_path"] = np.array(encode_path_losses, dtype=np.float32)
    if roundtrip_losses is not None:
        payload["roundtrip"] = np.array(roundtrip_losses, dtype=np.float32)
    if block_const_losses is not None:
        payload["block_const"] = np.array(block_const_losses, dtype=np.float32)
    np.savez(stats_path, **payload)

    series = [
        ("loss", losses),
        ("recon_mse", recon_losses),
    ]
    if recon_pixel_losses is not None:
        series.append(("recon_pixel_mse", recon_pixel_losses))
    if enc_latent_losses is not None:
        series.append(("enc_latent_mse", enc_latent_losses))
    if enc_pixel_losses is not None:
        series.append(("enc_pixel_mse", enc_pixel_losses))
    if state_align_losses is not None:
        series.append(("state_align_mse", state_align_losses))
    if roundtrip_state_losses is not None:
        series.append(("roundtrip_state_mse", roundtrip_state_losses))
    if roundtrip_state_pixel_losses is not None:
        series.append(("roundtrip_state_pixel_mse", roundtrip_state_pixel_losses))
    if decode_path_losses is not None:
        series.append(("decode_path_mse", decode_path_losses))
    if encode_path_losses is not None:
        series.append(("encode_path_mse", encode_path_losses))
    if roundtrip_losses is not None:
        series.append(("roundtrip_mse", roundtrip_losses))
    if block_const_losses is not None:
        series.append(("block_const_mse", block_const_losses))

    series = [(name, vals) for name, vals in series if vals is not None and len(vals) > 0]
    include_overview = True
    total_plots = len(series) + (1 if include_overview else 0)
    cols = 2
    rows = max(1, math.ceil(total_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
    axes = axes.flatten()
    idx = 0
    if include_overview and series:
        ax = axes[idx]
        idx += 1
        for name, vals in series:
            ax.plot(steps[: len(vals)], vals, label=name)
        ax.set_title("all_losses")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    for name, vals in series:
        ax = axes[idx]
        idx += 1
        ax.plot(steps[: len(vals)], vals, label=name)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_ylim(top=1.0)
        ax.legend(fontsize=8)
    for j in range(idx, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(log_dir / "stats.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def _make_gallery_image(inputs, recon):
    # 2 rows x 4 cols: top inputs, bottom reconstructions
    n = min(4, inputs.shape[0], recon.shape[0])
    if n == 0:
        return None
    inputs = inputs[:n].detach().cpu().clamp(0, 1)
    recon = recon[:n].detach().cpu().clamp(0, 1)
    grid = make_grid(torch.cat([inputs, recon], dim=0), nrow=4, padding=2)
    img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    return img


@torch.no_grad()
def save_gallery(log_dir, step, inputs, recon, tag=None, title=None, footer=None):
    img = _make_gallery_image(inputs, recon)
    if img is None:
        return None
    vis_dir = log_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img)
    if footer:
        plt.figtext(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)
    if tag:
        out_path = vis_dir / f"gallery_step_{step}_{tag}.png"
    else:
        out_path = vis_dir / f"gallery_step_{step}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


@torch.no_grad()
def save_gallery_rows(log_dir, step, rows, row_titles, row_notes=None, tag=None, footer=None):
    if len(rows) == 0:
        return None
    n = min(4, *(row.shape[0] for row in rows))
    if n == 0:
        return None
    grids = []
    for row in rows:
        row = row[:n].detach().cpu().clamp(0, 1)
        grid = make_grid(row, nrow=4, padding=2)
        img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
        grids.append(img)

    vis_dir = log_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(grids), 1, figsize=(8, 4 * len(grids)))
    if len(grids) == 1:
        axes = [axes]
    if row_notes is not None and len(row_notes) != len(grids):
        raise ValueError("row_notes length must match rows length.")
    for idx, (ax, img, title) in enumerate(zip(axes, grids, row_titles)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
        if row_notes is not None and row_notes[idx]:
            ax.text(
                0.5,
                -0.06,
                row_notes[idx],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if tag:
        out_path = vis_dir / f"gallery_step_{step}_{tag}.png"
    else:
        out_path = vis_dir / f"gallery_step_{step}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def display_gallery_inline(inputs, recon, key="default"):
    try:
        import uuid
        from IPython.display import Image as IPImage
        from IPython.display import display, update_display
    except Exception:
        return False
    img = _make_gallery_image(inputs, recon)
    if img is None:
        return False
    try:
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        buf.seek(0)

        if not hasattr(display_gallery_inline, "_display_ids"):
            display_gallery_inline._display_ids = {}
        if key not in display_gallery_inline._display_ids:
            display_id = f"taesd_gallery_{key}_{uuid.uuid4()}"
            display_gallery_inline._display_ids[key] = display_id
            display(IPImage(data=buf.getvalue()), display_id=display_id)
        else:
            update_display(
                IPImage(data=buf.getvalue()),
                display_id=display_gallery_inline._display_ids[key],
            )
    except Exception:
        plt.figure(figsize=(8, 4))
        plt.axis("off")
        plt.imshow(img)
        plt.show()
    return True


def save_checkpoint(log_dir, step, model, opt=None, scaler=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
    }
    if opt is not None:
        ckpt["opt"] = opt.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    ckpt_path = log_dir / f"model_step_{step}.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def save_best_checkpoint(log_dir, step, model, opt=None, scaler=None, prev_best_path=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
    }
    if opt is not None:
        ckpt["opt"] = opt.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    ckpt_path = log_dir / f"best_step_{step}.pt"
    torch.save(ckpt, ckpt_path)
    if prev_best_path is not None and prev_best_path.exists():
        if prev_best_path.resolve() != ckpt_path.resolve():
            prev_best_path.unlink(missing_ok=True)
    return ckpt_path


def _extract_step(path: Path):
    stem = path.stem
    for part in stem.split("_"):
        if part.isdigit():
            return int(part)
    return None


def find_resume_checkpoint(log_dir: Path):
    if not log_dir.exists():
        return None
    candidates = []
    candidates.extend(log_dir.glob("best_step_*.pt"))
    candidates.extend(log_dir.glob("model_step_*.pt"))
    if not candidates:
        return None
    scored = []
    for path in candidates:
        step = _extract_step(path)
        if step is not None:
            scored.append((step, path))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[-1][1]




def build_latent_state(img, img01, teacher, teacher_kind, state_channels):
    if teacher is None:
        raise ValueError("TAESD teacher is required to build latent targets.")

    latents = encode_teacher_latent(img, img01, teacher, teacher_kind)

    if state_channels < latents.shape[1]:
        raise ValueError("state_channels must be >= teacher latent channels.")
    if state_channels > latents.shape[1]:
        pad = torch.full(
            (latents.shape[0],
             state_channels - latents.shape[1],
             latents.shape[2],
             latents.shape[3]),
            STATE_PAD_VALUE,
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = torch.cat([latents, pad], dim=1)

    return latents


def expand_latents(latents, full_size):
    if latents.shape[-1] == full_size:
        return latents
    return F.interpolate(latents, size=full_size, mode="nearest")


def latent_to_subpixel_seed_state(latents, full_size, pad_value=STATE_PAD_VALUE):
    """
    Convert latent to full-resolution state using pixel-shuffle with center alignment.
    This creates a symmetric mapping for bidirectional encoding/decoding.
    """
    b, c, h, w = latents.shape
    if full_size % h != 0 or full_size % w != 0:
        raise ValueError("full_size must be divisible by latent spatial size.")
    stride_h = full_size // h
    stride_w = full_size // w
    if stride_h != stride_w:
        raise ValueError("only square latent-to-state scaling is supported.")
    stride = stride_h
    if stride == 1:
        return latents

    # Create shuffle space filled with padding value
    shuffle_space = torch.full(
        (b, c * stride * stride, h, w),
        float(pad_value),
        device=latents.device,
        dtype=latents.dtype,
    )
    # Place latent values at center position of each block
    offset_y = stride // 2
    offset_x = stride // 2
    offset_idx = offset_y * stride + offset_x
    for ch in range(c):
        shuffle_space[:, ch * stride * stride + offset_idx] = latents[:, ch]
    return F.pixel_shuffle(shuffle_space, stride)


def subpixel_seed_state_to_latent(x_state, latent_size):
    """
    Extract latent from full-resolution state using pixel-unshuffle with center alignment.
    This is the inverse of latent_to_subpixel_seed_state.
    """
    _, c, h, w = x_state.shape
    if h % latent_size != 0 or w % latent_size != 0:
        raise ValueError("state size must be divisible by latent_size.")
    stride_h = h // latent_size
    stride_w = w // latent_size
    if stride_h != stride_w:
        raise ValueError("only square state-to-latent scaling is supported.")
    stride = stride_h
    if stride == 1:
        return x_state

    # Apply pixel unshuffle
    folded = F.pixel_unshuffle(x_state, stride)
    # Extract center position from each block
    offset_y = stride // 2
    offset_x = stride // 2
    offset_idx = offset_y * stride + offset_x
    out = []
    for ch in range(c):
        out.append(folded[:, ch * stride * stride + offset_idx : ch * stride * stride + offset_idx + 1])
    return torch.cat(out, dim=1)


def image_to_state(img, state_channels, pad_value=STATE_PAD_VALUE):
    """
    Convert image to state by padding with extra channels if needed.
    Only the first img.shape[1] channels are meaningful, rest are padding.
    """
    b, c, h, w = img.shape
    if state_channels < c:
        raise ValueError("state_channels must be >= image channels.")
    if state_channels == c:
        return img
    pad = torch.full(
        (b, state_channels - c, h, w),
        float(pad_value),
        device=img.device,
        dtype=img.dtype,
    )
    return torch.cat([img, pad], dim=1)


def sample_latent_from_state(x_state, latent_size, full_size, use_avg_pool=True):
    stride = full_size // latent_size
    if full_size % latent_size != 0:
        raise ValueError("full_size must be divisible by latent_size.")
    if stride == 1:
        return x_state
    if use_avg_pool:
        return F.avg_pool2d(x_state, kernel_size=stride, stride=stride)
    b, c, _, _ = x_state.shape
    out = torch.zeros(b, c, latent_size, latent_size, device=x_state.device, dtype=x_state.dtype)
    base_y = torch.arange(latent_size, device=x_state.device) * stride
    base_x = torch.arange(latent_size, device=x_state.device) * stride
    for j in range(b):
        dy = torch.randint(0, stride, (latent_size, latent_size), device=x_state.device)
        dx = torch.randint(0, stride, (latent_size, latent_size), device=x_state.device)
        y = (base_y[:, None] + dy).clamp_max(full_size - 1)
        x = (base_x[None, :] + dx).clamp_max(full_size - 1)
        out[j] = x_state[j, :, y, x]
    return out


def center_crop_latent_from_state(x_state, latent_size):
    _, _, h, w = x_state.shape
    if h < latent_size or w < latent_size:
        raise ValueError("state size must be >= latent_size.")
    top = (h - latent_size) // 2
    left = (w - latent_size) // 2
    return x_state[:, :, top : top + latent_size, left : left + latent_size]


class LatentDownsampler(torch.nn.Module):
    def __init__(self, in_channels=4, base_channels=40):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 2, in_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


def encode_teacher_latent(img_m11, img_01, teacher, teacher_kind):
    with torch.no_grad():
        if teacher_kind == "diffusers":
            latents = teacher.encode(img_m11).latents
        else:
            latents = teacher.encoder(img_01)
    return latents


def decode_teacher_latent(latents_m11, teacher, teacher_kind):
    with torch.no_grad():
        if teacher_kind == "diffusers":
            recon = teacher.decode(latents_m11).sample
        else:
            recon = teacher.decoder(latents_m11)
    return recon


def decode_teacher_latent_train(latents_m11, teacher, teacher_kind):
    if teacher_kind == "diffusers":
        recon = teacher.decode(latents_m11).sample
    else:
        recon = teacher.decoder(latents_m11)
    return recon




class TimeConditionedVelocityNet(torch.nn.Module):
    def __init__(
        self,
        state_channels,
        pretrained=None,
        use_pos_enc=True,
        pos_enc_scale=0.01,
        use_time=True,
        downsample_factor=8,
        use_rearrange_control=True,
    ):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.use_time = use_time
        self.pos_enc_scale = float(pos_enc_scale)
        self.downsample_factor = int(downsample_factor)
        self.use_rearrange_control = bool(use_rearrange_control)
        self.pos_channels = 2 if use_pos_enc else 0
        self.time_channels = 1 if use_time else 0
        self._pos_cache = None
        self._pos_cache_shape = None
        self._pos_cache_dtype = None
        self._pos_cache_device = None
        self._pos_cache_scale = None

        control_channels = state_channels
        if self.use_rearrange_control:
            control_channels = state_channels * (self.downsample_factor ** 2)
        in_channels = control_channels + self.time_channels + self.pos_channels
        self.decoder = DecoderTiny(
            in_channels=in_channels,
            out_channels=state_channels,
            num_blocks=(3, 3, 3, 1),
            block_out_channels=(64, 64, 64, 64),
            upsampling_scaling_factor=2,
            act_fn="relu",
            upsample_fn="nearest",
        )
        if pretrained:
            state = torch.load(pretrained, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.load_state_dict(state, strict=False)

    def _get_pos_enc(self, x):
        if not self.use_pos_enc:
            return None
        _, _, h, w = x.shape
        if (
            self._pos_cache is not None
            and self._pos_cache_shape == (h, w)
            and self._pos_cache_dtype == x.dtype
            and self._pos_cache_device == x.device
            and self._pos_cache_scale == self.pos_enc_scale
        ):
            return self._pos_cache

        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
        pos = torch.stack([torch.cos(pi * grid_x), torch.cos(pi * grid_y)], dim=0)
        pos = pos * self.pos_enc_scale
        pos = pos.unsqueeze(0)
        self._pos_cache = pos
        self._pos_cache_shape = (h, w)
        self._pos_cache_dtype = x.dtype
        self._pos_cache_device = x.device
        self._pos_cache_scale = self.pos_enc_scale
        return pos

    def _downsample_state(self, x_state, target_hw):
        stride = self.downsample_factor
        if stride <= 0:
            raise ValueError("downsample_factor must be >= 1.")
        if x_state.shape[-1] % stride != 0 or x_state.shape[-2] % stride != 0:
            raise ValueError("state size must be divisible by downsample_factor.")
        if stride == 1:
            return x_state
        if self.use_rearrange_control:
            return F.pixel_unshuffle(x_state, stride)
        return F.avg_pool2d(x_state, kernel_size=stride, stride=stride)

    def forward(self, x_state, t):
        target_hw = x_state.shape[-1] // self.downsample_factor
        if target_hw <= 0:
            raise ValueError("state size is too small for downsampling.")
        x_down = self._downsample_state(x_state, target_hw)
        inputs = [x_down]
        if self.use_time:
            t_img = t.view(-1, 1, 1, 1).expand(x_down.shape[0], 1, x_down.shape[2], x_down.shape[3])
            inputs.append(t_img)
        if self.use_pos_enc:
            pos = self._get_pos_enc(x_down).expand(x_down.shape[0], -1, -1, -1)
            inputs.append(pos)
        x_in = torch.cat(inputs, dim=1)
        return self.decoder(x_in)


class RACDecoder(torch.nn.Module):
    def __init__(
        self,
        state_channels,
        pretrained=None,
        use_pos_enc=True,
        pos_enc_scale=0.01,
        use_time=True,
        downsample_factor=8,
        use_rearrange_control=True,
    ):
        super().__init__()
        self.flow = TimeConditionedVelocityNet(
            state_channels=state_channels,
            pretrained=pretrained,
            use_pos_enc=use_pos_enc,
            pos_enc_scale=pos_enc_scale,
            use_time=use_time,
            downsample_factor=downsample_factor,
            use_rearrange_control=use_rearrange_control,
        )
    def velocity(self, x_state, t):
        return self.flow(x_state, t)

    def project(self, x):
        if x.shape[1] < 3:
            raise ValueError("state_channels must be >= 3 to project to RGB.")
        return x[:, :3]


def autocast_ctx(use_amp, device_type):
    if not use_amp:
        return nullcontext()
    try:
        return amp.autocast(device_type=device_type, enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp, device_type):
    try:
        return amp.GradScaler(enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def integrate_flow(
    model,
    x_init,
    steps,
    device,
    amp_device,
    use_amp,
    full_size,
    reverse=False,
    random_time_grid=False,
    step_noise_std=0.0,
    step_noise_min=0.0,
):
    if steps <= 0:
        raise ValueError("sample_steps must be > 0.")
    if random_time_grid and steps > 1:
        t_inner = torch.rand(steps - 1, device=device, dtype=torch.float32)
        t_vals = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.sort(t_inner)[0],
                torch.ones(1, device=device, dtype=torch.float32),
            ]
        )
    else:
        t_vals = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)

    x = x_init
    if reverse:
        for i in range(steps - 1, -1, -1):
            dt = t_vals[i + 1] - t_vals[i]
            t = t_vals[i + 1].expand(x.shape[0])
            if step_noise_std > 0 or step_noise_min > 0:
                x = add_input_noise(x, step_noise_std, step_noise_min)
            with autocast_ctx(use_amp, amp_device):
                v = model.velocity(x, t)
            x = x - v * dt
    else:
        for i in range(steps):
            dt = t_vals[i + 1] - t_vals[i]
            t = t_vals[i].expand(x.shape[0])
            if step_noise_std > 0 or step_noise_min > 0:
                x = add_input_noise(x, step_noise_std, step_noise_min)
            with autocast_ctx(use_amp, amp_device):
                v = model.velocity(x, t)
            x = x + v * dt
    return x


def integrate_flow_path(
    model,
    x_init,
    steps,
    device,
    amp_device,
    use_amp,
    full_size,
    reverse=False,
    random_time_grid=False,
    step_noise_std=0.0,
    step_noise_min=0.0,
):
    if steps <= 0:
        raise ValueError("sample_steps must be > 0.")
    if random_time_grid and steps > 1:
        t_inner = torch.rand(steps - 1, device=device, dtype=torch.float32)
        t_vals = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.sort(t_inner)[0],
                torch.ones(1, device=device, dtype=torch.float32),
            ]
        )
    else:
        t_vals = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)

    x = x_init
    path = [x_init]
    t_path = [t_vals[0]]
    if reverse:
        t_path = [t_vals[-1]]
        for i in range(steps - 1, -1, -1):
            dt = t_vals[i + 1] - t_vals[i]
            t = t_vals[i + 1].expand(x.shape[0])
            if step_noise_std > 0 or step_noise_min > 0:
                x = add_input_noise(x, step_noise_std, step_noise_min)
            with autocast_ctx(use_amp, amp_device):
                v = model.velocity(x, t)
            x = x - v * dt
            path.append(x)
            t_path.append(t_vals[i])
    else:
        for i in range(steps):
            dt = t_vals[i + 1] - t_vals[i]
            t = t_vals[i].expand(x.shape[0])
            if step_noise_std > 0 or step_noise_min > 0:
                x = add_input_noise(x, step_noise_std, step_noise_min)
            with autocast_ctx(use_amp, amp_device):
                v = model.velocity(x, t)
            x = x + v * dt
            path.append(x)
            t_path.append(t_vals[i + 1])
    return x, torch.stack(t_path), path


def _path_features(states):
    feats = []
    for x in states:
        if x.dim() == 4:
            x = x[0]
        feat = x.float().mean(dim=(1, 2))
        feats.append(feat)
    feats = torch.stack(feats, dim=0)
    return feats


def save_path_plot(log_dir, step, t_vals, path_states, tag=None, title=None, footer=None):
    feats = _path_features(path_states)
    feats = feats - feats.mean(dim=0, keepdim=True)
    if feats.shape[1] >= 2:
        try:
            _, _, v = torch.linalg.svd(feats, full_matrices=False)
        except RuntimeError:
            v = torch.svd(feats).V
        coords = feats @ v[:2].T
    else:
        coords = torch.cat([feats, torch.zeros_like(feats)], dim=1)
    coords = coords.detach().cpu().numpy()
    t_vals = t_vals.detach().cpu().numpy()

    log_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = log_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(coords[:, 0], coords[:, 1], "-o", linewidth=1.5, markersize=4)
    for i, t in enumerate(t_vals):
        plt.text(coords[i, 0], coords[i, 1], f"{t:.2f}", fontsize=8)
    plt.xlabel("PC1 (channel-mean)")
    plt.ylabel("PC2 (channel-mean)")
    plt.title(title or "Flow path (PCA of channel means)")
    if footer:
        plt.figtext(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if tag:
        out_path = paths_dir / f"path_step_{step}_{tag}.png"
        latest_path = log_dir / f"path_{tag}.png"
    else:
        out_path = paths_dir / f"path_step_{step}.png"
        latest_path = log_dir / "path.png"
    plt.savefig(out_path, dpi=150)
    plt.savefig(latest_path, dpi=150)
    plt.close()
    return out_path


def path_state_mse(x_start, x_end, t_path, path_states):
    if len(path_states) <= 2:
        return torch.zeros((), device=x_start.device, dtype=x_start.dtype)
    t_use = t_path[1:-1]
    preds = torch.stack(path_states[1:-1], dim=0)
    t_view = t_use.view(-1, 1, 1, 1, 1).to(preds.dtype)
    x_start = x_start.unsqueeze(0)
    x_end = x_end.unsqueeze(0)
    targets = x_start + (x_end - x_start) * t_view
    return F.mse_loss(preds, targets)


def add_input_noise(x, std_scale, std_min):
    if std_scale <= 0 and std_min <= 0:
        return x
    std = x.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)
    sigma = std_scale * std + std_min
    noise = torch.randn_like(x) * sigma
    return x + noise


def sample_time_points(batch_size, device, random_time_grid, eps=1e-4):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if random_time_grid:
        t = torch.rand(batch_size, device=device, dtype=torch.float32)
    else:
        if batch_size == 1:
            t = torch.tensor([0.5], device=device, dtype=torch.float32)
        else:
            t = torch.linspace(0, 1, batch_size, device=device, dtype=torch.float32)
    return t.clamp(min=eps, max=1.0)


def mean_velocity_decode(model, x_noise, t_value=1.0):
    t = torch.full(
        (x_noise.shape[0],),
        float(t_value),
        device=x_noise.device,
        dtype=x_noise.dtype,
    )
    u = model.velocity(x_noise, t)
    return x_noise - u


def mean_velocity_loss(model, x_start, x_end, t):
    v = x_end - x_start

    def u_fn(t_in):
        t_view = t_in.view(-1, 1, 1, 1).to(x_start.dtype)
        x_t = x_start + v * t_view
        t_model = t_in.to(x_start.dtype)
        return model.velocity(x_t, t_model)

    u_pred = u_fn(t)
    _, du_dt = torch.autograd.functional.jvp(
        u_fn,
        (t,),
        (torch.ones_like(t),),
        create_graph=False,
    )
    dt_view = t.view(-1, 1, 1, 1).to(u_pred.dtype)
    u_target = (v - dt_view * du_dt).detach()
    return F.mse_loss(u_pred, u_target)


def load_teacher(teacher_pretrained, device):
    if teacher_pretrained:
        teacher = AutoencoderTiny.from_pretrained(teacher_pretrained)
        teacher.to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher, "diffusers"

    pretrained_dir = Path(__file__).resolve().parent / "pretrained"
    encoder_path = pretrained_dir / "taesd_encoder.pth"
    decoder_path = pretrained_dir / "taesd_decoder.pth"
    if encoder_path.exists() and decoder_path.exists():
        taesd_dir = Path(__file__).resolve().parents[1] / "taesd"
        sys.path.insert(0, str(taesd_dir))
        import taesd as taesd_mod

        teacher = taesd_mod.TAESD(encoder_path=encoder_path, decoder_path=decoder_path)
        teacher.to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher, "local_taesd"

    taesd_dir = Path(__file__).resolve().parents[1] / "taesd"
    encoder_path = taesd_dir / "taesd_encoder.pth"
    decoder_path = taesd_dir / "taesd_decoder.pth"
    if encoder_path.exists() and decoder_path.exists():
        sys.path.insert(0, str(taesd_dir))
        import taesd as taesd_mod

        teacher = taesd_mod.TAESD(encoder_path=encoder_path, decoder_path=decoder_path)
        teacher.to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher, "local_taesd"

    raise ValueError(
        "Missing teacher weights. Provide --teacher-pretrained or place TAESD weights in ./pretrained."
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. No GPU detected.")
    gpu_index = auto_select_gpu()
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    use_amp = bool(args.amp) and (not args.no_amp)
    amp_device = "cuda"
    if args.enc_loss_weight is not None:
        args.enc_latent_weight = float(args.enc_loss_weight)
        args.enc_pixel_weight = float(args.enc_loss_weight)
    if args.enc_teacher_weight is not None:
        args.enc_pixel_weight = float(args.enc_teacher_weight)
    use_input_noise = not args.no_input_noise
    noise_std = float(args.input_noise_std)
    noise_min = float(args.input_noise_min)
    step_noise_std = noise_std if use_input_noise else 0.0
    step_noise_min = noise_min if use_input_noise else 0.0
    enc_step_noise_std = step_noise_std if args.enc_noise else 0.0
    enc_step_noise_min = step_noise_min if args.enc_noise else 0.0
    latent_downsample = str(args.latent_downsample).lower()
    if latent_downsample == "pool":
        latent_downsample = "avg"
    if latent_downsample not in ("avg", "random", "learned"):
        raise ValueError("--latent-downsample must be 'avg', 'random', or 'learned'.")
    if args.latent_center_crop:
        print("latent_center_crop enabled: latent downsample mode ignored.")
    latent_use_avg_pool = latent_downsample == "avg"
    if latent_downsample == "learned" and args.block_const_weight > 0:
        args.block_const_weight = 0.0
        print("latent_downsample=learned: disabling block-const loss.")
    if args.fully_bidirectional:
        print("fully_bidirectional enabled: using pixel-shuffle aligned latent<->state mapping.")

    # Print state channel configuration
    print(f"State configuration: {args.state_channels} total channels")
    print(f"  - Target channels (RGB): {RGB_CHANNELS}")
    if args.state_channels > RGB_CHANNELS:
        print(f"  - Padding channels: {args.state_channels - RGB_CHANNELS} (filled with {STATE_PAD_VALUE})")
        print(f"  - Reconstruction loss computed only on first {RGB_CHANNELS} channels")
    else:
        print("  - No padding channels (state_channels == RGB)")

    if args.sample_steps < 1:
        raise ValueError("--sample-steps must be >= 1.")
    use_time = args.sample_steps > 1 or args.mean_velocity
    model = RACDecoder(
        state_channels=args.state_channels,
        pretrained=args.pretrained,
        use_pos_enc=not args.no_pos_enc,
        pos_enc_scale=args.pos_enc_scale,
        use_time=use_time,
        downsample_factor=args.state_downsample_factor,
        use_rearrange_control=args.rearrange_control,
    ).to(device)
    if latent_downsample == "learned":
        model.latent_downsampler = LatentDownsampler(
            in_channels=args.state_channels,
            base_channels=args.latent_downsample_channels,
        ).to(device)

    teacher, teacher_kind = load_teacher(args.teacher_pretrained, device)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        candidate = Path(__file__).resolve().parent / dataset_root
        if candidate.exists():
            dataset_root = candidate
    dataloader = build_dataloader(
        root=str(dataset_root),
        batch_size=args.batch_size,
        size=args.size,
        random_crop=True,
        split=args.dataset_split,
        num_workers=4,  # Use 4 workers for faster data loading
    )
    data_iter = iter(dataloader)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = make_grad_scaler(use_amp, amp_device)

    steps = []
    losses = []
    recon_losses = []
    recon_pixel_losses = []
    enc_latent_losses = [] if args.enc_loss else None
    enc_pixel_losses = [] if args.enc_loss else None
    state_align_losses = [] if args.enc_loss else None
    roundtrip_state_losses = [] if args.enc_loss else None
    roundtrip_state_pixel_losses = [] if args.enc_loss else None
    decode_path_losses = []
    encode_path_losses = [] if args.enc_loss else None
    roundtrip_losses = [] if args.enc_loss else None
    block_const_losses = [] if args.enc_loss and args.block_const_weight > 0 else None

    log_dir = Path(__file__).resolve().parent / "log" / Path(__file__).stem
    start_time = time.time()
    resume_ckpt_path = find_resume_checkpoint(log_dir)
    start_step = 1
    best_ckpt_path = None
    best_loss = float("inf")
    if resume_ckpt_path is not None and resume_ckpt_path.exists():
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        if isinstance(ckpt, dict) and "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if isinstance(ckpt, dict) and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if isinstance(ckpt, dict) and "step" in ckpt:
            start_step = int(ckpt["step"]) + 1
        best_ckpt_path = resume_ckpt_path
    config_name = "config.yaml"
    if (log_dir / config_name).exists():
        config_name = f"config_step_{start_step}.yaml"
    save_config_yaml(
        log_dir,
        args,
        extra={"resume_ckpt": str(resume_ckpt_path) if resume_ckpt_path else None},
        filename=config_name,
    )

    stats_path = log_dir / "stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        legacy_roundtrip_state_only = "state_align" not in stats and "roundtrip_state" in stats
        steps = stats.get("step", np.array([], dtype=np.int64)).astype(np.int64).tolist()
        losses = stats.get("loss", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        recon_losses = stats.get("recon", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        if "recon_pixel" in stats:
            recon_pixel_losses = stats.get("recon_pixel", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        else:
            recon_pixel_losses = [float("nan")] * len(steps)
        if len(recon_pixel_losses) < len(steps):
            recon_pixel_losses.extend([float("nan")] * (len(steps) - len(recon_pixel_losses)))
        elif len(recon_pixel_losses) > len(steps):
            recon_pixel_losses = recon_pixel_losses[:len(steps)]
        if enc_latent_losses is not None and "enc_latent" in stats:
            enc_latent_losses = stats.get("enc_latent", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        if enc_pixel_losses is not None and "enc_pixel" in stats:
            enc_pixel_losses = stats.get("enc_pixel", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        if state_align_losses is not None:
            if "state_align" in stats:
                state_align_losses = stats.get("state_align", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            elif "roundtrip_state" in stats:
                state_align_losses = stats.get("roundtrip_state", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            else:
                state_align_losses = [float("nan")] * len(steps)
            if len(state_align_losses) < len(steps):
                state_align_losses.extend([float("nan")] * (len(steps) - len(state_align_losses)))
            elif len(state_align_losses) > len(steps):
                state_align_losses = state_align_losses[:len(steps)]
        if roundtrip_state_losses is not None:
            if "roundtrip_state" in stats and not legacy_roundtrip_state_only:
                roundtrip_state_losses = stats.get("roundtrip_state", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            else:
                roundtrip_state_losses = [float("nan")] * len(steps)
            if len(roundtrip_state_losses) < len(steps):
                roundtrip_state_losses.extend([float("nan")] * (len(steps) - len(roundtrip_state_losses)))
            elif len(roundtrip_state_losses) > len(steps):
                roundtrip_state_losses = roundtrip_state_losses[:len(steps)]
        if roundtrip_state_pixel_losses is not None:
            if "roundtrip_state_pixel" in stats:
                roundtrip_state_pixel_losses = stats.get("roundtrip_state_pixel", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            else:
                roundtrip_state_pixel_losses = [float("nan")] * len(steps)
            if len(roundtrip_state_pixel_losses) < len(steps):
                roundtrip_state_pixel_losses.extend([float("nan")] * (len(steps) - len(roundtrip_state_pixel_losses)))
            elif len(roundtrip_state_pixel_losses) > len(steps):
                roundtrip_state_pixel_losses = roundtrip_state_pixel_losses[:len(steps)]
        if "decode_path" in stats:
            decode_path_losses = stats.get("decode_path", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        if encode_path_losses is not None and "encode_path" in stats:
            encode_path_losses = stats.get("encode_path", np.array([], dtype=np.float32)).astype(np.float32).tolist()
        if roundtrip_losses is not None:
            if "roundtrip" in stats:
                roundtrip_losses = stats.get("roundtrip", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            else:
                roundtrip_losses = [float("nan")] * len(steps)
            if len(roundtrip_losses) < len(steps):
                roundtrip_losses.extend([float("nan")] * (len(steps) - len(roundtrip_losses)))
            elif len(roundtrip_losses) > len(steps):
                roundtrip_losses = roundtrip_losses[:len(steps)]
        if block_const_losses is not None:
            if "block_const" in stats:
                block_const_losses = stats.get("block_const", np.array([], dtype=np.float32)).astype(np.float32).tolist()
            else:
                block_const_losses = [float("nan")] * len(steps)
            if len(block_const_losses) < len(steps):
                block_const_losses.extend([float("nan")] * (len(steps) - len(block_const_losses)))
            elif len(block_const_losses) > len(steps):
                block_const_losses = block_const_losses[:len(steps)]
        if len(losses) > 0:
            best_loss = float(min(losses))
        if start_step > 1:
            # Truncate stats to the resume step for consistency.
            keep_idx = [i for i, s in enumerate(steps) if s < start_step]
            steps = [steps[i] for i in keep_idx]
            losses = [losses[i] for i in keep_idx]
            recon_losses = [recon_losses[i] for i in keep_idx]
            recon_pixel_losses = [recon_pixel_losses[i] for i in keep_idx]
            if enc_latent_losses is not None:
                enc_latent_losses = [enc_latent_losses[i] for i in keep_idx]
            if enc_pixel_losses is not None:
                enc_pixel_losses = [enc_pixel_losses[i] for i in keep_idx]
            if state_align_losses is not None:
                state_align_losses = [state_align_losses[i] for i in keep_idx]
            if roundtrip_state_losses is not None:
                roundtrip_state_losses = [roundtrip_state_losses[i] for i in keep_idx]
            if roundtrip_state_pixel_losses is not None:
                roundtrip_state_pixel_losses = [roundtrip_state_pixel_losses[i] for i in keep_idx]
            if decode_path_losses:
                decode_path_losses = [decode_path_losses[i] for i in keep_idx]
            if encode_path_losses is not None and encode_path_losses:
                encode_path_losses = [encode_path_losses[i] for i in keep_idx]
            if roundtrip_losses is not None and roundtrip_losses:
                roundtrip_losses = [roundtrip_losses[i] for i in keep_idx]
            if block_const_losses is not None and block_const_losses:
                block_const_losses = [block_const_losses[i] for i in keep_idx]

    pbar = tqdm(range(start_step, args.steps + 1), desc="train", ncols=100)
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device)
        img = batch.mul(2).sub(1)  # [0, 1] -> [-1, 1]
        img01 = batch

        steps_used = args.sample_steps
        if args.sample_steps_random and args.sample_steps > 1:
            steps_used = int(torch.randint(1, args.sample_steps + 1, (1,)).item())

        with autocast_ctx(use_amp, amp_device):
            latent_small = build_latent_state(
                img,
                img01,
                teacher,
                teacher_kind,
                state_channels=args.state_channels,
            )
            latent_full = latent_to_subpixel_seed_state(latent_small, args.size) if args.fully_bidirectional else expand_latents(latent_small, args.size)
            x_dec, t_dec_path, dec_path_states = integrate_flow_path(
                model,
                latent_full,
                steps_used,
                device,
                amp_device,
                use_amp,
                full_size=args.size,
                reverse=False,
                random_time_grid=args.train_random_time_grid,
                step_noise_std=step_noise_std,
                step_noise_min=step_noise_min,
            )
            x_target_state = image_to_state(img, args.state_channels)
            pred_img = model.project(x_dec)
            # Only compute MSE on non-padding channels (RGB channels)
            recon_mse = F.mse_loss(x_dec[:, :RGB_CHANNELS], x_target_state[:, :RGB_CHANNELS])
            recon_pixel_mse = F.mse_loss(pred_img.mul(0.5).add(0.5).clamp(0, 1), img01)
            decode_path_mse = path_state_mse(
                latent_full, x_target_state, t_dec_path, dec_path_states
            )
            x_img_state = None
            latent_mse = None
            pixel_mse = None
            state_align_mse = None
            roundtrip_state_mse = None
            roundtrip_mse = None
            roundtrip_latent = None
            block_const_mse = None
            if args.enc_loss:
                x_img_state = image_to_state(img, args.state_channels)
                x_encoded, t_enc_path, enc_path_states = integrate_flow_path(
                    model,
                    x_img_state,
                    steps_used,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=True,
                    random_time_grid=args.train_random_time_grid,
                    step_noise_std=enc_step_noise_std,
                    step_noise_min=enc_step_noise_min,
                )
                encode_path_mse = path_state_mse(
                    latent_full, x_img_state, t_enc_path, enc_path_states
                )
                if args.fully_bidirectional:
                    latent_rac = subpixel_seed_state_to_latent(x_encoded, latent_small.shape[-1])
                elif args.latent_center_crop:
                    latent_rac = center_crop_latent_from_state(x_encoded, latent_small.shape[-1])
                elif latent_downsample == "learned":
                    if not hasattr(model, "latent_downsampler"):
                        raise RuntimeError("latent_downsampler is missing for learned downsample mode.")
                    latent_rac = model.latent_downsampler(x_encoded)
                    if latent_rac.shape[-1] != latent_small.shape[-1]:
                        raise RuntimeError("latent_downsampler output size mismatch.")
                else:
                    latent_rac = sample_latent_from_state(
                        x_encoded,
                        latent_small.shape[-1],
                        args.size,
                        use_avg_pool=latent_use_avg_pool,
                    )
                roundtrip_latent = latent_rac
                state_align_mse = F.mse_loss(x_encoded, latent_full)
                latent_mse = F.mse_loss(latent_rac, latent_small)
                if args.block_const_weight > 0:
                    block_target = latent_to_subpixel_seed_state(latent_rac, args.size) if args.fully_bidirectional else expand_latents(latent_rac, args.size)
                    block_const_mse = F.mse_loss(x_encoded, block_target)
                teacher_from_rac = decode_teacher_latent_train(latent_rac, teacher, teacher_kind)
                pixel_mse = F.mse_loss(teacher_from_rac, img01)
                if args.rac_roundtrip_pool:
                    roundtrip_latent = latent_rac
                    x_roundtrip_init = latent_to_subpixel_seed_state(roundtrip_latent, args.size) if args.fully_bidirectional else expand_latents(roundtrip_latent, args.size)
                else:
                    x_roundtrip_init = x_encoded
                x_roundtrip = integrate_flow(
                    model,
                    x_roundtrip_init,
                    steps_used,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=args.train_random_time_grid,
                    step_noise_std=step_noise_std,
                    step_noise_min=step_noise_min,
                )
                x_roundtrip_state = integrate_flow(
                    model,
                    x_encoded,
                    steps_used,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=args.train_random_time_grid,
                    step_noise_std=step_noise_std,
                    step_noise_min=step_noise_min,
                )
                # Only compute MSE on non-padding channels (RGB channels)
                roundtrip_state_mse = F.mse_loss(x_roundtrip_state[:, :RGB_CHANNELS], x_target_state[:, :RGB_CHANNELS])
                roundtrip_mse = F.mse_loss(x_roundtrip[:, :RGB_CHANNELS], x_target_state[:, :RGB_CHANNELS])
                enc_mse = (
                    args.enc_latent_weight * latent_mse
                    + args.enc_pixel_weight * pixel_mse
                )
                loss = (
                    recon_mse
                    + enc_mse
                    + 0.5 * state_align_mse
                    + args.roundtrip_weight * roundtrip_state_mse
                    + args.roundtrip_weight * roundtrip_mse
                    + 0.1 * decode_path_mse
                    + 0.1 * encode_path_mse
                )
                if block_const_mse is not None:
                    loss = loss + args.block_const_weight * block_const_mse
            else:
                encode_path_mse = None
                loss = recon_mse + 0.1 * decode_path_mse

            mv_loss = None
            mv_loss_rev = None
            if args.mean_velocity:
                if x_img_state is None:
                    x_img_state = image_to_state(img, args.state_channels)
                t = sample_time_points(
                    x_img_state.shape[0],
                    device,
                    random_time_grid=args.train_random_time_grid,
                )
                mv_loss = mean_velocity_loss(model, latent_full, x_img_state, t)
                loss = loss + mv_loss
                if args.enc_loss:
                    t_rev = sample_time_points(
                        x_img_state.shape[0],
                        device,
                        random_time_grid=args.train_random_time_grid,
                    )
                    mv_loss_rev = mean_velocity_loss(model, x_img_state, latent_full, t_rev)
                    loss = loss + mv_loss_rev

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        steps.append(step)
        losses.append(float(loss.item()))
        recon_losses.append(float(recon_mse.item()))
        recon_pixel_losses.append(float(recon_pixel_mse.item()))
        decode_path_losses.append(float(decode_path_mse.item()))
        if enc_latent_losses is not None and latent_mse is not None:
            enc_latent_losses.append(float(latent_mse.item()))
        if enc_pixel_losses is not None and pixel_mse is not None:
            enc_pixel_losses.append(float(pixel_mse.item()))
        if state_align_losses is not None and state_align_mse is not None:
            state_align_losses.append(float(state_align_mse.item()))
        if roundtrip_state_losses is not None and roundtrip_state_mse is not None:
            roundtrip_state_losses.append(float(roundtrip_state_mse.item()))
        if roundtrip_state_pixel_losses is not None and x_img_state is not None:
            roundtrip_state_pixel_losses.append(
                float(F.mse_loss(model.project(x_roundtrip_state).mul(0.5).add(0.5).clamp(0, 1), img01).item())
            )
        if encode_path_losses is not None and encode_path_mse is not None:
            encode_path_losses.append(float(encode_path_mse.item()))
        if roundtrip_losses is not None and roundtrip_mse is not None:
            roundtrip_losses.append(float(roundtrip_mse.item()))
        if block_const_losses is not None and block_const_mse is not None:
            block_const_losses.append(float(block_const_mse.item()))

        mv_loss_value = float(mv_loss.item()) if mv_loss is not None else None
        mv_loss_rev_value = float(mv_loss_rev.item()) if mv_loss_rev is not None else None
        postfix = {
            "loss": f"{losses[-1]:.5f}",
            "recon": f"{recon_losses[-1]:.5f}",
            "recon_px": f"{recon_pixel_losses[-1]:.5f}",
            "dec_path": f"{decode_path_losses[-1]:.5f}",
        }
        if enc_latent_losses is not None:
            postfix["enc_latent"] = f"{enc_latent_losses[-1]:.5f}"
        if enc_pixel_losses is not None:
            postfix["enc_pixel"] = f"{enc_pixel_losses[-1]:.5f}"
        if state_align_losses is not None:
            postfix["st_align"] = f"{state_align_losses[-1]:.5f}"
        if roundtrip_state_losses is not None:
            postfix["rt_state"] = f"{roundtrip_state_losses[-1]:.5f}"
        if roundtrip_state_pixel_losses is not None:
            postfix["rt_state_px"] = f"{roundtrip_state_pixel_losses[-1]:.5f}"
        if encode_path_losses is not None:
            postfix["enc_path"] = f"{encode_path_losses[-1]:.5f}"
        if roundtrip_losses is not None:
            postfix["roundtrip"] = f"{roundtrip_losses[-1]:.5f}"
        if block_const_losses is not None:
            postfix["block_const"] = f"{block_const_losses[-1]:.5f}"
        if mv_loss_value is not None:
            postfix["mv"] = f"{mv_loss_value:.5f}"
        if mv_loss_rev_value is not None:
            postfix["mv_rev"] = f"{mv_loss_rev_value:.5f}"
        pbar.set_postfix(**postfix)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            save_stats(
                log_dir,
                steps,
                losses,
                recon_losses,
                recon_pixel_losses,
                enc_latent_losses,
                enc_pixel_losses,
                state_align_losses,
                roundtrip_state_losses,
                roundtrip_state_pixel_losses,
                decode_path_losses,
                encode_path_losses,
                roundtrip_losses,
                block_const_losses,
            )
            elapsed = time.time() - start_time
            with torch.no_grad():
                latent_small = build_latent_state(
                    img,
                    img01,
                    teacher,
                    teacher_kind,
                    state_channels=args.state_channels,
                )
                latent_full = latent_to_subpixel_seed_state(latent_small, args.size) if args.fully_bidirectional else expand_latents(latent_small, args.size)
                _, t_path, path_states = integrate_flow_path(
                    model,
                    latent_full[:1],
                    steps_used,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=args.train_random_time_grid,
                    step_noise_std=step_noise_std,
                    step_noise_min=step_noise_min,
                )
                x_target_state = image_to_state(img, args.state_channels)
                dec_path_mse_vis = path_state_mse(
                    latent_full[:1], x_target_state[:1], t_path, path_states
                )
                save_path_plot(
                    log_dir,
                    step,
                    t_path,
                    path_states,
                    tag="decode",
                    title="Decode path (latent -> image state)",
                    footer=f"path_mse: {dec_path_mse_vis.item():.6f}",
                )
                if args.enc_loss:
                    x_img_state = image_to_state(img, args.state_channels)
                    _, t_path_enc, path_states_enc = integrate_flow_path(
                        model,
                        x_img_state[:1],
                        steps_used,
                        device,
                        amp_device,
                        use_amp,
                        full_size=args.size,
                        reverse=True,
                        random_time_grid=args.train_random_time_grid,
                        step_noise_std=enc_step_noise_std,
                        step_noise_min=enc_step_noise_min,
                    )
                    enc_path_mse_vis = path_state_mse(
                        latent_full[:1], x_img_state[:1], t_path_enc, path_states_enc
                    )
                    save_path_plot(
                        log_dir,
                        step,
                        t_path_enc,
                        path_states_enc,
                        tag="encode",
                        title="Encode path (image state -> latent)",
                        footer=f"path_mse: {enc_path_mse_vis.item():.6f}",
                    )
            if enc_latent_losses is not None and enc_pixel_losses is not None:
                tqdm.write(
                    f"step {step}/{args.steps} | "
                    f"loss {losses[-1]:.6f} | "
                    f"recon {recon_losses[-1]:.6f} | "
                    f"dec_path {decode_path_losses[-1]:.6f} | "
                    f"enc_latent {enc_latent_losses[-1]:.6f} | "
                    f"enc_pixel {enc_pixel_losses[-1]:.6f} | "
                    f"state_align {state_align_losses[-1]:.6f} | "
                    f"roundtrip_state {roundtrip_state_losses[-1]:.6f} | "
                    f"enc_path {encode_path_losses[-1]:.6f} | "
                    f"{'roundtrip ' + f'{roundtrip_losses[-1]:.6f} | ' if roundtrip_losses is not None else ''}"
                    f"{'block_const ' + f'{block_const_losses[-1]:.6f} | ' if block_const_losses is not None else ''}"
                    f"{'mv ' + f'{mv_loss_value:.6f} | ' if mv_loss_value is not None else ''}"
                    f"{'mv_rev ' + f'{mv_loss_rev_value:.6f} | ' if mv_loss_rev_value is not None else ''}"
                    f"{elapsed/60:.1f} min"
                )
            else:
                tqdm.write(
                    f"step {step}/{args.steps} | "
                    f"loss {losses[-1]:.6f} | "
                    f"recon {recon_losses[-1]:.6f} | "
                    f"dec_path {decode_path_losses[-1]:.6f} | "
                    f"{'mv ' + f'{mv_loss_value:.6f} | ' if mv_loss_value is not None else ''}"
                    f"{'mv_rev ' + f'{mv_loss_rev_value:.6f} | ' if mv_loss_rev_value is not None else ''}"
                    f"{elapsed/60:.1f} min"
                )

        if step % args.gallery_every == 0 or step == args.steps:
            with torch.no_grad():
                latent_small = build_latent_state(
                    img,
                    img01,
                    teacher,
                    teacher_kind,
                    state_channels=args.state_channels,
                )
                inputs_vis = img01.clamp(0, 1)
                latent_full = latent_to_subpixel_seed_state(latent_small, args.size) if args.fully_bidirectional else expand_latents(latent_small, args.size)

                x_teacher_dec = integrate_flow(
                    model,
                    latent_full,
                    args.sample_steps,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=False,
                )
                pred_img = model.project(x_teacher_dec)
                pred_img_01 = pred_img.mul(0.5).add(0.5).clamp(0, 1)
                x_img_state = image_to_state(img, args.state_channels)
                x_encoded = integrate_flow(
                    model,
                    x_img_state,
                    args.sample_steps,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=True,
                    random_time_grid=False,
                    step_noise_std=enc_step_noise_std,
                    step_noise_min=enc_step_noise_min,
                )
                if args.fully_bidirectional:
                    latent_rac = subpixel_seed_state_to_latent(x_encoded, latent_small.shape[-1])
                elif args.latent_center_crop:
                    latent_rac = center_crop_latent_from_state(x_encoded, latent_small.shape[-1])
                elif latent_downsample == "learned":
                    if not hasattr(model, "latent_downsampler"):
                        raise RuntimeError("latent_downsampler is missing for learned downsample mode.")
                    latent_rac = model.latent_downsampler(x_encoded)
                    if latent_rac.shape[-1] != latent_small.shape[-1]:
                        raise RuntimeError("latent_downsampler output size mismatch.")
                else:
                    latent_rac = sample_latent_from_state(
                        x_encoded,
                        latent_small.shape[-1],
                        args.size,
                        use_avg_pool=latent_use_avg_pool,
                    )
                latent_mse = F.mse_loss(latent_rac, latent_small)
                teacher_from_rac = decode_teacher_latent(latent_rac, teacher, teacher_kind)
                pixel_mse = F.mse_loss(teacher_from_rac, img01)
                enc_mse = (
                    args.enc_latent_weight * latent_mse
                    + args.enc_pixel_weight * pixel_mse
                ).item()
                teacher_from_rac_vis = teacher_from_rac.clamp(0, 1)

                x_roundtrip = integrate_flow(
                    model,
                    (latent_to_subpixel_seed_state(latent_rac, args.size) if args.fully_bidirectional else expand_latents(latent_rac, args.size)) if args.rac_roundtrip_pool else x_encoded,
                    args.sample_steps,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=False,
                )
                x_roundtrip_state = integrate_flow(
                    model,
                    x_encoded,
                    args.sample_steps,
                    device,
                    amp_device,
                    use_amp,
                    full_size=args.size,
                    reverse=False,
                    random_time_grid=False,
                )
                recon_teacher_vis = pred_img_01
                recon_teacher_mse = F.mse_loss(recon_teacher_vis, inputs_vis).item()
                recon_rac_vis = model.project(x_roundtrip).mul(0.5).add(0.5).clamp(0, 1)
                recon_rac_state_vis = model.project(x_roundtrip_state).mul(0.5).add(0.5).clamp(0, 1)
                recon_rac_mse = F.mse_loss(recon_rac_vis, inputs_vis).item()
                recon_rac_state_mse = F.mse_loss(recon_rac_state_vis, inputs_vis).item()
                if args.inline_gallery:
                    display_gallery_inline(inputs_vis, recon_teacher_vis, key="teacher")
                    display_gallery_inline(inputs_vis, recon_rac_vis, key="rac")
                    display_gallery_inline(inputs_vis, recon_rac_state_vis, key="rac_state")

                out_path_teacher = save_gallery(
                    log_dir,
                    step,
                    inputs_vis,
                    recon_teacher_vis,
                    tag="teacher_decode",
                    title="Teacher latent -> RAC decode",
                    footer=f"mse: {recon_teacher_mse:.6f}",
                )
                if out_path_teacher is not None:
                    tqdm.write(f"saved gallery: {out_path_teacher}")

                out_path_rac = save_gallery_rows(
                    log_dir,
                    step,
                    [inputs_vis, recon_rac_vis, recon_rac_state_vis, teacher_from_rac_vis],
                    [
                        "Input image",
                        "RAC encode -> RAC decode",
                        "RAC encode state -> RAC decode state",
                        "RAC latent -> TAESD decode",
                    ],
                    row_notes=[
                        None,
                        f"mse: {recon_rac_mse:.6f}",
                        f"mse: {recon_rac_state_mse:.6f}",
                        None,
                    ],
                    tag="rac_roundtrip",
                    footer=(
                        f"enc_latent: {latent_mse.item():.6f} | "
                        f"enc_pixel: {pixel_mse.item():.6f}"
                    ),
                )
                if out_path_rac is not None:
                    tqdm.write(f"saved gallery: {out_path_rac}")

        if step % args.save_every == 0 or step == args.steps:
            if losses and losses[-1] < best_loss:
                best_loss = float(losses[-1])
                best_ckpt_path = save_best_checkpoint(
                    log_dir,
                    step,
                    model,
                    opt=opt,
                    scaler=scaler,
                    prev_best_path=best_ckpt_path,
                )
                tqdm.write(f"saved best checkpoint: {best_ckpt_path}")
        if step % args.save_mid_every == 0 or step == args.steps:
            ckpt_path = save_checkpoint(log_dir, step, model, opt=opt, scaler=scaler)
            tqdm.write(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
