#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import image_paths, resize_center_crop, sha256, write_json
from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one immutable, flat evaluation image set.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = image_paths(args.source.resolve())
    if len(paths) < args.count:
        raise ValueError(f"Found {len(paths)} images, requested {args.count}.")
    random.Random(args.seed).shuffle(paths)
    selected = paths[: args.count]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.png"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{output} already contains PNG files; pass --overwrite.")

    records = []
    for index, source in enumerate(tqdm(selected, desc="prepare inputs")):
        name = f"{index:06d}.png"
        destination = output / name
        with Image.open(source) as image:
            image = resize_center_crop(image.convert("RGB"), args.size)
            image.save(destination)
        records.append(
            {
                "index": index,
                "name": name,
                "source": str(source),
                "source_sha256": sha256(source),
                "prepared_sha256": sha256(destination),
            }
        )
    write_json(
        {
            "source_root": str(args.source.resolve()),
            "count": args.count,
            "size": args.size,
            "seed": args.seed,
            "preprocessing": "resize_shorter_side_then_center_crop_bicubic",
            "images": records,
        },
        output / "manifest.json",
    )


if __name__ == "__main__":
    main()
