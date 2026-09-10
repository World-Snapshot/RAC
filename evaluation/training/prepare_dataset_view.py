#!/usr/bin/env python3
"""Create one symlink-only class-folder view shared by all training implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_split(paths: list[Path], split: Path, source_root: Path, preserve_classes: bool) -> None:
    for index, source in enumerate(paths):
        if preserve_classes and source.parent != source_root:
            class_name = source.parent.name
        else:
            class_name = "evaluation"
        destination_dir = split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{index:06d}{source.suffix.lower()}"
        destination.symlink_to(source)


def main() -> None:
    args = parse_args()
    train_source = args.train_source.resolve()
    val_source = args.val_source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory.")

    train = image_paths(train_source)
    val = image_paths(val_source)
    if not train or not val:
        raise ValueError(f"Empty split: train={len(train)}, val={len(val)}.")

    train_hashes = {sha256(path) for path in train}
    val_hashes = {sha256(path) for path in val}
    overlap = train_hashes & val_hashes
    if overlap:
        raise RuntimeError(f"Detected {len(overlap)} train/validation duplicate images.")

    link_split(train, output / "train", train_source, preserve_classes=True)
    link_split(val, output / "val", val_source, preserve_classes=False)
    manifest = {
        "train_source": str(train_source),
        "val_source": str(val_source),
        "train_images": len(train),
        "val_images": len(val),
        "train_classes": len(
            [path for path in (output / "train").iterdir() if path.is_dir()]
        ),
        "val_classes": 1,
        "duplicate_hashes": 0,
        "storage": "absolute_symlinks",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
