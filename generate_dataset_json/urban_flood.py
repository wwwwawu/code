"""Generate ProtoWD meta JSON for Urban Flood Image Dataset.

The dataset root is expected to contain two subsets:
  Deepflood/image + Deepflood/mask_bw
  Sazara/image + Sazara/mask_bw

Only a test split is generated because this dataset is used for evaluating an
already trained road flooding model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT.parent / "Urban Flood Image Dataset"
DEFAULT_META_PATH = REPO_ROOT / "data_meta" / "urban_flood_meta.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**12, path.name)


def add_deepflood_samples(root: Path, entries: list[dict], allow_missing_masks: bool) -> list[Path]:
    subset = "Deepflood"
    image_dir = root / subset / "image"
    mask_dir = root / subset / "mask_bw"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    if not mask_dir.is_dir() and not allow_missing_masks:
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    missing_masks = []
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=numeric_key,
    )
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            missing_masks.append(mask_path)
            if not allow_missing_masks:
                continue
        entries.append(
            {
                "img_path": f"{subset}/image/{image_path.name}",
                "mask_path": f"{subset}/mask_bw/{mask_path.name}",
                "cls_name": "road",
                "specie_name": "deepflood",
                "anomaly": 1,
            }
        )
    return missing_masks


def add_sazara_samples(root: Path, entries: list[dict], allow_missing_masks: bool) -> list[Path]:
    subset = "Sazara"
    image_dir = root / subset / "image"
    mask_dir = root / subset / "mask_bw"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    if not mask_dir.is_dir() and not allow_missing_masks:
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    missing_masks = []
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=numeric_key,
    )
    for image_path in image_paths:
        match = re.search(r"(\d+)", image_path.stem)
        if not match:
            continue
        mask_path = mask_dir / f"label_{match.group(1)}.png"
        if not mask_path.is_file():
            missing_masks.append(mask_path)
            if not allow_missing_masks:
                continue
        entries.append(
            {
                "img_path": f"{subset}/image/{image_path.name}",
                "mask_path": f"{subset}/mask_bw/{mask_path.name}",
                "cls_name": "road",
                "specie_name": "sazara",
                "anomaly": 1,
            }
        )
    return missing_masks


def build_meta(root: Path, allow_missing_masks: bool = False) -> dict:
    entries: list[dict] = []
    missing_masks = []
    missing_masks.extend(add_deepflood_samples(root, entries, allow_missing_masks))
    missing_masks.extend(add_sazara_samples(root, entries, allow_missing_masks))

    if missing_masks and not allow_missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} converted mask files. First missing examples:\n{preview}"
        )
    if not entries:
        raise FileNotFoundError(f"No Urban Flood test samples found under: {root}")

    return {"test": {"road": entries}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Urban Flood Image Dataset test meta JSON for ProtoWD.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Urban Flood Image Dataset root.")
    parser.add_argument("--meta_path", default=str(DEFAULT_META_PATH), help="Output meta JSON path.")
    parser.add_argument(
        "--allow_missing_masks",
        action="store_true",
        help="Write meta even if some mask_bw files are missing. Intended only for debugging.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser()
    if not meta_path.is_absolute():
        meta_path = (Path.cwd() / meta_path).resolve()

    meta = build_meta(root, allow_missing_masks=args.allow_missing_masks)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"Wrote {len(meta['test']['road'])} Urban Flood test samples to: {meta_path}")


if __name__ == "__main__":
    main()
