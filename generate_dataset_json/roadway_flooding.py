"""Generate VisualAD meta JSON for Image Dataset for Roadway Flooding.

This dataset is used as a test-only road flooding segmentation set. The raw
labels are expected to be converted first from labels/ to visible mask/ files
with convert_masks_to_bw.py in the dataset directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT.parent / "datasets" / "Image Dataset for Roadway Flooding" / "Dataset"
DEFAULT_META_PATH = REPO_ROOT / "data_meta" / "roadway_flooding_meta.json"
IMAGE_PATTERN = re.compile(r"^image_(\d+)$", re.IGNORECASE)


def image_id(path: Path) -> int | None:
    match = IMAGE_PATTERN.match(path.stem)
    return int(match.group(1)) if match else None


def build_meta(root: Path, allow_missing_masks: bool = False) -> dict:
    images_dir = root / "images"
    masks_dir = root / "mask"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir() and not allow_missing_masks:
        raise FileNotFoundError(
            f"Mask directory not found: {masks_dir}. Run convert_masks_to_bw.py first."
        )

    image_paths = []
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            current_id = image_id(path)
            if current_id is not None:
                image_paths.append((current_id, path))

    if not image_paths:
        raise FileNotFoundError(f"No image_N.* files found in: {images_dir}")

    meta = {"test": {"road": []}}
    missing_masks = []
    for current_id, image_path in sorted(image_paths, key=lambda item: item[0]):
        mask_name = f"label_{current_id}.png"
        mask_path = masks_dir / mask_name
        if not mask_path.is_file():
            missing_masks.append(mask_path)
            if not allow_missing_masks:
                continue

        meta["test"]["road"].append(
            {
                "img_path": f"images/{image_path.name}",
                "mask_path": f"mask/{mask_name}",
                "cls_name": "road",
                "specie_name": "flooding",
                "anomaly": 1,
            }
        )

    if missing_masks and not allow_missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} converted mask files. First missing examples:\n{preview}"
        )

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Roadway-Flooding test meta JSON for VisualAD.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Roadway-Flooding Dataset root containing images/ and mask/.")
    parser.add_argument("--meta_path", default=str(DEFAULT_META_PATH), help="Output meta JSON path.")
    parser.add_argument(
        "--allow_missing_masks",
        action="store_true",
        help="Write meta even if mask/label_N.png files are missing. Intended only for debugging.",
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

    count = len(meta["test"]["road"])
    print(f"Wrote {count} Roadway-Flooding test samples to: {meta_path}")


if __name__ == "__main__":
    main()
