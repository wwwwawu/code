import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUDDLE1000_ROOT = REPO_ROOT.parent / 'Puddle-1000'
DEFAULT_META_ROOT = REPO_ROOT / 'data_meta' / 'puddle-1000'


@dataclass
class Puddle1000Subset:
    name: str
    category: str
    root: Path
    images_dir: Path
    masks_dir: Path
    slug: str


def _slugify(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower()).strip('_')
    return slug or 'subset'


def discover_puddle1000_subsets(dataset_root: Path) -> list[Puddle1000Subset]:
    dataset_root = Path(dataset_root)
    subsets: list[Puddle1000Subset] = []
    for images_dir in sorted(dataset_root.rglob('images')):
        if not images_dir.is_dir():
            continue
        subset_root = images_dir.parent
        masks_bw_dir = subset_root / 'masks_bw'
        masks_dir = masks_bw_dir if masks_bw_dir.is_dir() else subset_root / 'masks'
        if not masks_dir.is_dir():
            continue

        rel_subset = subset_root.relative_to(dataset_root)
        category = rel_subset.parts[0] if len(rel_subset.parts) > 1 else subset_root.name
        category = _slugify(category)
        slug = _slugify("__".join(rel_subset.parts))
        subsets.append(
            Puddle1000Subset(
                name=subset_root.name,
                category=category,
                root=subset_root,
                images_dir=images_dir,
                masks_dir=masks_dir,
                slug=slug,
            )
        )
    return subsets


def _mask_has_foreground(mask_path: Path) -> bool:
    mask_array = np.array(Image.open(mask_path).convert('L'))
    return bool((mask_array > 0).any())


def _build_mask_index(masks_dir: Path) -> dict[str, Path]:
    mask_index: dict[str, Path] = {}
    for mask_path in sorted(p for p in masks_dir.rglob('*') if p.is_file()):
        mask_index.setdefault(mask_path.stem, mask_path)
    return mask_index


def build_puddle1000_meta(subset: Puddle1000Subset, output_path: Path) -> dict:
    records = []
    image_paths = sorted([p for p in subset.images_dir.iterdir() if p.is_file()])
    mask_index = _build_mask_index(subset.masks_dir)
    for image_path in image_paths:
        mask_path = mask_index.get(image_path.stem)
        if mask_path is None:
            continue
        has_puddle = _mask_has_foreground(mask_path)
        records.append(
            {
                "img_path": image_path.relative_to(subset.root).as_posix(),
                "mask_path": mask_path.relative_to(subset.root).as_posix() if has_puddle else "",
                "cls_name": "puddle",
                "specie_name": subset.name,
                "anomaly": 1 if has_puddle else 0,
            }
        )

    meta = {"test": {"puddle": records}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return meta


def create_all_puddle1000_meta(dataset_root: Path = DEFAULT_PUDDLE1000_ROOT, meta_root: Path = DEFAULT_META_ROOT) -> list[dict]:
    dataset_root = Path(dataset_root)
    meta_root = Path(meta_root)
    manifests = []
    for subset in discover_puddle1000_subsets(dataset_root):
        meta_path = meta_root / f'{subset.slug}.json'
        meta = build_puddle1000_meta(subset, meta_path)
        manifests.append(
            {
                "slug": subset.slug,
                "name": subset.name,
                "category": subset.category,
                "category_dir": subset.category,
                "root": str(subset.root),
                "meta_path": str(meta_path),
                "images_dir": str(subset.images_dir),
                "masks_dir": str(subset.masks_dir),
                "num_samples": len(meta["test"]["puddle"]),
            }
        )
    manifest_path = meta_root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifests, f, indent=2)
    return manifests


if __name__ == '__main__':
    manifests = create_all_puddle1000_meta()
    print(f'wrote {len(manifests)} puddle-1000 meta files to {DEFAULT_META_ROOT}')
    for item in manifests:
        print(item["slug"], item["num_samples"])
