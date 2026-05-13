import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_dataset_json.puddle1000 import create_all_puddle1000_meta
from generate_dataset_json.uwbench import UWBenchSolver


WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_UWBENCH_ROOT = WORKSPACE_ROOT / 'UW-Bench' / 'training_set'
DEFAULT_PUDDLE_ROOT = WORKSPACE_ROOT / 'Puddle-1000'
DEFAULT_META_ROOT = REPO_ROOT / 'data_meta' / 'uwbench7-puddle-train'


def _load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _rel_to_workspace(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def _build_uwbench_records(uwbench_meta: dict, split: str) -> list[dict]:
    records = []
    for item in uwbench_meta[split]['road']:
        record = dict(item)
        record['img_path'] = f"UW-Bench/training_set/{item['img_path']}"
        record['mask_path'] = f"UW-Bench/training_set/{item['mask_path']}" if item['mask_path'] else ""
        record['cls_name'] = 'water'
        record['specie_name'] = f"uwbench_{split}"
        records.append(record)
    return records


def _build_puddle_records(manifest_item: dict) -> list[dict]:
    meta = _load_json(Path(manifest_item['meta_path']))
    records = []
    subset_root = Path(manifest_item['root'])
    for item in meta['test']['puddle']:
        img_abs = subset_root / Path(item['img_path'])
        mask_abs = subset_root / Path(item['mask_path']) if item['mask_path'] else None
        records.append({
            "img_path": _rel_to_workspace(img_abs),
            "mask_path": _rel_to_workspace(mask_abs) if mask_abs else "",
            "cls_name": "water",
            "specie_name": manifest_item['slug'],
            "anomaly": int(item['anomaly']),
        })
    return records


def build_uwbench7_puddle_meta(
    uwbench_root: Path = DEFAULT_UWBENCH_ROOT,
    puddle_root: Path = DEFAULT_PUDDLE_ROOT,
    meta_root: Path = DEFAULT_META_ROOT,
):
    meta_root = Path(meta_root)
    meta_root.mkdir(parents=True, exist_ok=True)

    uwbench_meta_path = meta_root / 'uwbench_meta_local.json'
    UWBenchSolver(root=uwbench_root, meta_path=uwbench_meta_path).run()
    uwbench_meta = _load_json(uwbench_meta_path)

    puddle_meta_root = REPO_ROOT / 'data_meta' / 'puddle-1000'
    puddle_manifests = create_all_puddle1000_meta(puddle_root, puddle_meta_root)
    manifest_by_name = {item['name']: item for item in puddle_manifests}

    # Use non-duplicated Puddle training subsets only.
    puddle_train_names = [
        'Puddle-1000 Dataset_train',
        'Puddle-1000 Dataset_train_foggy3.2',
        'Puddle-1000 Dataset_train_night',
    ]
    puddle_eval_plan = [
        {
            "name": "uwbench3",
            "dataset_name": "uwbench",
            "root": str(uwbench_root),
            "meta_path": str(uwbench_meta_path),
            "test_mode": "test",
            "result_subdir": "uwbench3",
            "category": "uwbench3",
        },
        {
            "name": "base",
            "dataset_name": "puddle1000",
            "root": str(manifest_by_name['Puddle-1000 Dataset_val']['root']),
            "meta_path": str(manifest_by_name['Puddle-1000 Dataset_val']['meta_path']),
            "test_mode": "test",
            "result_subdir": "puddle-train/base",
            "category": "base",
        },
        {
            "name": "on",
            "dataset_name": "puddle1000",
            "root": str(manifest_by_name['Puddle-1000 Dataset_val_on']['root']),
            "meta_path": str(manifest_by_name['Puddle-1000 Dataset_val_on']['meta_path']),
            "test_mode": "test",
            "result_subdir": "puddle-train/on",
            "category": "on",
        },
        {
            "name": "off",
            "dataset_name": "puddle1000",
            "root": str(manifest_by_name['Puddle-1000 Dataset_val_off']['root']),
            "meta_path": str(manifest_by_name['Puddle-1000 Dataset_val_off']['meta_path']),
            "test_mode": "test",
            "result_subdir": "puddle-train/off",
            "category": "off",
        },
        {
            "name": "foggy3.2",
            "dataset_name": "puddle1000",
            "root": str(manifest_by_name['Puddle-1000 Dataset_val_foggy3.2']['root']),
            "meta_path": str(manifest_by_name['Puddle-1000 Dataset_val_foggy3.2']['meta_path']),
            "test_mode": "test",
            "result_subdir": "puddle-train/foggy3.2",
            "category": "foggy3.2",
        },
        {
            "name": "night",
            "dataset_name": "puddle1000",
            "root": str(manifest_by_name['Puddle-1000 Dataset_val_night']['root']),
            "meta_path": str(manifest_by_name['Puddle-1000 Dataset_val_night']['meta_path']),
            "test_mode": "test",
            "result_subdir": "puddle-train/night",
            "category": "night",
        },
    ]

    train_records = _build_uwbench_records(uwbench_meta, 'train')
    train_manifest = {
        "uwbench_train_samples": len(train_records),
        "puddle_train_subsets": [],
    }
    for subset_name in puddle_train_names:
        manifest_item = manifest_by_name[subset_name]
        subset_records = _build_puddle_records(manifest_item)
        train_records.extend(subset_records)
        train_manifest["puddle_train_subsets"].append({
            "name": subset_name,
            "slug": manifest_item['slug'],
            "num_samples": len(subset_records),
        })

    train_meta = {"train": {"water": train_records}}
    train_meta_path = meta_root / 'train_meta.json'
    _write_json(train_meta_path, train_meta)

    combined_val_records = _build_uwbench_records(uwbench_meta, 'test')
    combined_val_manifest = {
        "uwbench_val_samples": len(combined_val_records),
        "puddle_eval_subsets": [],
    }
    for item in puddle_eval_plan[1:]:
        manifest_item = _load_json(Path(item['meta_path']))
        subset_records = []
        subset_root = Path(item['root'])
        for record in manifest_item['test']['puddle']:
            img_abs = subset_root / Path(record['img_path'])
            mask_abs = subset_root / Path(record['mask_path']) if record['mask_path'] else None
            subset_records.append({
                "img_path": _rel_to_workspace(img_abs),
                "mask_path": _rel_to_workspace(mask_abs) if mask_abs else "",
                "cls_name": "water",
                "specie_name": item['name'],
                "anomaly": int(record['anomaly']),
            })
        combined_val_records.extend(subset_records)
        combined_val_manifest["puddle_eval_subsets"].append({
            "name": item['name'],
            "num_samples": len(subset_records),
        })

    val_meta = {"test": {"water": combined_val_records}}
    val_meta_path = meta_root / 'val_meta.json'
    _write_json(val_meta_path, val_meta)

    experiment_manifest = {
        "workspace_root": str(WORKSPACE_ROOT),
        "train_root": str(WORKSPACE_ROOT),
        "train_meta_path": str(train_meta_path),
        "val_meta_path": str(val_meta_path),
        "train_dataset_name": "mixed_water",
        "uwbench_root": str(uwbench_root),
        "puddle_root": str(puddle_root),
        "train_manifest": train_manifest,
        "combined_val_manifest": combined_val_manifest,
        "evaluation_plan": puddle_eval_plan,
    }
    manifest_path = meta_root / 'manifest.json'
    _write_json(manifest_path, experiment_manifest)
    return experiment_manifest


if __name__ == '__main__':
    manifest = build_uwbench7_puddle_meta()
    print(f"wrote mixed train meta to {manifest['train_meta_path']}")
    print(f"wrote mixed val meta to {manifest['val_meta_path']}")
    print(f"wrote manifest to {DEFAULT_META_ROOT / 'manifest.json'}")
