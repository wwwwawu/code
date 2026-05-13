"""Small experiment logging helpers."""

from __future__ import annotations

import csv
import json
import os


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save_args_json(args, path: str, extra=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(vars(args))
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=_json_default)


def save_per_image_results_csv(path: str, img_paths, cls_names, labels, scores) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["index", "img_path", "cls_name", "label", "score"],
        )
        writer.writeheader()
        for idx, (img_path, cls_name, label, score) in enumerate(zip(img_paths, cls_names, labels, scores)):
            writer.writerow({
                "index": idx,
                "img_path": img_path,
                "cls_name": cls_name,
                "label": int(label),
                "score": float(score),
            })
