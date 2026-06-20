import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
for parent in Path(__file__).resolve().parents:
    if (parent / "dataset.py").exists():
        REPO_ROOT = parent
        break
else:
    raise RuntimeError("Cannot locate repository root containing dataset.py")

for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(REPO_ROOT))

from dataset import Dataset
from baselines.visualization import save_prediction_visualization
from train import (
    DEFAULT_MODEL_CACHE,
    DEFAULT_UWBENCH_META,
    DEFAULT_UWBENCH_ROOT,
    MODEL_IDS,
    SegFormerBinary,
    build_transforms,
    compute_metrics,
    setup_seed,
    update_confusion_from_batch,
)


def test(args):
    setup_seed(args.seed)
    os.makedirs(args.model_cache, exist_ok=True)
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    image_size = int(args.image_size or config.get("image_size", 518))
    stretch_to_square = bool(args.stretch_to_square or config.get("stretch_to_square", False))
    variant = args.variant or config.get("variant", "b2")
    model_name = args.model_name or config.get("model_name", "")

    image_transform, mask_transform = build_transforms(image_size, stretch_to_square)
    test_data = Dataset(
        root=args.data_path,
        transform=image_transform,
        target_transform=mask_transform,
        dataset_name=args.dataset_name,
        mode=args.test_mode,
        meta_path=args.meta_path,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SegFormerBinary(
        variant=variant,
        model_name=model_name,
        cache_dir=args.model_cache,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tp = fp = fn = tn = 0
    per_image_rows = []
    with torch.no_grad():
        for items in tqdm(test_loader, desc="test"):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            logits = model(images)
            batch_tp, batch_fp, batch_fn, batch_tn = update_confusion_from_batch(logits, masks, args.threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn

            probs = torch.sigmoid(logits)
            preds = probs >= args.threshold
            gts = masks > 0.5
            for idx in range(images.shape[0]):
                prediction_mask_path = ""
                overlay_path = ""
                if args.save_visualizations:
                    prediction_mask_path, overlay_path = save_prediction_visualization(
                        preds[idx],
                        items["img_path"][idx],
                        os.path.join(args.save_path, "visualizations"),
                        len(per_image_rows) + 1,
                        args.overlay_alpha,
                        output_size=image_size,
                        image_tensor=images[idx],
                    )
                img_tp = torch.logical_and(preds[idx], gts[idx]).sum().item()
                img_fp = torch.logical_and(preds[idx], torch.logical_not(gts[idx])).sum().item()
                img_fn = torch.logical_and(torch.logical_not(preds[idx]), gts[idx]).sum().item()
                img_iou = img_tp / (img_tp + img_fp + img_fn + 1e-8)
                per_image_rows.append(
                    {
                        "img_path": items["img_path"][idx],
                        "cls_name": items["cls_name"][idx],
                        "anomaly": int(items["anomaly"][idx]),
                        "iou": float(img_iou),
                        "prediction_mask_path": prediction_mask_path,
                        "overlay_path": overlay_path,
                    }
                )

    metrics = compute_metrics(tp, fp, fn, tn)
    payload = {
        "method": "SegFormer",
        "variant": variant,
        "model_id": model.model_id,
        "checkpoint_path": args.checkpoint_path,
        "data_path": args.data_path,
        "meta_path": args.meta_path,
        "test_mode": args.test_mode,
        "image_size": image_size,
        "threshold": args.threshold,
        "num_samples": len(test_data),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "metrics": metrics,
    }
    with open(os.path.join(args.save_path, "metrics_summary.json"), "w", encoding="utf-8") as fp_out:
        json.dump(payload, fp_out, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_path, "per_image_results.csv"), "w", encoding="utf-8", newline="") as fp_out:
        writer = csv.DictWriter(
            fp_out,
            fieldnames=["img_path", "cls_name", "anomaly", "iou", "prediction_mask_path", "overlay_path"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {args.save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Test SegFormer baseline for road waterlogging segmentation")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_UWBENCH_ROOT), help="dataset root")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_UWBENCH_META), help="meta json path")
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--test_mode", type=str, default="test")
    parser.add_argument("--checkpoint_path", type=str, default="experiments/segformer_b2/checkpoints/epoch_50.pth")
    parser.add_argument("--save_path", type=str, default="experiments/segformer_b2/results")
    parser.add_argument("--image_size", type=int, default=0, help="0 means use checkpoint config")
    parser.add_argument("--variant", type=str, default="", choices=[""] + sorted(MODEL_IDS.keys()))
    parser.add_argument("--model_name", type=str, default="", help="optional HuggingFace model id/path override")
    parser.add_argument("--model_cache", type=str, default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_visualizations", action="store_false", dest="save_visualizations")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.set_defaults(save_visualizations=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
