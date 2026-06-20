import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.visualization import save_prediction_visualization

DEFAULT_IMAGE_ROOT = Path("..") / "datasets" / "城市内涝识别数据"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_train_module(method):
    module_path = REPO_ROOT / "baselines" / method / "train.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find train.py for method '{method}': {module_path}")
    module_name = f"_visualad_{method}_train"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def list_images(root):
    root = Path(root)
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda item: str(item.relative_to(root)).lower())


def safe_name_from_relative(relative_path):
    stem = str(Path(relative_path).with_suffix(""))
    return stem.replace("\\", "__").replace("/", "__").replace(" ", "_")


class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = list_images(self.root)
        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        relative_path = image_path.relative_to(self.root)
        group_name = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        return {
            "img": tensor,
            "img_path": str(image_path),
            "relative_path": str(relative_path).replace("\\", "/"),
            "group_name": group_name,
        }


def build_model(args, checkpoint, device):
    config = checkpoint.get("config", {})
    method = args.method
    module = load_train_module(method)
    image_size = int(args.image_size or config.get("image_size", 518))
    stretch_to_square = bool(args.stretch_to_square or config.get("stretch_to_square", False))
    image_transform, _ = module.build_transforms(image_size, stretch_to_square)

    if method == "unet":
        base_channels = int(args.base_channels or config.get("base_channels", 64))
        model = module.UNet(base_channels=base_channels)
        model_config = {"base_channels": base_channels}
    elif method == "deeplabv3plus":
        model_cache = args.model_cache or str(module.DEFAULT_MODEL_CACHE)
        if model_cache:
            os.environ["TORCH_HOME"] = str(resolve_repo_path(model_cache))
            os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
        backbone = args.backbone or config.get("backbone", "resnet101")
        model = module.DeepLabV3PlusBinary(backbone=backbone, pretrained_backbone=False)
        model_config = {"backbone": backbone}
    elif method == "segformer":
        model_cache = str(resolve_repo_path(args.model_cache)) if args.model_cache else str(module.DEFAULT_MODEL_CACHE)
        os.makedirs(model_cache, exist_ok=True)
        variant = args.variant or config.get("variant", "b5")
        model_name = args.model_name or config.get("model_name", "")
        model = module.SegFormerBinary(variant=variant, model_name=model_name, cache_dir=model_cache)
        model_config = {"variant": variant, "model_id": model.model_id}
    elif method == "u2net":
        model_type = args.model_type or config.get("model_type", "u2net")
        model = module.build_model(model_type)
        model_config = {"model_type": model_type}
    elif method == "setr":
        model_cache = str(resolve_repo_path(args.model_cache)) if args.model_cache else str(module.DEFAULT_MODEL_CACHE)
        os.makedirs(model_cache, exist_ok=True)
        backbone = args.backbone or config.get("backbone", "vit_b16")
        model_name = args.model_name or config.get("model_name", "")
        decoder_dim = int(args.decoder_dim or config.get("decoder_dim", 256))
        model = module.SETRPUPBinary(
            backbone=backbone,
            model_name=model_name,
            cache_dir=model_cache,
            decoder_dim=decoder_dim,
        )
        model_config = {"backbone": backbone, "model_id": model.model_id, "decoder_dim": decoder_dim}
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, image_transform, image_size, stretch_to_square, model_config


def reduce_image_scores(probs, topk_ratio):
    flat = probs.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * topk_ratio)))
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


def predict(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    save_path = resolve_repo_path(args.save_path)
    visualization_dir = save_path / "visualizations"
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, image_transform, image_size, stretch_to_square, model_config = build_model(args, checkpoint, device)

    data_root = Path(args.image_dir)
    dataset = UnlabeledImageDataset(data_root, image_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    group_stats = {}
    with torch.no_grad():
        for items in tqdm(loader, desc=f"{args.method} unlabeled"):
            images = items["img"].to(device)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            preds = probs >= args.threshold
            scores = reduce_image_scores(probs, args.topk_ratio)

            for idx in range(images.shape[0]):
                relative_path = items["relative_path"][idx]
                group_name = items["group_name"][idx]
                pred_area_ratio = preds[idx].float().mean().item()
                image_score = scores[idx].item()
                prediction_mask_path, overlay_path = save_prediction_visualization(
                    preds[idx],
                    items["img_path"][idx],
                    visualization_dir,
                    len(rows) + 1,
                    args.overlay_alpha,
                    name_hint=safe_name_from_relative(relative_path),
                    output_size=image_size,
                    image_tensor=images[idx],
                )
                row = {
                    "index": len(rows) + 1,
                    "img_path": items["img_path"][idx],
                    "relative_path": relative_path,
                    "group_name": group_name,
                    "image_score": float(image_score),
                    "predicted_water": int(image_score >= args.image_threshold),
                    "pred_area_ratio": float(pred_area_ratio),
                    "prediction_mask_path": prediction_mask_path,
                    "overlay_path": overlay_path,
                }
                rows.append(row)

                stats = group_stats.setdefault(
                    group_name,
                    {"num_images": 0, "predicted_water": 0, "score_sum": 0.0, "area_sum": 0.0},
                )
                stats["num_images"] += 1
                stats["predicted_water"] += row["predicted_water"]
                stats["score_sum"] += row["image_score"]
                stats["area_sum"] += row["pred_area_ratio"]

    for stats in group_stats.values():
        count = max(1, stats["num_images"])
        stats["mean_image_score"] = stats.pop("score_sum") / count
        stats["mean_pred_area_ratio"] = stats.pop("area_sum") / count
        stats["predicted_water_ratio"] = stats["predicted_water"] / count

    with open(save_path / "predictions.csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "index",
                "img_path",
                "relative_path",
                "group_name",
                "image_score",
                "predicted_water",
                "pred_area_ratio",
                "prediction_mask_path",
                "overlay_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "method": args.method,
        "checkpoint_path": str(checkpoint_path),
        "image_dir": str(data_root),
        "save_path": str(save_path),
        "num_images": len(dataset),
        "image_size": image_size,
        "threshold": args.threshold,
        "topk_ratio": args.topk_ratio,
        "image_threshold": args.image_threshold,
        "stretch_to_square": stretch_to_square,
        "model_config": model_config,
        "group_stats": group_stats,
        "note": "No GT masks are available; IoU/F1 and other supervised segmentation metrics are not computed.",
    }
    with open(save_path / "prediction_summary.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(json.dumps({"num_images": len(dataset), "group_stats": group_stats}, indent=2, ensure_ascii=False))
    print(f"Saved unlabeled predictions to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Predict unlabeled image folder with a baseline segmentation model")
    parser.add_argument("--method", type=str, required=True, choices=["unet", "deeplabv3plus", "segformer", "u2net", "setr"])
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=0, help="0 means use checkpoint config")
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk_ratio", type=float, default=0.01)
    parser.add_argument("--image_threshold", type=float, default=0.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_cache", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--decoder_dim", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
