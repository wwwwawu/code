import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None

import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "dataset.py").exists():
        REPO_ROOT = parent
        break
else:
    raise RuntimeError("Cannot locate repository root containing dataset.py")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from ProtoWD_lib.transform import ResizeMaxSize
from utils.transforms import ResizePadMask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_UWBENCH_ROOT = REPO_ROOT.parent / "uwbench"
DEFAULT_UWBENCH_META = REPO_ROOT / "data_meta" / "uwbench_meta.json"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.enc1 = DoubleConv(in_channels, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels[3], channels[4])

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0])
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    @staticmethod
    def _center_crop_like(skip, target):
        if skip.shape[-2:] == target.shape[-2:]:
            return skip
        diff_y = skip.size(-2) - target.size(-2)
        diff_x = skip.size(-1) - target.size(-1)
        return skip[
            :,
            :,
            diff_y // 2 : skip.size(-2) - (diff_y - diff_y // 2),
            diff_x // 2 : skip.size(-1) - (diff_x - diff_x // 2),
        ]

    def forward(self, x):
        input_hw = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self._center_crop_like(e4, d4), d4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self._center_crop_like(e3, d3), d3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self._center_crop_like(e2, d2), d2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._center_crop_like(e1, d1), d1], dim=1))
        logits = self.head(d1)
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


def build_transforms(image_size, stretch_to_square=False):
    if stretch_to_square:
        image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        return image_transform, mask_transform

    image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            ResizeMaxSize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    mask_transform = transforms.Compose([ResizePadMask(image_size), transforms.ToTensor()])
    return image_transform, mask_transform


def compute_metrics(logits_list, masks_list, threshold=0.5, compute_ap=False, ap_stride=1):
    logits = torch.cat(logits_list, dim=0).float()
    masks = torch.cat(masks_list, dim=0).float()
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    pred_flat = preds.reshape(-1).cpu().numpy().astype(np.uint8)
    mask_flat = (masks.reshape(-1).cpu().numpy() > 0.5).astype(np.uint8)
    prob_flat = probs.reshape(-1).cpu().numpy().astype(np.float32)

    tp = np.logical_and(pred_flat == 1, mask_flat == 1).sum()
    fp = np.logical_and(pred_flat == 1, mask_flat == 0).sum()
    fn = np.logical_and(pred_flat == 0, mask_flat == 1).sum()
    tn = np.logical_and(pred_flat == 0, mask_flat == 0).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    bg_iou = tn / (tn + fp + fn + 1e-8)
    miou = (iou + bg_iou) / 2.0
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    ap = None
    if compute_ap and average_precision_score is not None and len(np.unique(mask_flat)) > 1:
        if ap_stride > 1:
            ap = float(average_precision_score(mask_flat[::ap_stride], prob_flat[::ap_stride]))
        else:
            ap = float(average_precision_score(mask_flat, prob_flat))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "miou": float(miou),
        "accuracy": float(accuracy),
        "pixel_accuracy": float(accuracy),
        "ap": ap,
    }


def evaluate(model, dataloader, device, threshold, compute_ap=False, ap_stride=1):
    model.eval()
    logits_list = []
    masks_list = []
    val_loss = 0.0
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    with torch.no_grad():
        for items in tqdm(dataloader, desc="val", leave=False):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
            val_loss += float(loss.item())
            logits_list.append(logits.detach().cpu())
            masks_list.append(masks.detach().cpu())

    metrics = compute_metrics(
        logits_list,
        masks_list,
        threshold=threshold,
        compute_ap=compute_ap,
        ap_stride=ap_stride,
    )
    metrics["loss"] = val_loss / max(1, len(dataloader))
    return metrics


def train(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)
    checkpoints_dir = os.path.join(args.save_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    image_transform, mask_transform = build_transforms(args.image_size, args.stretch_to_square)
    train_data = Dataset(
        root=args.data_path,
        transform=image_transform,
        target_transform=mask_transform,
        dataset_name=args.dataset_name,
        mode=args.train_mode,
        meta_path=args.meta_path,
    )
    val_data = Dataset(
        root=args.data_path,
        transform=image_transform,
        target_transform=mask_transform,
        dataset_name=args.dataset_name,
        mode=args.val_mode,
        meta_path=args.meta_path,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(base_channels=args.base_channels).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_iou = -1.0
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    config = vars(args).copy()
    config.update({"num_train": len(train_data), "num_val": len(val_data)})
    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        metrics = checkpoint.get("metrics", {})
        best_iou = float(checkpoint.get("best_iou", metrics.get("iou", best_iou)))
        history = checkpoint.get("history", history)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from {args.resume} at epoch {start_epoch - 1}; target epoch is {args.epochs}.")

    remaining_epochs = max(1, args.epochs - start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for items in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        scheduler.step()
        train_loss /= max(1, len(train_loader))
        should_eval = args.eval_freq > 0 and (epoch % args.eval_freq == 0 or epoch == args.epochs)
        metrics = {}
        if should_eval:
            metrics = evaluate(
                model,
                val_loader,
                device,
                args.threshold,
                compute_ap=args.compute_ap,
                ap_stride=args.ap_stride,
            )
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)

        if should_eval:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_iou={metrics['iou']:.4f} | val_f1={metrics['f1']:.4f} | "
                f"val_recall={metrics['recall']:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        is_best = bool(should_eval and metrics["iou"] > best_iou)
        if is_best:
            best_iou = metrics["iou"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": config,
                    "best_iou": best_iou,
                    "history": history,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "target_epochs": args.epochs,
                },
                os.path.join(checkpoints_dir, "best_model.pth"),
            )
        elif should_eval:
            epochs_without_improvement += 1

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": config,
                "best_iou": best_iou,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "target_epochs": args.epochs,
            }
            torch.save(checkpoint_payload, os.path.join(checkpoints_dir, f"epoch_{epoch}.pth"))
            torch.save(checkpoint_payload, os.path.join(checkpoints_dir, "latest.pth"))

        with open(os.path.join(args.save_path, "metrics_history.json"), "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if should_eval and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping: val_iou did not improve for {args.patience} epochs.")
            break

    best_payload = {"best_iou": best_iou, "history": history}
    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(best_payload, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser("Train U-Net baseline for road waterlogging segmentation")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_UWBENCH_ROOT), help="dataset root")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_UWBENCH_META), help="meta json path")
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--train_mode", type=str, default="train")
    parser.add_argument("--val_mode", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="experiments/unet")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--compute_ap", action="store_true", help="compute pixel AP during validation")
    parser.add_argument("--ap_stride", type=int, default=16, help="subsample stride for pixel AP when enabled")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=0, help="validation frequency in epochs; 0 disables validation")
    parser.add_argument("--patience", type=int, default=0, help="early-stop patience by validation IoU; 0 disables")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to continue training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
