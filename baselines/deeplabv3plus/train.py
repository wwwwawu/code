import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parent / "model_cache"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)]
            + [
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
                for rate in rates
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
        )
        self.project = nn.Sequential(
            ConvBNReLU(out_channels * (len(rates) + 2), out_channels, kernel_size=1, padding=0),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class ResNetEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=False):
        super().__init__()
        from torchvision.models import resnet101, resnet50

        weights = None
        if pretrained:
            try:
                if backbone == "resnet101":
                    from torchvision.models import ResNet101_Weights

                    weights = ResNet101_Weights.DEFAULT
                else:
                    from torchvision.models import ResNet50_Weights

                    weights = ResNet50_Weights.DEFAULT
            except ImportError:
                weights = "DEFAULT"

        factory = resnet101 if backbone == "resnet101" else resnet50
        try:
            net = factory(weights=weights, replace_stride_with_dilation=[False, False, True])
        except TypeError:
            net = factory(pretrained=bool(pretrained), replace_stride_with_dilation=[False, False, True])

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)
        return low_level, high_level


class DeepLabV3PlusBinary(nn.Module):
    def __init__(self, backbone="resnet50", pretrained_backbone=False):
        super().__init__()
        if backbone not in {"resnet50", "resnet101"}:
            raise ValueError("backbone must be resnet50 or resnet101")
        self.backbone_name = backbone
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained_backbone)
        self.aspp = ASPP(2048, out_channels=256, rates=(6, 12, 18))
        self.low_proj = ConvBNReLU(256, 48, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(304, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x):
        input_hw = x.shape[-2:]
        low_level, high_level = self.encoder(x)
        x = self.aspp(high_level)
        low_level = self.low_proj(low_level)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([x, low_level], dim=1))
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


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


def update_confusion_from_batch(logits, masks, threshold):
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    gts = masks > 0.5
    tp = torch.logical_and(preds, gts).sum().item()
    fp = torch.logical_and(preds, torch.logical_not(gts)).sum().item()
    fn = torch.logical_and(torch.logical_not(preds), gts).sum().item()
    tn = torch.logical_and(torch.logical_not(preds), torch.logical_not(gts)).sum().item()
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    bg_iou = tn / (tn + fp + fn + 1e-8)
    miou = (iou + bg_iou) / 2.0
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "miou": float(miou),
        "accuracy": float(accuracy),
        "pixel_accuracy": float(accuracy),
    }


def evaluate(model, dataloader, device, threshold):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for items in tqdm(dataloader, desc="val", leave=False):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()
            logits = model(images)
            batch_tp, batch_fp, batch_fn, batch_tn = update_confusion_from_batch(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn
    return compute_metrics(tp, fp, fn, tn)


def train(args):
    setup_seed(args.seed)
    if args.model_cache:
        os.environ["TORCH_HOME"] = os.path.abspath(args.model_cache)
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
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

    model = DeepLabV3PlusBinary(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_iou = -1.0
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
        best_iou = float(checkpoint.get("best_iou", best_iou))
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
        metrics = evaluate(model, val_loader, device, args.threshold) if should_eval else {}
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if should_eval:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_iou={metrics['iou']:.4f} | val_f1={metrics['f1']:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

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
        if should_eval and metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            checkpoint_payload["best_iou"] = best_iou
            torch.save(checkpoint_payload, os.path.join(checkpoints_dir, "best_model.pth"))

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(checkpoint_payload, os.path.join(checkpoints_dir, f"epoch_{epoch}.pth"))
            torch.save(checkpoint_payload, os.path.join(checkpoints_dir, "latest.pth"))

        with open(os.path.join(args.save_path, "metrics_history.json"), "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"best_iou": best_iou, "history": history}, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser("Train DeepLabV3+ baseline for road waterlogging segmentation")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_UWBENCH_ROOT), help="dataset root")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_UWBENCH_META), help="meta json path")
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--train_mode", type=str, default="train")
    parser.add_argument("--val_mode", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="experiments/deeplabv3plus")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--model_cache", type=str, default=str(DEFAULT_MODEL_CACHE),
                        help="torch model cache root; weights are stored under model_cache/hub/checkpoints")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=0, help="validation frequency in epochs; 0 disables validation")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to continue training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
