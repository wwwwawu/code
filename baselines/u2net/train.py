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


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def upsample_like(src, target):
    return F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool(hx1))
        hx3 = self.rebnconv3(self.pool(hx2))
        hx4 = self.rebnconv4(self.pool(hx3))
        hx5 = self.rebnconv5(self.pool(hx4))
        hx6 = self.rebnconv6(self.pool(hx5))
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), dim=1))
        hx6dup = upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), dim=1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool(hx1))
        hx3 = self.rebnconv3(self.pool(hx2))
        hx4 = self.rebnconv4(self.pool(hx3))
        hx5 = self.rebnconv5(self.pool(hx4))
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), dim=1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool(hx1))
        hx3 = self.rebnconv3(self.pool(hx2))
        hx4 = self.rebnconv4(self.pool(hx3))
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool(hx1))
        hx3 = self.rebnconv3(self.pool(hx2))
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), dim=1))
        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.stage2 = RSU6(64, 32, 128)
        self.stage3 = RSU5(128, 64, 256)
        self.stage4 = RSU4(256, 128, 512)
        self.stage5 = RSU4F(512, 256, 512)
        self.stage6 = RSU4F(512, 256, 512)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        input_hw = x.shape[-2:]
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool(hx1))
        hx3 = self.stage3(self.pool(hx2))
        hx4 = self.stage4(self.pool(hx3))
        hx5 = self.stage5(self.pool(hx4))
        hx6 = self.stage6(self.pool(hx5))

        hx6up = upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), dim=1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        d1 = self.side1(hx1d)
        d2 = upsample_like(self.side2(hx2d), d1)
        d3 = upsample_like(self.side3(hx3d), d1)
        d4 = upsample_like(self.side4(hx4d), d1)
        d5 = upsample_like(self.side5(hx5d), d1)
        d6 = upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
        outputs = [d0, d1, d2, d3, d4, d5, d6]
        return tuple(
            F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
            if out.shape[-2:] != input_hw else out
            for out in outputs
        )


class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.stage2 = RSU6(64, 16, 64)
        self.stage3 = RSU5(64, 16, 64)
        self.stage4 = RSU4(64, 16, 64)
        self.stage5 = RSU4F(64, 16, 64)
        self.stage6 = RSU4F(64, 16, 64)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        input_hw = x.shape[-2:]
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool(hx1))
        hx3 = self.stage3(self.pool(hx2))
        hx4 = self.stage4(self.pool(hx3))
        hx5 = self.stage5(self.pool(hx4))
        hx6 = self.stage6(self.pool(hx5))
        hx6up = upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), dim=1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))
        d1 = self.side1(hx1d)
        d2 = upsample_like(self.side2(hx2d), d1)
        d3 = upsample_like(self.side3(hx3d), d1)
        d4 = upsample_like(self.side4(hx4d), d1)
        d5 = upsample_like(self.side5(hx5d), d1)
        d6 = upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
        outputs = [d0, d1, d2, d3, d4, d5, d6]
        return tuple(
            F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
            if out.shape[-2:] != input_hw else out
            for out in outputs
        )


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


def build_model(model_type="u2net"):
    if model_type == "u2netp":
        return U2NETP()
    return U2NET()


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


def side_output_loss(outputs, masks, bce_loss, dice_loss):
    losses = []
    for output in outputs:
        losses.append(bce_loss(output, masks) + dice_loss(output, masks))
    return sum(losses)


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
            logits = model(images)[0]
            batch_tp, batch_fp, batch_fn, batch_tn = update_confusion_from_batch(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn
    return compute_metrics(tp, fp, fn, tn)


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

    model = build_model(args.model_type).to(device)
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

            outputs = model(images)
            loss = side_output_loss(outputs, masks, bce_loss, dice_loss)
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
    parser = argparse.ArgumentParser("Train U2Net baseline for road waterlogging segmentation")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_UWBENCH_ROOT), help="dataset root")
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_UWBENCH_META), help="meta json path")
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--train_mode", type=str, default="train")
    parser.add_argument("--val_mode", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="experiments/u2net")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="u2net", choices=["u2net", "u2netp"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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
