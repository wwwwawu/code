import numpy as np
import logging
import json
try:
    from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
except ImportError:
    def _binary_clf_curve(y_true, y_score):
        y_true = np.asarray(y_true).astype(bool)
        y_score = np.asarray(y_score)
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        return fps.astype(float), tps.astype(float), y_score[threshold_idxs]

    def auc(x, y):
        return np.trapz(y, x)

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return 0.0
        fps, tps, _ = _binary_clf_curve(y_true, y_score)
        fps = np.r_[0, fps]
        tps = np.r_[0, tps]
        fpr = fps / fps[-1] if fps[-1] > 0 else fps
        tpr = tps / tps[-1] if tps[-1] > 0 else tps
        return auc(fpr, tpr)

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(y_true) == 0:
            return np.array([1.0]), np.array([0.0]), np.array([])
        fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
        precision = tps / np.maximum(tps + fps, 1e-12)
        recall = tps / np.maximum(tps[-1], 1e-12)
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    def average_precision_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt=None):
        rows = [headers] + rows
        widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
        lines = []
        for row_idx, row in enumerate(rows):
            lines.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
            if row_idx == 0:
                lines.append("-+-".join("-" * width for width in widths))
        return "\n".join(lines)

try:
    from skimage import measure
except ImportError:
    measure = None

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    if measure is None:
        return 0.0
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = np.min(amaps), np.max(amaps)
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[:] = amaps > th
        pro = []
        for i in range(len(amaps)):
            mask = masks[i]
            amap = binary_amaps[i]
            if np.sum(mask) == 0:
                continue  # Skip if no anomaly region
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                tp = np.sum(amap[coords[:, 0], coords[:, 1]])
                pro.append(tp / region.area)
        avg_pro = np.mean(pro) if pro else 0
        pros.append(avg_pro)
        fp = np.sum(binary_amaps * (1 - masks))
        total_fp_pixels = np.sum(1 - masks)
        fpr = fp / total_fp_pixels if total_fp_pixels != 0 else 0
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    valid = fprs <= expect_fpr
    if not np.any(valid):
        return 0.0
    fpr_valid = fprs[valid]
    pro_valid = pros[valid]
    if len(fpr_valid) < 2:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / (fpr_valid.max() - fpr_valid.min())
    return auc(fpr_norm, pro_valid)

def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def _subsample_pair(gt_values, pred_values, stride):
    if stride <= 1 or gt_values.size == 0:
        return gt_values, pred_values
    return gt_values[::stride], pred_values[::stride]


def _safe_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.0


def compute_metrics(
    results,
    obj_list,
    logger,
    eval_threshold=0.5,
    metrics_mode="all",
    sample_threshold=0.0,
    fast_metrics_stride=1,
):
    table_ls = []
    summary = {
        "metrics_mode": metrics_mode,
        "eval_threshold": float(eval_threshold),
        "sample_threshold": float(sample_threshold),
        "fast_metrics_stride": int(fast_metrics_stride),
        "per_class": {},
        "mean": {},
    }
    for obj in obj_list:
        obj_data = results[obj]
        # Pixel-level data
        gt_chunks, pr_chunks = [], []
        for mask_batch, anomaly_map_batch in zip(obj_data['imgs_masks'], obj_data['anomaly_maps']):
            mask_np = mask_batch.squeeze().cpu().numpy().astype(np.uint8, copy=False)
            amap_np = anomaly_map_batch.squeeze().cpu().numpy().astype(np.float32, copy=False)
            gt_chunks.append(mask_np.reshape(-1))
            pr_chunks.append(amap_np.reshape(-1))
        gt_px = np.concatenate(gt_chunks) if gt_chunks else np.array([], dtype=np.uint8)
        pr_px = np.concatenate(pr_chunks) if pr_chunks else np.array([], dtype=np.float32)
        # UW-Bench paper style metrics. Avoid materializing a full sigmoid array:
        # sigmoid(x) >= t is equivalent to x >= log(t / (1 - t)).
        clipped_threshold = float(np.clip(eval_threshold, 1e-6, 1.0 - 1e-6))
        logit_threshold = np.log(clipped_threshold / (1.0 - clipped_threshold))
        pred_bin = (pr_px >= logit_threshold).astype(np.uint8)
        gt_bin = (gt_px > 0.5).astype(np.uint8)

        tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
        fp = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
        fn = np.logical_and(pred_bin == 0, gt_bin == 1).sum()
        tn = np.logical_and(pred_bin == 0, gt_bin == 0).sum()

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        iou = _safe_divide(tp, tp + fp + fn)
        bg_iou = _safe_divide(tn, tn + fp + fn)
        miou = (iou + bg_iou) / 2.0
        gt_px_rank, pr_px_rank = _subsample_pair(gt_px, pr_px, fast_metrics_stride)
        pixel_ap = average_precision_score(gt_px_rank, pr_px_rank) if gt_px_rank.size else 0.0

        gt_sp = np.array(obj_data['gt_sp'])
        pr_sp = np.array(obj_data['pr_sp'])
        sample_pred = (pr_sp >= sample_threshold).astype(np.uint8) if pr_sp.size else np.array([], dtype=np.uint8)
        sample_gt = gt_sp.astype(np.uint8)
        image_accuracy = np.mean(sample_pred == sample_gt) if gt_sp.size else 0.0

        class_metrics = {
            "num_images": int(len(obj_data['imgs_masks'])),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "miou": float(miou),
            "ap": float(pixel_ap),
            "accuracy": float(image_accuracy),
            # Backward-compatible aliases
            "pixel_precision_at_0_5": float(precision),
            "pixel_recall_at_0_5": float(recall),
            "pixel_f1_at_0_5": float(f1),
            "pixel_iou_at_0_5": float(iou),
            "pixel_miou_at_0_5": float(miou),
            "sample_accuracy": float(image_accuracy),
        }

        table = [
            obj,
            f"{precision * 100:.1f}",
            f"{recall * 100:.1f}",
            f"{f1 * 100:.1f}",
            f"{iou * 100:.1f}",
            f"{miou * 100:.1f}",
            f"{pixel_ap * 100:.1f}",
            f"{image_accuracy * 100:.1f}",
        ]

        if metrics_mode == "all":
            # Calculate anomaly-detection style metrics
            pixel_auroc = _safe_roc_auc_score(gt_px_rank, pr_px_rank)
            image_auroc = _safe_roc_auc_score(gt_sp, pr_sp)
            image_ap = average_precision_score(gt_sp, pr_sp) if gt_sp.size else 0

            # F1 scores and the corresponding IoU at the best F1 threshold.
            precisions, recalls, _ = precision_recall_curve(gt_px_rank, pr_px_rank)
            pixel_f1 = np.max(2 * (precisions * recalls) / (precisions + recalls + 1e-8)) if gt_px_rank.size else 0
            precisions_sp, recalls_sp, _ = precision_recall_curve(gt_sp, pr_sp)
            image_f1 = np.max(2 * (precisions_sp * recalls_sp) / (precisions_sp + recalls_sp + 1e-8)) if gt_sp.size else 0

            class_metrics.update({
                "pixel_auroc": float(pixel_auroc),
                "pixel_ap": float(pixel_ap),
                "pixel_f1": float(pixel_f1),
                "image_auroc": float(image_auroc),
                "image_ap": float(image_ap),
                "image_f1": float(image_f1),
                # Backward-compatible aliases
                "pixel_f1_best": float(pixel_f1),
                "pixel_ap": float(pixel_ap),
                "sample_auroc": float(image_auroc),
                "sample_f1_best": float(image_f1),
                "sample_ap": float(image_ap),
            })
            table.extend([
                f"{pixel_auroc * 100:.1f}",
                f"{pixel_ap * 100:.1f}",
                f"{pixel_f1 * 100:.1f}",
                f"{image_auroc * 100:.1f}",
                f"{image_ap * 100:.1f}",
                f"{image_f1 * 100:.1f}",
            ])

        summary["per_class"][obj] = class_metrics
        table_ls.append(table)
    
    # === New: Calculate and add mean row ===
    if len(table_ls) == 0:
        return

    # Extract numeric part (skip first column class name)
    numeric_data = []
    for row in table_ls:
        numeric_values = [float(x.strip('%')) for x in row[1:]]  # Remove possible % and convert to float
        numeric_data.append(numeric_values)

    # Calculate mean for each column
    mean_values = np.array(numeric_data).mean(axis=0)
    mean_values = [f"{v:.1f}" for v in mean_values]

    # Add mean row
    mean_row = ['Mean'] + mean_values
    table_ls.append(mean_row)
    metric_keys = list(summary["per_class"][obj_list[0]].keys()) if obj_list else []
    for key in metric_keys:
        values = [summary["per_class"][obj][key] for obj in obj_list]
        summary["mean"][key] = float(np.mean(values)) if values else 0.0

    # === Generate table ===
    headers = ['Class', 'Precision', 'Recall', 'F1', 'IoU', 'mIoU', 'AP', 'Accuracy']
    if metrics_mode == "all":
        headers.extend([
            'Pixel-AUROC', 'Pixel-AP', 'Pixel-F1',
            'Image-AUROC', 'Image-AP', 'Image-F1'
        ])
    results_table = tabulate(table_ls, headers=headers, tablefmt='pipe')
    logger.info("\n%s", results_table)
    return summary
