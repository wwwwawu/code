import argparse
import csv
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_dataset_json.uwbench7_puddle_train import (  # noqa: E402
    DEFAULT_META_ROOT,
    DEFAULT_PUDDLE_ROOT,
    DEFAULT_UWBENCH_ROOT,
    build_uwbench7_puddle_meta,
)
from test import setup_seed as setup_test_seed, test  # noqa: E402
from train import setup_seed as setup_train_seed, train  # noqa: E402


DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / 'experiments' / 'uwbench7+puddle-train'


def parse_args():
    parser = argparse.ArgumentParser(description='Train VisualAD on UW-Bench train + Puddle train, then evaluate UW-Bench val and Puddle validation subsets.')
    parser.add_argument('--uwbench_root', type=Path, default=DEFAULT_UWBENCH_ROOT)
    parser.add_argument('--puddle_root', type=Path, default=DEFAULT_PUDDLE_ROOT)
    parser.add_argument('--experiment_root', type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument('--meta_root', type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument('--backbone', type=str, default='ViT-L/14@336px')
    parser.add_argument('--features_list', type=int, nargs='*', default=[6, 12, 18, 24])
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--sigma', type=int, default=4)
    parser.add_argument('--eval_threshold', type=float, default=0.5)
    parser.add_argument('--sample_threshold', type=float, default=0.0)
    parser.add_argument('--metrics_mode', type=str, default='all', choices=['segmentation', 'all'])
    parser.add_argument('--fast_metrics_stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stretch_to_square', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--checkpoint_path', type=Path, default=None, help='Use an existing checkpoint instead of training in this run. Required with --skip_train.')
    return parser.parse_args()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[h]) for h in headers) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _format_command(args) -> str:
    parts = ['python', 'scripts/run_uwbench7_puddle_train.py']
    for key, value in vars(args).items():
        option = f'--{key}'
        if isinstance(value, bool):
            if value:
                parts.append(option)
        elif isinstance(value, list):
            parts.append(option)
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.extend([option, str(value)])
    return ' '.join(shlex.quote(part) for part in parts)


def _metric_to_percent(value):
    return round(float(value) * 100.0, 3)


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    experiment_root = args.experiment_root
    checkpoints_root = experiment_root / 'checkpoints'
    result_root = experiment_root / f'result-sigma{args.sigma}'
    top_log_path = experiment_root / 'log.txt'

    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest = build_uwbench7_puddle_meta(args.uwbench_root, args.puddle_root, args.meta_root)

    train_meta_path = Path(manifest['train_meta_path'])
    eval_plan = manifest['evaluation_plan']

    if args.skip_train:
        if args.checkpoint_path is None:
            raise ValueError('--checkpoint_path is required when --skip_train is set.')
        checkpoint_path = args.checkpoint_path
    else:
        train_args = SimpleNamespace(
            train_data_path=manifest['train_root'],
            train_meta_path=str(train_meta_path),
            save_path=str(checkpoints_root),
            train_dataset='mixed_water',
            backbone=args.backbone,
            feature_config=os.path.join('configs', 'backbone_layers.yaml'),
            features_list=args.features_list,
            epoch=args.epoch,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            image_size=args.image_size,
            stretch_to_square=args.stretch_to_square,
            print_freq=args.print_freq,
            save_freq=args.save_freq,
            seed=args.seed,
            device=args.device,
        )
        setup_train_seed(args.seed)
        train(train_args)
        checkpoint_path = checkpoints_root / 'final_model.pth'

    eval_rows = []
    category_rows = {}
    weighted = {}
    weighted_count = 0
    for item in eval_plan:
        save_path = result_root / item['result_subdir']
        test_args = SimpleNamespace(
            test_data_path=item['root'],
            test_meta_path=item['meta_path'],
            save_path=str(save_path),
            test_dataset=item['dataset_name'],
            checkpoint_path=str(checkpoint_path),
            test_mode=item['test_mode'],
            sigma=args.sigma,
            eval_threshold=args.eval_threshold,
            metrics_mode=args.metrics_mode,
            sample_threshold=args.sample_threshold,
            fast_metrics_stride=args.fast_metrics_stride,
            skip_metrics=False,
            stretch_to_square=args.stretch_to_square,
            device=args.device,
            enable_analysis=True,
            seed=args.seed,
        )
        setup_test_seed(args.seed)
        output = test(test_args)
        metrics = output['metrics']['mean']
        row = {
            'split': item['name'],
            'category': item['category'],
            'num_samples': output['num_samples'],
            'result_dir': str(save_path),
        }
        for key in ['precision', 'recall', 'f1', 'iou', 'miou', 'ap', 'accuracy', 'pixel_auroc', 'pixel_ap', 'pixel_f1', 'image_auroc', 'image_ap', 'image_f1']:
            if key in metrics:
                row[key] = _metric_to_percent(metrics[key])
                weighted[key] = weighted.get(key, 0.0) + float(metrics[key]) * output['num_samples']
        weighted_count += output['num_samples']
        eval_rows.append(row)
        category_rows.setdefault(item['category'], []).append(row)

    category_summary = []
    for category, rows in sorted(category_rows.items()):
        metric_keys = [k for k in rows[0] if k not in {'split', 'category', 'num_samples', 'result_dir'}]
        category_summary.append({
            'category': category,
            'num_splits': len(rows),
            'num_samples': sum(int(r['num_samples']) for r in rows),
            **{key: round(_safe_mean([float(r[key]) for r in rows]), 3) for key in metric_keys},
        })

    macro_average_percent = {}
    if eval_rows:
        metric_keys = [k for k in eval_rows[0] if k not in {'split', 'category', 'num_samples', 'result_dir'}]
        macro_average_percent = {
            key: round(_safe_mean([float(row[key]) for row in eval_rows]), 3)
            for key in metric_keys
        }

    overall_summary = {
        'checkpoint_path': str(checkpoint_path),
        'experiment_root': str(experiment_root),
        'sigma': args.sigma,
        'metrics_mode': args.metrics_mode,
        'fast_metrics_stride': args.fast_metrics_stride,
        'eval_threshold': args.eval_threshold,
        'sample_threshold': args.sample_threshold,
        'macro_average_percent': macro_average_percent,
        'weighted_average_percent': {key: round((value / weighted_count) * 100.0, 3) for key, value in weighted.items()} if weighted_count else {},
        'num_eval_splits': len(eval_rows),
        'num_total_eval_samples': weighted_count,
    }

    _write_json(result_root / 'overall_summary.json', overall_summary)
    _write_json(result_root / 'per_split_metrics.json', eval_rows)
    _write_csv(result_root / 'per_split_metrics.csv', eval_rows)
    _write_markdown(result_root / 'per_split_metrics.md', eval_rows)
    _write_json(result_root / 'category_summary.json', category_summary)
    _write_csv(result_root / 'category_summary.csv', category_summary)
    _write_markdown(result_root / 'category_summary.md', category_summary)

    top_log_lines = [
        f'generated_at: {datetime.now().isoformat(timespec="seconds")}',
        f'command: {_format_command(args)}',
        f'uwbench_root: {args.uwbench_root}',
        f'puddle_root: {args.puddle_root}',
        f'train_meta_path: {train_meta_path}',
        f'checkpoint_path: {checkpoint_path}',
        f'checkpoints_dir: {checkpoints_root}',
        f'result_root: {result_root}',
        f'metrics_mode: {args.metrics_mode}',
        f'fast_metrics_stride: {args.fast_metrics_stride}',
        f'sigma: {args.sigma}',
        f'uwbench_train_samples: {manifest["train_manifest"]["uwbench_train_samples"]}',
        'puddle_train_subsets:',
    ]
    for subset in manifest['train_manifest']['puddle_train_subsets']:
        top_log_lines.append(f'  - {subset["name"]}: {subset["num_samples"]}')
    top_log_lines.append('evaluation_splits:')
    for row in eval_rows:
        top_log_lines.append(f'  - {row["split"]}: {row["num_samples"]} -> {row["result_dir"]}')
    top_log_lines.append('weighted_average_percent:')
    for key, value in overall_summary['weighted_average_percent'].items():
        top_log_lines.append(f'  {key}: {value}')
    top_log_lines.append('macro_average_percent:')
    for key, value in overall_summary['macro_average_percent'].items():
        top_log_lines.append(f'  {key}: {value}')
    top_log_path.write_text('\n'.join(top_log_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
