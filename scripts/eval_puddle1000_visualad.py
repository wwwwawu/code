import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_dataset_json.puddle1000 import (  # noqa: E402
    DEFAULT_PUDDLE1000_ROOT,
    create_all_puddle1000_meta,
    discover_puddle1000_subsets,
)
from test import test, setup_seed  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / 'experiments' / 'uwbench_l14_518_e5' / 'checkpoints' / 'final_model.pth'
DEFAULT_RESULT_ROOT = REPO_ROOT / 'result' / 'visualad-uwbench' / 'puddle-1000'


def parse_args():
    parser = argparse.ArgumentParser(description='Batch-evaluate a VisualAD checkpoint on all discovered Puddle-1000 subsets.')
    parser.add_argument('--puddle_root', type=Path, default=DEFAULT_PUDDLE1000_ROOT, help='Root folder of the Puddle-1000 extended dataset.')
    parser.add_argument('--checkpoint_path', type=Path, default=DEFAULT_CHECKPOINT, help='Trained VisualAD checkpoint to evaluate.')
    parser.add_argument('--result_root', type=Path, default=DEFAULT_RESULT_ROOT, help='Top-level output directory for puddle-1000 metrics and visualizations.')
    parser.add_argument('--sigma', type=int, default=4, help='Gaussian smoothing sigma used by test.py.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device passed through to test.py.')
    parser.add_argument('--metrics_mode', type=str, default='all', choices=['segmentation', 'all'], help='Metric set to compute for each subset.')
    parser.add_argument('--eval_threshold', type=float, default=0.5, help='Pixel threshold for binary segmentation metrics.')
    parser.add_argument('--sample_threshold', type=float, default=0.0, help='Image-level threshold for water / no-water accuracy.')
    parser.add_argument('--stretch_to_square', action='store_true', help='Use stretched-square preprocessing instead of aspect-ratio preserving padding.')
    parser.add_argument('--disable_analysis', action='store_true', help='Skip per-image visualizations and distribution plots.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used before each subset evaluation.')
    return parser.parse_args()


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


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


def main():
    args = parse_args()
    subsets = discover_puddle1000_subsets(args.puddle_root)
    if not subsets:
        raise FileNotFoundError(f'No Puddle-1000 subsets with images/ and masks/ were found under {args.puddle_root}')

    result_root = args.result_root
    meta_root = REPO_ROOT / 'data_meta' / 'puddle-1000'
    summary_root = result_root / 'summary'

    manifests = create_all_puddle1000_meta(args.puddle_root, meta_root)
    manifest_by_slug = {item['slug']: item for item in manifests}
    _write_json(summary_root / 'dataset_manifest.json', manifests)
    skipped_subsets = []

    per_subset_rows = []
    weighted_accumulator = {}
    weighted_total_samples = 0
    macro_accumulator = {}
    category_rows = {}

    for subset in subsets:
        manifest = manifest_by_slug[subset.slug]
        if manifest['num_samples'] == 0:
            skipped_subsets.append({
                'subset_slug': subset.slug,
                'category': manifest['category_dir'],
                'subset_name': subset.name,
                'reason': 'no image-mask pairs were found when generating meta',
            })
            continue
        save_path = result_root / manifest['category_dir'] / subset.slug
        test_args = SimpleNamespace(
            test_data_path=str(subset.root),
            test_meta_path=manifest['meta_path'],
            save_path=str(save_path),
            test_dataset='puddle1000',
            checkpoint_path=str(args.checkpoint_path),
            test_mode='test',
            sigma=args.sigma,
            eval_threshold=args.eval_threshold,
            metrics_mode=args.metrics_mode,
            sample_threshold=args.sample_threshold,
            skip_metrics=False,
            stretch_to_square=args.stretch_to_square,
            device=args.device,
            enable_analysis=not args.disable_analysis,
            seed=args.seed,
        )

        setup_seed(args.seed)
        run_output = test(test_args)
        metrics = run_output['metrics']
        mean_metrics = metrics['mean']

        row = {
            'subset_slug': subset.slug,
            'category': manifest['category_dir'],
            'subset_name': subset.name,
            'num_samples': run_output['num_samples'],
            'result_dir': str(save_path),
        }
        for key, value in sorted(mean_metrics.items()):
            if key == 'num_images':
                continue
            row[key] = round(value * 100, 3) if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key or 'iou' in key or 'ap' in key or 'auroc' in key or 'aupro' in key else round(value, 6)
            weighted_accumulator[key] = weighted_accumulator.get(key, 0.0) + value * run_output['num_samples']
            macro_accumulator.setdefault(key, []).append(value)
        weighted_total_samples += run_output['num_samples']
        per_subset_rows.append(row)
        category_rows.setdefault(manifest['category_dir'], []).append(row)

    macro_mean = {key: _safe_mean(values) for key, values in macro_accumulator.items()}
    weighted_mean = {
        key: (weighted_accumulator[key] / weighted_total_samples if weighted_total_samples else 0.0)
        for key in weighted_accumulator
    }

    category_summary_rows = []
    for category, rows in sorted(category_rows.items()):
        metric_keys = [key for key in rows[0].keys() if key not in {'subset_slug', 'category', 'subset_name', 'num_samples', 'result_dir'}]
        category_row = {
            'category': category,
            'num_subsets': len(rows),
            'num_samples': sum(int(r['num_samples']) for r in rows),
        }
        for key in metric_keys:
            category_row[key] = round(_safe_mean([float(r[key]) for r in rows]), 3)
        category_summary_rows.append(category_row)

    overall_summary = {
        'checkpoint_path': str(args.checkpoint_path),
        'puddle_root': str(args.puddle_root),
        'sigma': args.sigma,
        'metrics_mode': args.metrics_mode,
        'eval_threshold': args.eval_threshold,
        'sample_threshold': args.sample_threshold,
        'num_subsets_discovered': len(subsets),
        'num_subsets_evaluated': len(per_subset_rows),
        'num_subsets_skipped': len(skipped_subsets),
        'num_total_samples': weighted_total_samples,
        'macro_average_raw': macro_mean,
        'weighted_average_raw': weighted_mean,
        'macro_average_percent': {key: round(value * 100, 3) for key, value in macro_mean.items()},
        'weighted_average_percent': {key: round(value * 100, 3) for key, value in weighted_mean.items()},
        'skipped_subsets': skipped_subsets,
    }

    _write_json(summary_root / 'overall_summary.json', overall_summary)
    _write_json(summary_root / 'per_subset_metrics.json', per_subset_rows)
    _write_csv(summary_root / 'per_subset_metrics.csv', per_subset_rows)
    _write_markdown(summary_root / 'per_subset_metrics.md', per_subset_rows)
    _write_json(summary_root / 'category_summary.json', category_summary_rows)
    _write_csv(summary_root / 'category_summary.csv', category_summary_rows)
    _write_markdown(summary_root / 'category_summary.md', category_summary_rows)
    _write_json(summary_root / 'skipped_subsets.json', skipped_subsets)


if __name__ == '__main__':
    main()
