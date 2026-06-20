import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UWBENCH_ROOT = REPO_ROOT.parent / "UW-Bench" / "UW-Bench" / "training_set"
DEFAULT_UWBENCH_META = REPO_ROOT / "data_meta" / "uwbench_meta.json"


def run_command(command, dry_run=False):
    print("\n" + "=" * 80)
    print(" ".join(str(item) for item in command))
    print("=" * 80)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def maybe_add_method(runs, name, script, experiment, checkpoint, save_path, extra_args, strict):
    checkpoint_path = REPO_ROOT / checkpoint
    if checkpoint_path.exists() or strict:
        runs.append((name, script, checkpoint, save_path, extra_args))
    else:
        print(f"[skip] {name}: checkpoint not found: {checkpoint}")
        print(f"       expected experiment directory: {experiment}")


def main():
    parser = argparse.ArgumentParser("Evaluate baseline checkpoints and save prediction visualizations")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_UWBENCH_ROOT))
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_UWBENCH_META))
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--test_mode", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=str, default="8")
    parser.add_argument("--num_workers", type=str, default="4")
    parser.add_argument("--epoch", type=str, default="50")
    parser.add_argument("--methods", nargs="*", default=["unet", "deeplabv3plus", "segformer", "u2net", "setr"])
    parser.add_argument("--result_tag", type=str, default="", help="default: <dataset_name>_visual_epoch<epoch>")
    parser.add_argument("--overlay_alpha", type=str, default="0.45")
    parser.add_argument("--segformer_variant", type=str, default="b5", choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    parser.add_argument(
        "--segformer_variants",
        nargs="*",
        default=["b5", "b2"],
        choices=["b0", "b1", "b2", "b3", "b4", "b5"],
        help="SegFormer variants to evaluate when method 'segformer' is selected",
    )
    parser.add_argument("--setr_backbone", type=str, default="vit_b16")
    parser.add_argument("--deeplab_experiment", type=str, default="experiments/deeplabv3plus_resnet101")
    parser.add_argument("--segformer_experiment", type=str, default="")
    parser.add_argument("--setr_experiment", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="run even if a checkpoint path is missing")
    args = parser.parse_args()

    epoch_name = f"epoch_{args.epoch}.pth"
    result_tag = args.result_tag or f"{args.dataset_name}_visual_epoch{args.epoch}"
    methods = set(args.methods)
    runs = []

    setr_experiment = args.setr_experiment or f"experiments/setr_{args.setr_backbone}"

    if "unet" in methods:
        experiment = "experiments/unet"
        maybe_add_method(
            runs,
            "unet",
            "baselines/unet/test.py",
            experiment,
            f"{experiment}/checkpoints/{epoch_name}",
            f"{experiment}/{result_tag}",
            [],
            args.strict,
        )

    if "deeplabv3plus" in methods:
        maybe_add_method(
            runs,
            "deeplabv3plus",
            "baselines/deeplabv3plus/test.py",
            args.deeplab_experiment,
            f"{args.deeplab_experiment}/checkpoints/{epoch_name}",
            f"{args.deeplab_experiment}/{result_tag}",
            ["--backbone", "resnet101"],
            args.strict,
        )

    if "segformer" in methods:
        segformer_variants = args.segformer_variants or [args.segformer_variant]
        for variant in segformer_variants:
            segformer_experiment = args.segformer_experiment or f"experiments/segformer_{variant}"
            maybe_add_method(
                runs,
                f"segformer_{variant}",
                "baselines/segformer/test.py",
                segformer_experiment,
                f"{segformer_experiment}/checkpoints/{epoch_name}",
                f"{segformer_experiment}/{result_tag}",
                ["--variant", variant],
                args.strict,
            )

    if "u2net" in methods:
        experiment = "experiments/u2net"
        maybe_add_method(
            runs,
            "u2net",
            "baselines/u2net/test.py",
            experiment,
            f"{experiment}/checkpoints/{epoch_name}",
            f"{experiment}/{result_tag}",
            ["--model_type", "u2net"],
            args.strict,
        )

    if "setr" in methods:
        maybe_add_method(
            runs,
            "setr",
            "baselines/setr/test.py",
            setr_experiment,
            f"{setr_experiment}/checkpoints/{epoch_name}",
            f"{setr_experiment}/{result_tag}",
            ["--backbone", args.setr_backbone],
            args.strict,
        )

    if not runs:
        print("No runnable checkpoints found. Use --strict to force commands, or check experiment paths.")
        return 1

    for _name, script, checkpoint, save_path, extra_args in runs:
        command = [
            sys.executable,
            script,
            "--data_path",
            args.data_path,
            "--meta_path",
            args.meta_path,
            "--dataset_name",
            args.dataset_name,
            "--test_mode",
            args.test_mode,
            "--checkpoint_path",
            checkpoint,
            "--save_path",
            save_path,
            "--device",
            args.device,
            "--batch_size",
            args.batch_size,
            "--num_workers",
            args.num_workers,
            "--overlay_alpha",
            args.overlay_alpha,
            *extra_args,
        ]
        code = run_command(command, dry_run=args.dry_run)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
