import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command, dry_run=False):
    print("\n" + "#" * 88)
    print(" ".join(str(item) for item in command))
    print("#" * 88)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def add_common_baseline_args(command, args):
    command.extend(
        [
            "--device",
            args.device,
            "--batch_size",
            args.batch_size,
            "--num_workers",
            args.num_workers,
            "--epoch",
            args.epoch,
            "--overlay_alpha",
            args.overlay_alpha,
            "--methods",
            *args.methods,
            "--segformer_variants",
            *args.segformer_variants,
        ]
    )
    if args.strict:
        command.append("--strict")
    if args.dry_run:
        command.append("--dry_run")
    return command


def main():
    parser = argparse.ArgumentParser("Run all baseline visual evaluations on three datasets")
    parser.add_argument("--datasets", nargs="*", default=["uwbench", "roadway", "city"],
                        choices=["uwbench", "roadway", "city"])
    parser.add_argument("--uwbench_path", type=str, default="../UW-Bench/UW-Bench/training_set")
    parser.add_argument("--uwbench_meta", type=str, default="data_meta/uwbench_meta.json")
    parser.add_argument("--roadway_path", type=str, default="../datasets/Image Dataset for Roadway Flooding/Dataset")
    parser.add_argument("--roadway_meta", type=str, default="data_meta/roadway_flooding_meta.json")
    parser.add_argument("--city_path", type=str, default="../datasets/城市内涝识别数据")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=str, default="8")
    parser.add_argument("--num_workers", type=str, default="4")
    parser.add_argument("--epoch", type=str, default="50")
    parser.add_argument("--overlay_alpha", type=str, default="0.45")
    parser.add_argument("--methods", nargs="*", default=["unet", "deeplabv3plus", "segformer", "u2net", "setr"])
    parser.add_argument(
        "--segformer_variants",
        nargs="*",
        default=["b5", "b2"],
        choices=["b0", "b1", "b2", "b3", "b4", "b5"],
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="run even if a checkpoint path is missing")
    args = parser.parse_args()

    commands = []
    datasets = set(args.datasets)

    if "uwbench" in datasets:
        command = [
            sys.executable,
            "baselines/eval_baselines.py",
            "--data_path",
            args.uwbench_path,
            "--meta_path",
            args.uwbench_meta,
            "--dataset_name",
            "uwbench",
        ]
        commands.append(add_common_baseline_args(command, args))

    if "roadway" in datasets:
        command = [
            sys.executable,
            "baselines/eval_roadway_flooding.py",
            "--data_path",
            args.roadway_path,
            "--meta_path",
            args.roadway_meta,
        ]
        commands.append(add_common_baseline_args(command, args))

    if "city" in datasets:
        command = [
            sys.executable,
            "baselines/eval_city_unlabeled.py",
            "--data_path",
            args.city_path,
            "--device",
            args.device,
            "--batch_size",
            args.batch_size,
            "--num_workers",
            args.num_workers,
            "--epoch",
            args.epoch,
            "--overlay_alpha",
            args.overlay_alpha,
            "--methods",
            *args.methods,
            "--segformer_variants",
            *args.segformer_variants,
        ]
        if args.strict:
            command.append("--strict")
        if args.dry_run:
            command.append("--dry_run")
        commands.append(command)

    for command in commands:
        code = run_command(command, dry_run=False)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
