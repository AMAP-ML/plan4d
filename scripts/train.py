"""Launch one of the three Plan4D training stages from its canonical YAML."""

import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "scene-flow": ("scene_flow.yaml", "Depth-Anything-3/training/train_scene_flow_head.py"),
    "stitching": ("stitching.yaml", "model_stitching_training.py"),
    "joint": ("joint.yaml", "DiffSynth-Studio/examples/wanvideo/model_training/train.py"),
}


def expand_model_paths(groups, *, require_files=True):
    result = []
    for group in groups:
        patterns = group if isinstance(group, list) else [group]
        matches = []
        for pattern in patterns:
            files = sorted(glob.glob(str(ROOT / pattern)))
            if not files and require_files:
                raise FileNotFoundError(f"No model weights match {pattern}")
            matches.extend(files or [str(ROOT / pattern)])
        result.append(matches if isinstance(group, list) else matches[0] if len(matches) == 1 else matches)
    return json.dumps(result)


def build_command(args, overrides):
    config_name, entrypoint = STAGES[args.stage]
    config_path = Path(args.config).resolve() if args.config else ROOT / "configs" / config_name
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive.")
    with (ROOT / "configs/accelerate.yaml").open(encoding="utf-8") as handle:
        distributed_type = str(yaml.safe_load(handle)["distributed_type"])
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", str(ROOT / "configs/accelerate.yaml"),
        "--num_processes", str(args.num_processes),
    ]
    if args.num_processes > 1 and distributed_type in {"NO", "MULTI_GPU"}:
        command.append("--multi_gpu")
    command.append(str(ROOT / entrypoint))
    if args.stage == "scene-flow":
        command.extend(["--config", str(config_path)])
    else:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for name, value in config["args"].items():
            if value is None or value is False:
                continue
            option = f"--{name}"
            if value is True:
                command.append(option)
            elif name == "model_paths":
                command.extend([option, expand_model_paths(value, require_files=not args.dry_run)])
            elif isinstance(value, list):
                for item in value:
                    command.extend([option, str(item)])
            else:
                command.extend([option, str(value)])
    command.extend(overrides[1:] if overrides[:1] == ["--"] else overrides)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", help="Override the stage YAML path.")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    command = build_command(args, overrides)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    environment = dict(os.environ)
    paths = [ROOT, ROOT / "Depth-Anything-3", ROOT / "Depth-Anything-3/src", ROOT / "DiffSynth-Studio"]
    environment["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), environment.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
