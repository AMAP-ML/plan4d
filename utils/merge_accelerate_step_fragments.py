#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.accelerate_resume import merge_accelerate_step_fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge fragmented multi-node Accelerate/DeepSpeed step checkpoints into one complete step directory.",
    )
    parser.add_argument(
        "--source_step_dir",
        action="append",
        required=True,
        help="Step directory to merge, e.g. /path/to/run_timestamp/step_300. Repeat for each fragment.",
    )
    parser.add_argument(
        "--output_step_dir",
        required=True,
        help="Destination step directory, e.g. /path/to/merged/step_300.",
    )
    parser.add_argument(
        "--expected_processes",
        type=int,
        default=None,
        help="If set, validate that the merged checkpoint contains exactly this many ranks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = merge_accelerate_step_fragments(
        args.source_step_dir,
        args.output_step_dir,
        expected_processes=args.expected_processes,
    )
    print(f"Merged checkpoint step written to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
