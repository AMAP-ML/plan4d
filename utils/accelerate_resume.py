from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence


_RANDOM_STATE_PATTERN = re.compile(r"^random_states_(\d+)\.pkl$")
_MODEL_STATE_PATTERN = re.compile(r"^zero_pp_rank_(\d+)_mp_rank_\d+_model_states\.pt$")
_OPTIM_STATE_PATTERN = re.compile(r"^(?:bf16_|fp16_)?zero_pp_rank_(\d+)_mp_rank_\d+_optim_states\.pt$")
_LARGE_FILE_COMPARE_THRESHOLD = 16 * 1024 * 1024
_RUN_DATE_PATTERN = re.compile(r"^(\d{8})_\d{6}$")


@dataclass(frozen=True)
class AccelerateStateSummary:
    accelerate_state_dir: Path
    random_state_ranks: tuple[int, ...]
    model_state_ranks: tuple[int, ...]
    optim_state_ranks: tuple[int, ...]

    @property
    def has_any_ranked_files(self) -> bool:
        return bool(self.random_state_ranks or self.model_state_ranks or self.optim_state_ranks)


def _extract_rank_ids(paths: Iterable[Path], pattern: re.Pattern[str]) -> tuple[int, ...]:
    ranks: list[int] = []
    for path in paths:
        match = pattern.match(path.name)
        if match is not None:
            ranks.append(int(match.group(1)))
    return tuple(sorted(set(ranks)))


def _format_rank_ranges(ranks: Sequence[int]) -> str:
    if not ranks:
        return "none"

    spans: list[str] = []
    start = ranks[0]
    end = ranks[0]
    for rank in ranks[1:]:
        if rank == end + 1:
            end = rank
            continue
        spans.append(f"{start}-{end}" if start != end else str(start))
        start = end = rank
    spans.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(spans)


def inspect_accelerate_state_dir(accelerate_state_dir: str | Path) -> AccelerateStateSummary:
    accelerate_state_path = Path(accelerate_state_dir)
    if not accelerate_state_path.is_dir():
        return AccelerateStateSummary(
            accelerate_state_dir=accelerate_state_path,
            random_state_ranks=(),
            model_state_ranks=(),
            optim_state_ranks=(),
        )

    top_level_files = tuple(path for path in accelerate_state_path.iterdir() if path.is_file())
    pytorch_model_dir = accelerate_state_path / "pytorch_model"
    model_files = tuple(path for path in pytorch_model_dir.iterdir() if path.is_file()) if pytorch_model_dir.is_dir() else ()

    return AccelerateStateSummary(
        accelerate_state_dir=accelerate_state_path,
        random_state_ranks=_extract_rank_ids(top_level_files, _RANDOM_STATE_PATTERN),
        model_state_ranks=_extract_rank_ids(model_files, _MODEL_STATE_PATTERN),
        optim_state_ranks=_extract_rank_ids(model_files, _OPTIM_STATE_PATTERN),
    )


def _find_related_step_fragments(step_dir: Path) -> list[tuple[Path, AccelerateStateSummary]]:
    save_root = step_dir.parent.parent
    step_name = step_dir.name
    siblings: list[tuple[Path, AccelerateStateSummary]] = []
    run_dir_name = step_dir.parent.name
    run_date_match = _RUN_DATE_PATTERN.match(run_dir_name)
    preferred_run_date = run_date_match.group(1) if run_date_match is not None else None

    if not save_root.is_dir():
        return siblings

    candidate_run_dirs = sorted(path for path in save_root.iterdir() if path.is_dir())
    if preferred_run_date is not None:
        same_date_run_dirs = [
            path
            for path in candidate_run_dirs
            if _RUN_DATE_PATTERN.match(path.name) is not None
            and _RUN_DATE_PATTERN.match(path.name).group(1) == preferred_run_date
        ]
        if same_date_run_dirs:
            candidate_run_dirs = same_date_run_dirs

    for run_dir in candidate_run_dirs:
        candidate_step_dir = run_dir / step_name
        if candidate_step_dir == step_dir:
            continue
        summary = inspect_accelerate_state_dir(candidate_step_dir / "accelerate_state")
        if summary.has_any_ranked_files:
            siblings.append((candidate_step_dir, summary))
    return siblings


def _missing_rank_ranges(ranks: Sequence[int], expected_processes: int) -> str:
    expected = tuple(rank for rank in range(expected_processes) if rank not in set(ranks))
    return _format_rank_ranges(expected)


def _build_incomplete_resume_message(
    resume_dir: Path,
    expected_processes: int,
    summary: AccelerateStateSummary,
    siblings: Sequence[tuple[Path, AccelerateStateSummary]],
) -> str:
    lines = [
        f"Incomplete accelerate_state under {resume_dir}.",
        (
            f"Expected {expected_processes} ranks, but found random states for "
            f"{len(summary.random_state_ranks)} ranks ({_format_rank_ranges(summary.random_state_ranks)})."
        ),
    ]

    if summary.model_state_ranks:
        lines.append(
            "Found DeepSpeed model shards for "
            f"{len(summary.model_state_ranks)} ranks ({_format_rank_ranges(summary.model_state_ranks)})."
        )
    if summary.optim_state_ranks:
        lines.append(
            "Found DeepSpeed optimizer shards for "
            f"{len(summary.optim_state_ranks)} ranks ({_format_rank_ranges(summary.optim_state_ranks)})."
        )

    lines.append(
        "Missing rank ranges: "
        f"{_missing_rank_ranges(summary.random_state_ranks, expected_processes)}."
    )
    lines.append(
        "This usually means different nodes wrote different timestamped run directories, "
        "so --resume_checkpoint_path points at only one checkpoint fragment."
    )

    if siblings:
        lines.append("Sibling step directories with additional shards:")
        for sibling_dir, sibling_summary in siblings:
            lines.append(
                f"- {sibling_dir}: random/model/optim="
                f"{_format_rank_ranges(sibling_summary.random_state_ranks)} / "
                f"{_format_rank_ranges(sibling_summary.model_state_ranks)} / "
                f"{_format_rank_ranges(sibling_summary.optim_state_ranks)}"
            )

    stitched_model_candidates = sorted(resume_dir.glob("stitched_model_step_*.pth"))
    if stitched_model_candidates:
        lines.append(
            "To unblock quickly, you can resume weights only with "
            f"--init_stitched_model_path {stitched_model_candidates[-1]}."
        )
    lines.append(
        "To resume optimizer/DeepSpeed state exactly, merge all step fragments into one complete directory first."
    )
    return "\n".join(lines)


def validate_accelerate_resume_checkpoint(
    resume_dir: str | Path,
    *,
    expected_processes: int,
) -> None:
    if expected_processes <= 1:
        return

    resume_path = Path(resume_dir)
    accelerate_state_dir = resume_path / "accelerate_state"
    summary = inspect_accelerate_state_dir(accelerate_state_dir)
    if not summary.has_any_ranked_files:
        return

    incomplete_random_states = len(summary.random_state_ranks) != expected_processes
    incomplete_model_states = bool(summary.model_state_ranks) and len(summary.model_state_ranks) != expected_processes
    incomplete_optim_states = bool(summary.optim_state_ranks) and len(summary.optim_state_ranks) != expected_processes

    if incomplete_random_states or incomplete_model_states or incomplete_optim_states:
        siblings = _find_related_step_fragments(resume_path)
        raise RuntimeError(
            _build_incomplete_resume_message(
                resume_path,
                expected_processes=expected_processes,
                summary=summary,
                siblings=siblings,
            )
        )


def _files_match(source: Path, destination: Path) -> bool:
    if source.stat().st_size != destination.stat().st_size:
        return False
    if source.stat().st_size > _LARGE_FILE_COMPARE_THRESHOLD:
        return True
    return source.read_bytes() == destination.read_bytes()


def _copy_tree_once(source_dir: Path, destination_dir: Path) -> None:
    for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_dir)
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            if not _files_match(source_path, destination_path):
                raise RuntimeError(
                    f"Conflicting checkpoint fragment file for {relative_path}: "
                    f"{source_path} vs {destination_path}"
                )
            continue
        shutil.copy2(source_path, destination_path)


def merge_accelerate_step_fragments(
    source_step_dirs: Sequence[str | Path],
    output_step_dir: str | Path,
    *,
    expected_processes: int | None = None,
) -> Path:
    if not source_step_dirs:
        raise ValueError("source_step_dirs must contain at least one checkpoint step directory.")

    normalized_sources = [Path(path) for path in source_step_dirs]
    step_names = {path.name for path in normalized_sources}
    if len(step_names) != 1:
        raise ValueError(f"All source step directories must share the same step name, got: {sorted(step_names)}")

    output_path = Path(output_step_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for source_dir in normalized_sources:
        _copy_tree_once(source_dir, output_path)

    if expected_processes is not None:
        validate_accelerate_resume_checkpoint(output_path, expected_processes=expected_processes)

    return output_path
