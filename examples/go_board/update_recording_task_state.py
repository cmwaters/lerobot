#!/usr/bin/env python
"""Backfill latched done task state into raw Go-board recordings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2

from convert_recordings_to_lerobot import (
    DEFAULT_BOARD_CONFIG,
    DEFAULT_DONE_STABLE_FRAMES,
    _board_detection_kwargs,
    _board_state_from_json,
    _detect_board_state_from_frame,
    _done_from_delta,
    _single_added_delta,
)
from detect_board_state import board_delta_to_jsonable, delta_between_board_states


def _target_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    target = _single_added_delta(metadata)
    if target is not None:
        return {
            "coord": str(target["coord"]).upper(),
            "row": int(target["row"]),
            "col": int(target["col"]),
            "color": str(target["color"]).lower(),
        }
    board_target = (metadata.get("board") or {}).get("target")
    if isinstance(board_target, dict) and board_target.get("coord"):
        return {
            "coord": str(board_target["coord"]).upper(),
            "row": int(board_target["row"]),
            "col": int(board_target["col"]),
            "color": str(board_target["color"]).lower(),
        }
    return None


def _read_samples(recording_dir: Path) -> list[dict[str, Any]]:
    telemetry_path = recording_dir / "telemetry.jsonl"
    if not telemetry_path.is_file():
        return []
    samples: list[dict[str, Any]] = []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
    return samples


def _write_samples(recording_dir: Path, samples: list[dict[str, Any]]) -> None:
    telemetry_path = recording_dir / "telemetry.jsonl"
    tmp_path = telemetry_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text(
        "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in samples),
        encoding="utf-8",
    )
    tmp_path.replace(telemetry_path)


def _frame_path(recording_dir: Path, sample: dict[str, Any], camera: str) -> Path:
    relative = sample.get("cameras", {}).get(camera)
    return recording_dir / Path(str(relative or ""))


def _sample_task_state(
    *,
    sample_index: int,
    target: dict[str, Any] | None,
    done: bool,
    candidate_done: bool,
    stable_count: int,
    stable_required: int,
    delta_counts: dict[str, int] | None,
    error: str = "",
) -> dict[str, Any]:
    if done:
        reason = "target occupied with correct colour and no other board changes"
    elif candidate_done:
        reason = f"target condition stable for {stable_count}/{stable_required} frames"
    elif error:
        reason = error
    else:
        reason = "target is not the only board change"
    return {
        "index": sample_index,
        "done": done,
        "candidate_done": candidate_done,
        "stable_count": stable_count,
        "stable_required": stable_required,
        "target": target,
        "reason": reason,
        "delta": delta_counts,
    }


def update_recording(
    recording_dir: Path,
    *,
    config_path: Path,
    stable_frames: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    metadata_path = recording_dir / "metadata.json"
    if not metadata_path.is_file():
        return {"recording": recording_dir.name, "updated": False, "reason": "missing metadata"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = _read_samples(recording_dir)
    target = _target_from_metadata(metadata)
    kwargs = _board_detection_kwargs(metadata, config_path)
    board = metadata.setdefault("board", {})
    board_camera = str(board.get("camera") or "overhead")
    if target is not None:
        board["target"] = target
    baseline = _board_state_from_json(board.get("baseline"))
    stable_required = max(1, int(stable_frames))
    stable_count = 0
    latched = False
    completed_index: int | None = None
    completed_elapsed_s: float | None = None
    detection_errors = 0

    for fallback_index, sample in enumerate(samples):
        sample_index = int(sample.get("index", fallback_index))
        candidate_done = False
        delta_counts: dict[str, int] | None = None
        error = ""
        if target is None:
            error = "no target available"
        elif kwargs is None:
            error = "board detector unavailable"
        else:
            try:
                image = cv2.imread(str(_frame_path(recording_dir, sample, board_camera)), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("could not read raw board frame")
                current = _detect_board_state_from_frame(image, kwargs)
                if baseline is None:
                    baseline = current
                    error = "baseline captured"
                else:
                    delta = board_delta_to_jsonable(delta_between_board_states(baseline, current))
                    candidate_done = _done_from_delta(delta, target)
                    delta_counts = {
                        "added": len(delta.get("added") or []),
                        "removed": len(delta.get("removed") or []),
                        "changed": len(delta.get("changed") or []),
                    }
            except Exception as exc:  # noqa: BLE001 - keep backfill moving across recordings.
                detection_errors += 1
                error = f"board: {exc}"

        if not latched:
            stable_count = stable_count + 1 if candidate_done else 0
            latched = stable_count >= stable_required
            if latched:
                completed_index = sample_index
                completed_elapsed_s = float(sample.get("elapsed_s", 0.0))

        sample["done"] = latched
        sample["task_state"] = _sample_task_state(
            sample_index=sample_index,
            target=target,
            done=latched,
            candidate_done=candidate_done,
            stable_count=stable_count,
            stable_required=stable_required,
            delta_counts=delta_counts,
            error=error,
        )
        sample.pop("phase_detection", None)

    task_state = {
        "done": latched,
        "target": target,
        "reason": (
            "target occupied with correct colour and no other board changes"
            if latched
            else "target condition never held long enough"
        ),
        "stable_required": stable_required,
        "completed_index": completed_index,
        "completed_elapsed_s": completed_elapsed_s,
        "source": "raw_overhead_frames",
        "detection_errors": detection_errors,
        "updated_at": time.time(),
    }
    board["task_state"] = task_state
    metadata.pop("phase_detection", None)

    if not dry_run:
        tmp_path = metadata_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(metadata_path)
        _write_samples(recording_dir, samples)

    return {
        "recording": recording_dir.name,
        "updated": not dry_run,
        "done": latched,
        "completed_index": completed_index,
        "completed_elapsed_s": completed_elapsed_s,
        "samples": len(samples),
        "detection_errors": detection_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("examples/go_board/recordings"))
    parser.add_argument("--config", type=Path, default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--done-stable-frames", type=int, default=DEFAULT_DONE_STABLE_FRAMES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recording_dirs = sorted(path for path in args.recordings_dir.iterdir() if (path / "metadata.json").is_file())
    if args.limit is not None:
        recording_dirs = recording_dirs[: args.limit]

    results = []
    for index, recording_dir in enumerate(recording_dirs, start=1):
        result = update_recording(
            recording_dir,
            config_path=args.config,
            stable_frames=args.done_stable_frames,
            dry_run=args.dry_run,
        )
        results.append(result)
        status = "done" if result.get("done") else "not done"
        print(
            f"[{index}/{len(recording_dirs)}] {recording_dir.name}: {status}, "
            f"completed_index={result.get('completed_index')}, errors={result.get('detection_errors')}"
        )

    summary = {
        "recordings": len(results),
        "done": sum(1 for result in results if result.get("done")),
        "not_done": sum(1 for result in results if not result.get("done")),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
