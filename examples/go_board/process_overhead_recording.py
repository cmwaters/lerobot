#!/usr/bin/env python
"""Render target-stone overlays onto saved overhead recording frames."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_board_state import curved_grid_points_from_corners, inverse_transform_row_col


STATUS_FILENAME = "status.json"
PROCESSED_DIR_NAME = "overhead_processed"


def overhead_processed_status(recording_dir: Path) -> str:
    """Return a compact status for dashboard summaries."""
    processed_dir = recording_dir / PROCESSED_DIR_NAME
    status_path = processed_dir / STATUS_FILENAME
    if status_path.is_file():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "error"
        return str(data.get("status", "missing"))
    if any(processed_dir.glob("*.jpg")):
        return "ready"
    return "missing"


def _write_status(processed_dir: Path, status: str, **extra: Any) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at": time.time(), **extra}
    (processed_dir / STATUS_FILENAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _single_added_target(metadata: dict[str, Any]) -> dict[str, Any]:
    delta = ((metadata.get("board") or {}).get("delta") or {})
    added = delta.get("added") or []
    removed = delta.get("removed") or []
    changed = delta.get("changed") or []
    if len(added) != 1 or removed or changed:
        raise ValueError("Recording must have exactly one added stone and no removed/changed stones.")
    target = added[0]
    if "row" not in target or "col" not in target or "color" not in target:
        raise ValueError("Added stone is missing row, col, or color.")
    return target


def _target_metadata(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "coord": str(target["coord"]).upper(),
        "row": int(target["row"]),
        "col": int(target["col"]),
        "color": str(target["color"]).lower(),
    }


def _load_done_by_frame(recording_dir: Path) -> dict[int, bool]:
    telemetry_path = recording_dir / "telemetry.jsonl"
    if not telemetry_path.is_file():
        return {}
    done_by_frame: dict[int, bool] = {}
    for fallback_index, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        index = int(sample.get("index", fallback_index))
        task_state = sample.get("task_state")
        if isinstance(task_state, dict) and "done" in task_state:
            done_by_frame[index] = bool(task_state["done"])
        elif "done" in sample:
            done_by_frame[index] = bool(sample["done"])
    return done_by_frame


def _board_config(metadata: dict[str, Any]) -> tuple[int, np.ndarray, float, int, str]:
    board = metadata.get("board") or {}
    corners = board.get("corners_tl_tr_br_bl")
    if not corners or len(corners) != 4:
        raise ValueError("metadata.json is missing board.corners_tl_tr_br_bl.")
    board_size = int(board.get("size") or 19)
    curve_k = float(board.get("overlay_fisheye_k") or 0.0)
    rotation = int(board.get("camera_to_robot_rotation_degrees") or 0)
    camera_name = str(board.get("camera") or "overhead")
    return board_size, np.array(corners, dtype=np.float32), curve_k, rotation, camera_name


def _marker_radius(points: list[list[tuple[float, float]]], row: int, col: int) -> int:
    distances: list[float] = []
    if col > 0:
        distances.append(float(np.hypot(points[row][col][0] - points[row][col - 1][0], points[row][col][1] - points[row][col - 1][1])))
    if col + 1 < len(points[row]):
        distances.append(float(np.hypot(points[row][col][0] - points[row][col + 1][0], points[row][col][1] - points[row][col + 1][1])))
    if row > 0:
        distances.append(float(np.hypot(points[row][col][0] - points[row - 1][col][0], points[row][col][1] - points[row - 1][col][1])))
    if row + 1 < len(points):
        distances.append(float(np.hypot(points[row][col][0] - points[row + 1][col][0], points[row][col][1] - points[row + 1][col][1])))
    if not distances:
        return 16
    return int(max(10, min(24, round(min(distances) * 0.42))))


def _draw_target_marker(frame: np.ndarray, center: tuple[float, float], radius: int, color: str) -> np.ndarray:
    output = frame.copy()
    point = tuple(np.round(center).astype(int))
    normalized = color.lower()
    if normalized == "black":
        fill = (255, 80, 20)
        outline = (255, 255, 255)
    else:
        fill = (20, 20, 245)
        outline = (255, 255, 255)
    cv2.circle(output, point, radius + 4, outline, 3, cv2.LINE_AA)
    cv2.circle(output, point, radius, fill, -1, cv2.LINE_AA)
    return output


def process_recording_overhead(recording_dir: Path, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Create target-overlay overhead frames for one recording.

    The raw frames under ``frames/<board camera>/`` are only read. Processed
    copies are written to ``overhead_processed/``.
    """
    recording_dir = Path(recording_dir)
    metadata_path = recording_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = _single_added_target(metadata)
    target_meta = _target_metadata(target)
    done_by_frame = _load_done_by_frame(recording_dir)
    board_size, corners, curve_k, rotation, camera_name = _board_config(metadata)
    raw_dir = recording_dir / "frames" / camera_name
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    raw_frames = sorted(raw_dir.glob("*.jpg"))
    if not raw_frames:
        raise ValueError(f"No overhead frames found in {raw_dir}.")

    processed_dir = recording_dir / PROCESSED_DIR_NAME
    existing_frames = sorted(processed_dir.glob("*.jpg")) if processed_dir.is_dir() else []
    if existing_frames and len(existing_frames) == len(raw_frames) and not force:
        return {
            "ok": True,
            "status": "ready",
            "recording": recording_dir.name,
            "processed_dir": str(processed_dir),
            "frames": len(existing_frames),
            "skipped": True,
        }

    camera_row, camera_col = inverse_transform_row_col(
        int(target["row"]),
        int(target["col"]),
        board_size,
        rotation,
    )

    if dry_run:
        return {
            "ok": True,
            "status": "dry_run",
            "recording": recording_dir.name,
            "frames": len(raw_frames),
            "target": target_meta,
            "camera_grid": {"row": camera_row, "col": camera_col},
            "processed_dir": str(processed_dir),
        }

    if processed_dir.exists() and force:
        for frame_path in processed_dir.glob("*.jpg"):
            frame_path.unlink()
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        processed_dir,
        "processing",
        frames_total=len(raw_frames),
        target=target_meta,
        done_aware=True,
    )

    try:
        written = 0
        marked = 0
        unmarked_after_done = 0
        for raw_frame in raw_frames:
            frame = cv2.imread(str(raw_frame), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read frame {raw_frame}.")
            frame_index = int(raw_frame.stem)
            if done_by_frame.get(frame_index, False):
                output = frame
                unmarked_after_done += 1
            else:
                points = curved_grid_points_from_corners(frame.shape, corners, board_size, curve_k)
                center = points[camera_row][camera_col]
                radius = _marker_radius(points, camera_row, camera_col)
                output = _draw_target_marker(frame, center, radius, str(target_meta["color"]))
                marked += 1
            output_path = processed_dir / raw_frame.name
            if not cv2.imwrite(str(output_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                raise ValueError(f"Could not write processed frame {output_path}.")
            written += 1
        _write_status(
            processed_dir,
            "ready",
            frames_total=len(raw_frames),
            frames_written=written,
            frames_marked=marked,
            frames_unmarked_after_done=unmarked_after_done,
            target=target_meta,
            done_aware=True,
        )
        return {
            "ok": True,
            "status": "ready",
            "recording": recording_dir.name,
            "processed_dir": str(processed_dir),
            "frames": written,
            "frames_marked": marked,
            "frames_unmarked_after_done": unmarked_after_done,
            "target": target_meta,
        }
    except Exception as exc:
        _write_status(processed_dir, "error", error=str(exc), target=target_meta, done_aware=True)
        raise


def _recording_dirs(path: Path, process_all: bool) -> list[Path]:
    if process_all:
        if not path.is_dir():
            raise FileNotFoundError(path)
        return sorted(candidate.parent for candidate in path.glob("*/metadata.json"))
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Recording directory, or recordings root with --all.")
    parser.add_argument("--all", action="store_true", help="Process every recording under PATH.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing processed frames.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing frames.")
    args = parser.parse_args()

    results = []
    for recording_dir in _recording_dirs(args.path, args.all):
        try:
            results.append(process_recording_overhead(recording_dir, force=args.force, dry_run=args.dry_run))
        except Exception as exc:  # noqa: BLE001 - CLI should continue through a batch.
            results.append({"ok": False, "recording": recording_dir.name, "error": str(exc)})

    print(json.dumps({"results": results}, indent=2))
    if any(not result.get("ok") for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
