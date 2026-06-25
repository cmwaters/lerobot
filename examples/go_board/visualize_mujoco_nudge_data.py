#!/usr/bin/env python
"""Create preview videos from MuJoCo nudge synthetic recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("outputs/go_board_mujoco_nudge_recordings")
DEFAULT_OUTPUT = Path("outputs/go_board_mujoco_nudge_preview.mp4")


def _read_rows(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "telemetry.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_frame(episode_dir: Path, relative: str, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(episode_dir / relative), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _draw_text(image: np.ndarray, lines: list[str]) -> None:
    pad = 10
    line_h = 22
    height = pad * 2 + line_h * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (20, 24, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (pad, pad + 15 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 248, 245),
            1,
            cv2.LINE_AA,
        )


def _episode_dirs(root: Path, max_episodes: int | None) -> list[Path]:
    episodes = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "metadata.json").is_file()]
    return episodes if max_episodes is None else episodes[:max_episodes]


def create_preview_video(
    input_dir: Path,
    output: Path,
    max_episodes: int | None = 8,
    fps: float = 5.0,
    panel_width: int = 480,
    panel_height: int = 360,
) -> dict[str, Any]:
    episodes = _episode_dirs(input_dir, max_episodes)
    if not episodes:
        raise FileNotFoundError(f"No generated nudge episodes found under {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (panel_width * 2, panel_height)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    written = 0
    try:
        for episode_dir in episodes:
            metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
            target = ((metadata.get("board") or {}).get("target") or {})
            synthetic = metadata.get("synthetic") or {}
            rows = _read_rows(episode_dir)
            for row in rows:
                cameras = row.get("cameras") or {}
                overhead = _read_frame(episode_dir, str(cameras.get("overhead", "")), (panel_width, panel_height))
                wrist = _read_frame(episode_dir, str(cameras.get("wrist", "")), (panel_width, panel_height))
                telemetry = (row.get("telemetry") or {}).get("synthetic") or {}
                action = telemetry.get("action") or {}
                lines = [
                    f"{episode_dir.name}  frame {row.get('index', 0)}",
                    f"target {target.get('color', '?')} {target.get('coord', '?')}  done={telemetry.get('done')}",
                    f"error={float(telemetry.get('target_error_m', 0.0)) * 1000:.1f}mm  final_success={synthetic.get('success')}",
                    f"push=({float(action.get('push_dx_m', 0.0)) * 1000:.1f}, {float(action.get('push_dy_m', 0.0)) * 1000:.1f})mm",
                ]
                combined = np.concatenate([overhead, wrist], axis=1)
                _draw_text(combined, lines)
                writer.write(combined)
                written += 1
    finally:
        writer.release()

    return {
        "ok": True,
        "input_dir": str(input_dir),
        "output": str(output),
        "episodes": len(episodes),
        "frames": written,
        "fps": fps,
        "frame_size": frame_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = create_preview_video(
        input_dir=args.input_dir,
        output=args.output,
        max_episodes=args.max_episodes,
        fps=args.fps,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
