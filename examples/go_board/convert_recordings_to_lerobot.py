#!/usr/bin/env python
"""Convert Go dashboard recordings into a LeRobot dataset for ACT training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_CROP_LEFT_RIGHT_RATIO = 0.10


def _joint_vector(joints: list[dict[str, Any]], names: list[str]) -> np.ndarray | None:
    by_name = {str(joint.get("name")): float(joint.get("value")) for joint in joints if "name" in joint}
    if any(name not in by_name for name in names):
        return None
    return np.array([by_name[name] for name in names], dtype=np.float32)


def _crop_left_right(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    if crop_ratio <= 0:
        return image
    if crop_ratio >= 0.5:
        raise ValueError("crop_ratio must be less than 0.5")
    width = image.shape[1]
    left = int(width * crop_ratio)
    right = int(width * (1.0 - crop_ratio))
    return image[:, left:right]


def _read_image(path: Path, size: int, crop_left_right_ratio: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _crop_left_right(image, crop_left_right_ratio)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _single_added_delta(metadata: dict[str, Any]) -> dict[str, Any] | None:
    delta = ((metadata.get("board") or {}).get("delta") or {})
    added = delta.get("added") or []
    removed = delta.get("removed") or []
    changed = delta.get("changed") or []
    if len(added) == 1 and not removed and not changed:
        return added[0]
    return None


def _recording_frame_paths(recording_dir: Path, sample: dict[str, Any], camera: str) -> Path:
    relative = Path(sample.get("cameras", {}).get(camera, ""))
    if camera == "overhead":
        processed = recording_dir / "overhead_processed" / relative.name
        if processed.is_file():
            return processed
    return recording_dir / relative


def _sample_joint_vector(sample: dict[str, Any], telemetry_key: str) -> np.ndarray | None:
    telemetry = sample.get("telemetry") or {}
    return _joint_vector(telemetry.get(telemetry_key) or [], JOINT_NAMES)


def _future_follower_action(samples: list[dict[str, Any]], sample_index: int, offset: int) -> np.ndarray | None:
    future_index = sample_index + offset
    if future_index >= len(samples):
        return None
    return _sample_joint_vector(samples[future_index], "joints")


def iter_valid_recordings(recordings_dir: Path) -> list[Path]:
    recordings = []
    for metadata_path in sorted(recordings_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _single_added_delta(metadata) is not None:
            recordings.append(metadata_path.parent)
    return recordings


def convert_go_recordings(
    recordings_dir: Path,
    dataset_root: Path,
    repo_id: str,
    image_size: int = 224,
    fps: int = 10,
    force: bool = False,
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    pre_teleop_action: str = "skip",
    future_action_offset: int = 1,
    crop_left_right_ratio: float = DEFAULT_CROP_LEFT_RIGHT_RATIO,
) -> dict[str, Any]:
    if pre_teleop_action not in {"skip", "future_follower"}:
        raise ValueError("pre_teleop_action must be 'skip' or 'future_follower'.")
    if future_action_offset < 1:
        raise ValueError("future_action_offset must be at least 1.")
    if crop_left_right_ratio < 0 or crop_left_right_ratio >= 0.5:
        raise ValueError("crop_left_right_ratio must be >= 0 and < 0.5.")

    if dataset_root.exists():
        if not force:
            raise FileExistsError(f"{dataset_root} already exists. Use --force to replace it.")
        shutil.rmtree(dataset_root)

    features = {
        "observation.images.overhead": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="so101",
        use_videos=False,
        image_writer_threads=4,
    )

    recordings = iter_valid_recordings(recordings_dir)
    if max_episodes is not None:
        recordings = recordings[:max_episodes]

    converted = 0
    skipped = []
    frames_total = 0
    teleop_frames_total = 0
    pre_teleop_frames_total = 0
    try:
        for recording_dir in tqdm(recordings, desc="Converting recordings"):
            metadata = json.loads((recording_dir / "metadata.json").read_text(encoding="utf-8"))
            target = _single_added_delta(metadata)
            task = f"place {target['color']} stone at {target['coord']}" if target else "place stone"
            episode_frames = 0
            episode_teleop_frames = 0
            episode_pre_teleop_frames = 0
            with (recording_dir / "telemetry.jsonl").open("r", encoding="utf-8") as telemetry_file:
                samples = [json.loads(line) for line in telemetry_file if line.strip()]

            for sample_index, sample in enumerate(samples):
                telemetry = sample.get("telemetry") or {}
                teleop_enabled = bool(telemetry.get("teleop_enabled", False))
                state = _sample_joint_vector(sample, "joints")
                if teleop_enabled:
                    action = _sample_joint_vector(sample, "leader_joints")
                elif pre_teleop_action == "future_follower":
                    action = _future_follower_action(samples, sample_index, future_action_offset)
                else:
                    continue
                if state is None or action is None:
                    continue

                overhead_path = _recording_frame_paths(recording_dir, sample, "overhead")
                wrist_path = _recording_frame_paths(recording_dir, sample, "wrist")
                if not overhead_path.is_file() or not wrist_path.is_file():
                    continue

                frame = {
                    "observation.images.overhead": _read_image(
                        overhead_path, image_size, crop_left_right_ratio
                    ),
                    "observation.images.wrist": _read_image(wrist_path, image_size, crop_left_right_ratio),
                    "observation.state": state,
                    "action": action,
                    "task": task,
                }
                dataset.add_frame(frame)
                episode_frames += 1
                if teleop_enabled:
                    episode_teleop_frames += 1
                else:
                    episode_pre_teleop_frames += 1
                if max_frames_per_episode is not None and episode_frames >= max_frames_per_episode:
                    break

            if episode_frames == 0:
                dataset.clear_episode_buffer()
                skipped.append({"recording": recording_dir.name, "reason": "no trainable teleop frames"})
                continue
            dataset.save_episode()
            converted += 1
            frames_total += episode_frames
            teleop_frames_total += episode_teleop_frames
            pre_teleop_frames_total += episode_pre_teleop_frames
    finally:
        dataset.finalize()

    summary = {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "recordings_seen": len(recordings),
        "episodes": converted,
        "frames": frames_total,
        "teleop_frames": teleop_frames_total,
        "pre_teleop_frames": pre_teleop_frames_total,
        "pre_teleop_action": pre_teleop_action,
        "future_action_offset": future_action_offset,
        "skipped": skipped,
        "image_size": image_size,
        "crop_left_right_ratio": crop_left_right_ratio,
        "fps": fps,
    }
    (dataset_root / "go_conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("examples/go_board/recordings"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/go_board_act_v1"))
    parser.add_argument("--repo-id", default="callum/go_board_act_v1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-left-right-ratio", type=float, default=DEFAULT_CROP_LEFT_RIGHT_RATIO)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument(
        "--pre-teleop-action",
        choices=["skip", "future_follower"],
        default="skip",
        help="How to label frames where teleop_enabled is false. 'skip' preserves the old behavior. "
        "'future_follower' uses follower joints from a future frame as the action label.",
    )
    parser.add_argument(
        "--future-action-offset",
        type=int,
        default=1,
        help="Frame offset for --pre-teleop-action=future_follower.",
    )
    args = parser.parse_args()

    summary = convert_go_recordings(
        recordings_dir=args.recordings_dir,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        image_size=args.image_size,
        fps=args.fps,
        force=args.force,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        pre_teleop_action=args.pre_teleop_action,
        future_action_offset=args.future_action_offset,
        crop_left_right_ratio=args.crop_left_right_ratio,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
