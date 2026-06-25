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

from detect_board_state import (
    BoardState,
    Stone,
    board_delta_to_jsonable,
    board_state_from_image,
    delta_between_board_states,
    transform_board_state,
)
from lerobot.datasets import LeRobotDataset


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_CROP_LEFT_RIGHT_RATIO = 0.10
DONE_ENV_STATE_NAMES = ["done"]
DEFAULT_DONE_STABLE_FRAMES = 5
DEFAULT_BOARD_CONFIG = Path("examples/go_board/dashboard_config.json")


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


def _done_from_delta(delta: dict[str, Any], target: dict[str, Any] | None = None) -> bool:
    added = delta.get("added") or []
    removed = delta.get("removed") or []
    changed = delta.get("changed") or []
    if len(added) != 1 or removed or changed:
        return False
    expected = target if isinstance(target, dict) and target.get("coord") else added[0]
    stone = added[0]
    return (
        str(stone.get("coord", "")).upper() == str(expected.get("coord", "")).upper()
        and str(stone.get("color", "")).lower() == str(expected.get("color", "")).lower()
    )


def _metadata_done(metadata: dict[str, Any]) -> bool:
    board = metadata.get("board") or {}
    task_state = board.get("task_state")
    if isinstance(task_state, dict) and "done" in task_state:
        return bool(task_state["done"])
    return _done_from_delta(board.get("delta") or {}, board.get("target"))


def _done_env_state_for_sample(
    sample: dict[str, Any],
    fallback_done: bool,
    detected_done: bool | None = None,
) -> np.ndarray:
    if detected_done is not None:
        done = detected_done
    elif "done" in sample:
        done = bool(sample["done"])
    else:
        task_state = sample.get("task_state")
        done = bool(task_state.get("done")) if isinstance(task_state, dict) and "done" in task_state else fallback_done
    return np.array([1.0 if done else 0.0], dtype=np.float32)


def _stone_from_json(raw: dict[str, Any]) -> Stone:
    return Stone(
        coord=str(raw["coord"]).upper(),
        row=int(raw["row"]),
        col=int(raw["col"]),
        color=str(raw["color"]).lower(),
        confidence=float(raw.get("confidence", 1.0)),
        board_xy=tuple(raw.get("board_xy") or (0.0, 0.0)),
        image_xy=tuple(raw.get("image_xy") or (0.0, 0.0)),
    )


def _board_state_from_json(raw: dict[str, Any] | None) -> BoardState | None:
    if not isinstance(raw, dict):
        return None
    stones = [_stone_from_json(stone) for stone in raw.get("stones") or []]
    occupied = raw.get("occupied") if isinstance(raw.get("occupied"), dict) else {stone.coord: stone.color for stone in stones}
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {
        "black": sum(stone.color == "black" for stone in stones),
        "white": sum(stone.color == "white" for stone in stones),
        "total": len(stones),
    }
    return BoardState(
        board_size=int(raw.get("board_size", 19)),
        corners_tl_tr_br_bl=raw.get("corners_tl_tr_br_bl") or [],
        stones=stones,
        occupied={str(coord).upper(): str(color).lower() for coord, color in occupied.items()},
        summary={str(key): int(value) for key, value in summary.items()},
    )


def _board_detection_kwargs(metadata: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    board_meta = metadata.get("board") or {}
    board_cfg: dict[str, Any] = {}
    if config_path.is_file():
        try:
            board_cfg = (json.loads(config_path.read_text(encoding="utf-8")).get("board") or {})
        except json.JSONDecodeError:
            board_cfg = {}
    corners = board_meta.get("corners_tl_tr_br_bl") or board_cfg.get("corners_tl_tr_br_bl")
    if corners is None:
        return None
    return {
        "size": int(board_meta.get("size") or board_cfg.get("size", 19)),
        "corners": np.array(corners, dtype=np.float32),
        "board_pixels": int(board_cfg.get("board_pixels", 1000)),
        "sample_radius_ratio": float(board_cfg.get("sample_radius_ratio", 0.34)),
        "black_l_threshold": float(board_cfg.get("black_l_threshold", 80.0)),
        "white_l_threshold": float(board_cfg.get("white_l_threshold", 165.0)),
        "white_s_threshold": float(board_cfg.get("white_s_threshold", 75.0)),
        "stone_min_radius_ratio": float(board_cfg.get("stone_min_radius_ratio", 0.2)),
        "stone_max_radius_ratio": float(board_cfg.get("stone_max_radius_ratio", 0.48)),
        "stone_min_circularity": float(board_cfg.get("stone_min_circularity", 0.45)),
        "stone_max_snap_distance_ratio": float(board_cfg.get("stone_max_snap_distance_ratio", 0.52)),
        "black_grid_min_edge_score": float(board_cfg.get("black_grid_min_edge_score", 0.18)),
        "overlay_fisheye_k": float(board_meta.get("overlay_fisheye_k") or board_cfg.get("overlay_fisheye_k", 0.0)),
        "camera_to_robot_rotation_degrees": int(
            board_meta.get("camera_to_robot_rotation_degrees")
            if board_meta.get("camera_to_robot_rotation_degrees") is not None
            else board_cfg.get("camera_to_robot_rotation_degrees", 0)
        ),
    }


def _detect_board_state_from_frame(frame: np.ndarray, kwargs: dict[str, Any]) -> BoardState:
    state = board_state_from_image(
        frame,
        corners=kwargs["corners"],
        size=kwargs["size"],
        board_pixels=kwargs["board_pixels"],
        sample_radius_ratio=kwargs["sample_radius_ratio"],
        black_l_threshold=kwargs["black_l_threshold"],
        white_l_threshold=kwargs["white_l_threshold"],
        white_s_threshold=kwargs["white_s_threshold"],
        stone_min_radius_ratio=kwargs["stone_min_radius_ratio"],
        stone_max_radius_ratio=kwargs["stone_max_radius_ratio"],
        stone_min_circularity=kwargs["stone_min_circularity"],
        stone_max_snap_distance_ratio=kwargs["stone_max_snap_distance_ratio"],
        black_grid_min_edge_score=kwargs["black_grid_min_edge_score"],
        overlay_fisheye_k=kwargs["overlay_fisheye_k"],
    )
    return transform_board_state(state, int(kwargs.get("camera_to_robot_rotation_degrees", 0)))


def _recording_frame_paths(recording_dir: Path, sample: dict[str, Any], camera: str) -> Path:
    relative = Path(sample.get("cameras", {}).get(camera, ""))
    if camera == "overhead":
        processed = recording_dir / "overhead_processed" / relative.name
        if processed.is_file():
            return processed
    return recording_dir / relative


def _raw_recording_frame_path(recording_dir: Path, sample: dict[str, Any], camera: str) -> Path:
    return recording_dir / Path(sample.get("cameras", {}).get(camera, ""))


def _latched_done_by_sample(
    recording_dir: Path,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    target: dict[str, Any] | None,
    config_path: Path,
    stable_frames: int,
) -> dict[int, bool]:
    if target is None:
        return {}
    kwargs = _board_detection_kwargs(metadata, config_path)
    if kwargs is None:
        return {}
    board_meta = metadata.get("board") or {}
    board_camera = str(board_meta.get("camera") or "overhead")
    baseline = _board_state_from_json(board_meta.get("baseline"))
    stable_required = max(1, int(stable_frames))
    stable_count = 0
    latched = False
    done_by_index: dict[int, bool] = {}
    for fallback_index, sample in enumerate(samples):
        sample_index = int(sample.get("index", fallback_index))
        frame_path = _raw_recording_frame_path(recording_dir, sample, board_camera)
        candidate_done = False
        try:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read {frame_path}")
            current = _detect_board_state_from_frame(image, kwargs)
            if baseline is None:
                baseline = current
            else:
                delta = board_delta_to_jsonable(delta_between_board_states(baseline, current))
                candidate_done = _done_from_delta(delta, target)
        except Exception:
            candidate_done = False
        if not latched:
            stable_count = stable_count + 1 if candidate_done else 0
            latched = stable_count >= stable_required
        done_by_index[sample_index] = latched
    return done_by_index


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
    include_done_env_state: bool = True,
    done_stable_frames: int = DEFAULT_DONE_STABLE_FRAMES,
    config_path: Path = DEFAULT_BOARD_CONFIG,
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
    if include_done_env_state:
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (len(DONE_ENV_STATE_NAMES),),
            "names": DONE_ENV_STATE_NAMES,
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
            episode_done = _metadata_done(metadata)
            episode_frames = 0
            episode_teleop_frames = 0
            episode_pre_teleop_frames = 0
            with (recording_dir / "telemetry.jsonl").open("r", encoding="utf-8") as telemetry_file:
                samples = [json.loads(line) for line in telemetry_file if line.strip()]
            done_by_sample = (
                _latched_done_by_sample(
                    recording_dir=recording_dir,
                    samples=samples,
                    metadata=metadata,
                    target=target,
                    config_path=config_path,
                    stable_frames=done_stable_frames,
                )
                if include_done_env_state
                else {}
            )

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
                if include_done_env_state:
                    detected_done = done_by_sample.get(int(sample.get("index", sample_index)))
                    frame["observation.environment_state"] = _done_env_state_for_sample(
                        sample,
                        episode_done,
                        detected_done=detected_done,
                    )
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
        "include_done_env_state": include_done_env_state,
        "done_env_state_names": DONE_ENV_STATE_NAMES if include_done_env_state else [],
        "done_stable_frames": done_stable_frames,
        "board_config": str(config_path),
        "fps": fps,
    }
    (dataset_root / "go_conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("examples/go_board/recordings"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/go_board_act_v1"))
    parser.add_argument("--repo-id", default="callum/go_board_act_v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_BOARD_CONFIG)
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
    parser.add_argument(
        "--no-done-env-state",
        dest="include_done_env_state",
        action="store_false",
        default=True,
        help="Do not include the done value as observation.environment_state.",
    )
    parser.add_argument(
        "--done-stable-frames",
        type=int,
        default=DEFAULT_DONE_STABLE_FRAMES,
        help="Latch done only after the target-only board condition holds for this many consecutive frames.",
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
        include_done_env_state=args.include_done_env_state,
        done_stable_frames=args.done_stable_frames,
        config_path=args.config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
