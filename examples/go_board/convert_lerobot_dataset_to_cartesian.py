#!/usr/bin/env python
"""Convert an existing Go-board LeRobot dataset from joint vectors to Cartesian pose vectors."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.model.kinematics import RobotKinematics


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CARTESIAN_POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
DEFAULT_SOURCE_ROOT = Path("outputs/datasets/go_board_act_v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets/go_board_cartesian_v1")
DEFAULT_SO101_URDF = Path("examples/go_board/assets/robotstudio_so101/so101_new_calib.urdf")
DEFAULT_TARGET_FRAME_NAME = "gripper_frame_link"
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if singular:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    else:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _cartesian_pose_from_joints(
    joints: np.ndarray,
    kinematics: RobotKinematics,
    *,
    include_gripper: bool,
) -> np.ndarray:
    transform = kinematics.forward_kinematics(joints.astype(np.float64))
    roll, pitch, yaw = _rotation_matrix_to_rpy_deg(transform[:3, :3])
    pose = np.array(
        [transform[0, 3], transform[1, 3], transform[2, 3], roll, pitch, yaw],
        dtype=np.float32,
    )
    if include_gripper:
        pose = np.concatenate([pose, joints[len(JOINT_NAMES) - 1 : len(JOINT_NAMES)].astype(np.float32)])
    return pose


def _fixed_size_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    rows = values.to_pylist()
    return np.asarray(rows, dtype=np.float32)


def _numpy_to_fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _transform_vectors(
    vectors: np.ndarray,
    kinematics: RobotKinematics,
    *,
    include_gripper: bool,
) -> np.ndarray:
    return np.stack(
        [
            _cartesian_pose_from_joints(vector, kinematics, include_gripper=include_gripper)
            for vector in vectors
        ]
    ).astype(np.float32)


def _feature_names(include_gripper: bool) -> list[str]:
    names = CARTESIAN_POSE_NAMES.copy()
    if include_gripper:
        names.append("gripper")
    return names


def _update_info(output_root: Path, *, include_gripper: bool) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = _feature_names(include_gripper)
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [len(names)]
        info["features"][key]["names"] = names
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def _stats_for(values: np.ndarray) -> dict[str, list[Any]]:
    stats: dict[str, list[Any]] = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, quantile in QUANTILES.items():
        stats[name] = np.quantile(values, quantile, axis=0).astype(float).tolist()
    return stats


def _update_stats(output_root: Path, state_values: np.ndarray, action_values: np.ndarray) -> None:
    stats_path = output_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["observation.state"] = _stats_for(state_values)
    stats["action"] = _stats_for(action_values)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def _copy_source_dataset(source_root: Path, output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists. Use --force to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def _validate_source_features(source_root: Path) -> None:
    info_path = source_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    for key in ("observation.state", "action"):
        feature = features.get(key) or {}
        names = feature.get("names")
        shape = feature.get("shape")
        if names != JOINT_NAMES or shape != [len(JOINT_NAMES)]:
            raise ValueError(
                f"{source_root} does not look like a joint-space Go-board dataset: "
                f"{key} has names={names!r}, shape={shape!r}."
            )


def convert_dataset_to_cartesian(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    urdf_path: Path,
    target_frame_name: str,
    include_gripper: bool,
    force: bool,
) -> dict[str, Any]:
    _validate_source_features(source_root)
    _copy_source_dataset(source_root, output_root, force=force)
    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=target_frame_name,
        joint_names=JOINT_NAMES,
    )

    data_files = sorted((output_root / "data").glob("chunk-*/*.parquet"))
    all_states = []
    all_actions = []
    total_rows = 0
    for parquet_path in tqdm(data_files, desc="Converting dataset parquet"):
        table = pq.read_table(parquet_path)
        states = _fixed_size_list_to_numpy(table["observation.state"])
        actions = _fixed_size_list_to_numpy(table["action"])
        cart_states = _transform_vectors(states, kinematics, include_gripper=include_gripper)
        cart_actions = _transform_vectors(actions, kinematics, include_gripper=include_gripper)

        state_index = table.schema.get_field_index("observation.state")
        action_index = table.schema.get_field_index("action")
        table = table.set_column(
            state_index,
            "observation.state",
            _numpy_to_fixed_size_list(cart_states),
        )
        table = table.set_column(action_index, "action", _numpy_to_fixed_size_list(cart_actions))
        pq.write_table(table.replace_schema_metadata(None), parquet_path)

        all_states.append(cart_states)
        all_actions.append(cart_actions)
        total_rows += table.num_rows

    state_values = np.concatenate(all_states, axis=0)
    action_values = np.concatenate(all_actions, axis=0)
    _update_info(output_root, include_gripper=include_gripper)
    _update_stats(output_root, state_values, action_values)

    summary = {
        "repo_id": repo_id,
        "source_root": str(source_root),
        "dataset_root": str(output_root),
        "frames": int(total_rows),
        "data_files": len(data_files),
        "action_space": "cartesian",
        "vector_names": _feature_names(include_gripper),
        "urdf_path": str(urdf_path),
        "target_frame_name": target_frame_name,
        "include_gripper_in_cartesian": include_gripper,
        "units": {
            "x": "m",
            "y": "m",
            "z": "m",
            "roll": "deg",
            "pitch": "deg",
            "yaw": "deg",
            "gripper": "raw_dataset_value",
        },
    }
    (output_root / "go_cartesian_conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default="callum/go_board_cartesian_v1")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_SO101_URDF)
    parser.add_argument("--target-frame-name", default=DEFAULT_TARGET_FRAME_NAME)
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = convert_dataset_to_cartesian(
        source_root=args.source_root,
        output_root=args.dataset_root,
        repo_id=args.repo_id,
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        include_gripper=args.include_gripper,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
