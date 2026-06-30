#!/usr/bin/env python
"""Evaluate a trained Go-board diffusion policy inside the MuJoCo scene.

The output intentionally mirrors the dashboard/evaluation recording layout:
each rollout gets raw overhead and wrist frames, target-overlaid overhead
frames, telemetry rows, metadata, and optional camera-feed videos.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.factory import make_policy, make_pre_post_processors

from generate_mujoco_nudge_data import (
    JOINT_NAMES,
    REAL_CENTER_TOUCH_JOINTS,
    NudgeConfig,
    NudgeMuJoCo,
    _draw_overhead_target_overlay,
    _joint_rows,
    _scenario,
    _write_rgb_jpeg,
    default_targets,
)
from mujoco_board_sim import BoardSpec


DEFAULT_POLICY_PATH = Path("outputs/train/diffusion_go_board_v1_30k/checkpoints/030000/pretrained_model")
DEFAULT_DATASET_ROOT = Path("outputs/datasets/go_board_v1_diffusion")
DEFAULT_DATASET_REPO_ID = "callum/go_board_v1_diffusion"
DEFAULT_OUTPUT_ROOT = Path("outputs/go_board_mujoco_diffusion_eval")


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 3
    max_steps: int = 300
    seed: int = 29
    width: int = 640
    height: int = 480
    fps: int = 10
    settle_steps: int = 750
    control_substeps: int = 12
    tolerance_m: float = 0.0035
    stable_success_frames: int = 2
    no_neighbors: bool = False
    stone_color: str | None = None
    write_videos: bool = True


def _policy_image_size(policy_config: Any, key: str) -> tuple[int, int]:
    feature = policy_config.input_features[key]
    if len(feature.shape) != 3:
        raise ValueError(f"{key} must be a CHW visual feature, got {feature.shape}.")
    _channels, height, width = feature.shape
    return int(width), int(height)


def _resize_for_policy(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image_rgb.shape[1] == width and image_rgb.shape[0] == height:
        return image_rgb
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def _state_vector(joints: dict[str, float]) -> np.ndarray:
    return np.array([float(joints[name]) for name in JOINT_NAMES], dtype=np.float32)


def _action_tensor_to_joints(action: torch.Tensor) -> dict[str, float]:
    values = action.detach().cpu().numpy()
    if values.ndim == 2:
        values = values[0]
    if values.shape[0] != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} action values, got shape {values.shape}.")
    joints = {name: float(value) for name, value in zip(JOINT_NAMES, values, strict=True)}
    joints["gripper"] = float(np.clip(joints["gripper"], 0.0, 100.0))
    return joints


def _move_arm_towards(sim: NudgeMuJoCo, action_joints: dict[str, float], substeps: int) -> None:
    start = sim.current_arm_joint_targets()
    for idx in range(max(1, int(substeps))):
        alpha = (idx + 1) / max(1, int(substeps))
        interpolated = {
            name: start[name] + (float(action_joints[name]) - start[name]) * alpha for name in JOINT_NAMES
        }
        sim.set_arm_joint_targets(interpolated)
        sim.step(1)


def _write_video(frame_paths: list[Path], destination: Path, fps: int) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()


def _recording_timestamp(started_at: float, sample_index: int, fps: int) -> float:
    return started_at + sample_index / max(1, fps)


def run_episode(
    *,
    run_dir: Path,
    scenario: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    policy_config: Any,
    device: torch.device,
    eval_cfg: EvalConfig,
    spec: BoardSpec,
) -> dict[str, Any]:
    episode_dir = run_dir / scenario.episode_id
    if episode_dir.exists():
        raise FileExistsError(f"Episode output already exists: {episode_dir}")
    (episode_dir / "frames" / "overhead").mkdir(parents=True)
    (episode_dir / "frames" / "wrist").mkdir(parents=True)
    (episode_dir / "overhead_processed").mkdir(parents=True)
    (episode_dir / "videos").mkdir(parents=True)

    sim = NudgeMuJoCo(
        scenario,
        spec,
        (eval_cfg.width, eval_cfg.height),
        robot_arm=True,
    )
    policy.reset()
    sim.set_arm_joint_targets(dict(REAL_CENTER_TOUCH_JOINTS))
    sim.step(eval_cfg.settle_steps)

    started_at = time.time()
    rows: list[dict[str, Any]] = []
    overhead_paths: list[Path] = []
    processed_paths: list[Path] = []
    wrist_paths: list[Path] = []
    stable = 0
    overhead_size = _policy_image_size(policy_config, "observation.images.overhead")
    wrist_size = _policy_image_size(policy_config, "observation.images.wrist")

    try:
        for sample_index in range(eval_cfg.max_steps):
            state_joints = sim.current_arm_joint_targets()
            error = sim.target_error()
            done = error <= eval_cfg.tolerance_m
            stable = stable + 1 if done else 0

            overhead_rgb = sim.render("overhead")
            overhead_processed_rgb = _draw_overhead_target_overlay(overhead_rgb, scenario, spec)
            wrist_rgb = sim.render("wrist")

            overhead_rel = Path("frames") / "overhead" / f"{sample_index:06d}.jpg"
            wrist_rel = Path("frames") / "wrist" / f"{sample_index:06d}.jpg"
            processed_rel = Path("overhead_processed") / f"{sample_index:06d}.jpg"
            _write_rgb_jpeg(episode_dir / overhead_rel, overhead_rgb)
            _write_rgb_jpeg(episode_dir / wrist_rel, wrist_rgb)
            _write_rgb_jpeg(episode_dir / processed_rel, overhead_processed_rgb)
            overhead_paths.append(episode_dir / overhead_rel)
            wrist_paths.append(episode_dir / wrist_rel)
            processed_paths.append(episode_dir / processed_rel)

            raw_observation = {
                "observation.images.overhead": _resize_for_policy(overhead_processed_rgb, overhead_size),
                "observation.images.wrist": _resize_for_policy(wrist_rgb, wrist_size),
                "observation.state": _state_vector(state_joints),
            }
            observation = prepare_observation_for_inference(raw_observation, device)
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            action_joints = _action_tensor_to_joints(action)

            timestamp = _recording_timestamp(started_at, sample_index, eval_cfg.fps)
            rows.append(
                {
                    "index": sample_index,
                    "timestamp": timestamp,
                    "elapsed_s": sample_index / max(1, eval_cfg.fps),
                    "cameras": {
                        "overhead": str(overhead_rel),
                        "wrist": str(wrist_rel),
                        "overhead_processed": str(processed_rel),
                    },
                    "telemetry": {
                        "timestamp": timestamp,
                        "connected": True,
                        "mode": "mujoco_diffusion_policy_eval",
                        "fps": eval_cfg.fps,
                        "joints": _joint_rows(state_joints),
                        "leader_joints": _joint_rows(action_joints),
                        "teleop_enabled": True,
                        "synthetic": {
                            "policy_action": {name: round(value, 6) for name, value in action_joints.items()},
                            "target_coord": scenario.target_coord,
                            "target_xy_m": list(scenario.target_xy),
                            "stone_xy_m": sim.stone_xy().round(6).tolist(),
                            "stone_offset_xy_m": (sim.stone_xy() - sim.target_xy).round(6).tolist(),
                            "neighbor_xy_m": sim.neighbor_xy(),
                            "target_error_m": error,
                            "done": done,
                            "stable_done_count": stable,
                        },
                        "note": "MuJoCo diffusion policy evaluation sample.",
                    },
                }
            )

            if stable >= eval_cfg.stable_success_frames:
                break
            _move_arm_towards(sim, action_joints, eval_cfg.control_substeps)
    finally:
        final_error = sim.target_error()
        final_stone_xy = sim.stone_xy().round(6).tolist()
        sim.close()

    with (episode_dir / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry_file:
        for row in rows:
            telemetry_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    if eval_cfg.write_videos:
        _write_video(overhead_paths, episode_dir / "videos" / "overhead.mp4", eval_cfg.fps)
        _write_video(processed_paths, episode_dir / "videos" / "overhead_processed.mp4", eval_cfg.fps)
        _write_video(wrist_paths, episode_dir / "videos" / "wrist.mp4", eval_cfg.fps)

    success = final_error <= eval_cfg.tolerance_m
    metadata = {
        "schema_version": 1,
        "run_type": "mujoco_diffusion_policy_eval",
        "id": scenario.episode_id,
        "name": scenario.episode_id,
        "status": "complete",
        "started_at": started_at,
        "ended_at": time.time(),
        "move_name": f"mujoco_policy_{scenario.target_color}_to_{scenario.target_coord.lower()}",
        "sample_hz": eval_cfg.fps,
        "samples": len(rows),
        "cameras": {
            "overhead": {
                "name": "overhead",
                "width": eval_cfg.width,
                "height": eval_cfg.height,
                "fps": eval_cfg.fps,
            },
            "wrist": {
                "name": "wrist",
                "width": eval_cfg.width,
                "height": eval_cfg.height,
                "fps": eval_cfg.fps,
            },
        },
        "board": {
            "size": spec.size,
            "target": {
                "coord": scenario.target_coord,
                "row": scenario.target_row,
                "col": scenario.target_col,
                "color": scenario.target_color,
            },
            "target_xy_m": list(scenario.target_xy),
            "initial_offset_xy_m": list(scenario.initial_offset_xy),
            "neighbors": scenario.neighbor_stones,
        },
        "synthetic": {
            "generator": Path(__file__).name,
            "action_space": "so101_joint_targets",
            "observation_space": "rendered_overhead_target_overlay_wrist_plus_so101_joint_state",
            "robot_arm": True,
            "tolerance_m": eval_cfg.tolerance_m,
            "final_error_m": final_error,
            "final_stone_xy_m": final_stone_xy,
            "success": success,
            "control_substeps": eval_cfg.control_substeps,
            "settle_steps": eval_cfg.settle_steps,
            "policy_input_overhead": "overhead_processed",
        },
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (episode_dir / "overhead_processed" / "status.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "updated_at": time.time(),
                "frames_total": len(rows),
                "frames_written": len(rows),
                "target": metadata["board"]["target"],
                "source": "synthetic_overhead_target_overlay",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "episode": scenario.episode_id,
        "coord": scenario.target_coord,
        "color": scenario.target_color,
        "frames": len(rows),
        "success": success,
        "final_error_m": final_error,
        "final_stone_xy_m": final_stone_xy,
        "episode_dir": str(episode_dir),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = Path(args.policy_path)
    dataset_root = Path(args.dataset_root)
    device = torch.device(args.device)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    policy_config.device = str(device)
    dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    policy = make_policy(policy_config, ds_meta=dataset_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_config, pretrained_path=str(policy_path))

    eval_cfg = EvalConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
        settle_steps=args.settle_steps,
        control_substeps=args.control_substeps,
        tolerance_m=args.tolerance_m,
        stable_success_frames=args.stable_success_frames,
        no_neighbors=args.no_neighbors,
        stone_color=args.stone_color,
        write_videos=not args.no_videos,
    )
    nudge_cfg = NudgeConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        tolerance_m=args.tolerance_m,
        stable_success_frames=args.stable_success_frames,
        fps=args.fps,
        robot_arm=True,
        episode_prefix=args.episode_prefix,
        stone_color=args.stone_color,
        no_neighbors=args.no_neighbors,
    )

    spec = BoardSpec()
    run_name = args.run_name or f"{policy_path.parents[2].name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    targets = args.targets or default_targets()
    rng = random.Random(args.seed)

    started_at = time.time()
    episodes = []
    for episode_index in range(args.episodes):
        scenario = _scenario(rng, episode_index, spec, targets, nudge_cfg)
        print(f"[{episode_index + 1}/{args.episodes}] {scenario.episode_id}")
        episodes.append(
            run_episode(
                run_dir=run_dir,
                scenario=scenario,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                policy_config=policy_config,
                device=device,
                eval_cfg=eval_cfg,
                spec=spec,
            )
        )

    attempts = [
        {
            "index": index,
            "coord": str(episode["coord"]),
            "color": str(episode["color"]),
            "success": bool(episode["success"]),
            "status": "succeeded" if episode["success"] else "failed",
            "reason": "" if episode["success"] else "target stone did not reach tolerance in MuJoCo",
            "timed_out": not bool(episode["success"]) and int(episode["frames"]) >= args.max_steps,
            "rollout": str(episode["episode"]),
            "returncode": 0,
            "final_error_m": float(episode["final_error_m"]),
            "frames": int(episode["frames"]),
        }
        for index, episode in enumerate(episodes)
    ]
    successes = sum(1 for episode in episodes if episode["success"])
    summary = {
        "schema_version": 1,
        "id": run_name,
        "status": "complete",
        "environment": "mujoco",
        "started_at": started_at,
        "ended_at": time.time(),
        "run_type": "mujoco_diffusion_policy_eval",
        "run_dir": str(run_dir),
        "policy_path": str(policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "config": asdict(eval_cfg),
        "commands": [{"coord": attempt["coord"], "color": attempt["color"]} for attempt in attempts],
        "attempts": attempts,
        "targets": targets,
        "episodes": len(episodes),
        "successes": successes,
        "failures": len(episodes) - successes,
        "frames": sum(int(episode["frames"]) for episode in episodes),
        "episode_summaries": episodes,
        "message": f"MuJoCo diffusion evaluation completed: {successes}/{len(episodes)} succeeded.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--target", dest="targets", action="append")
    parser.add_argument("--episode-prefix", default="mujoco_diffusion")
    parser.add_argument("--stone-color", choices=["black", "white"])
    parser.add_argument("--no-neighbors", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=750)
    parser.add_argument("--control-substeps", type=int, default=12)
    parser.add_argument("--tolerance-m", type=float, default=0.0035)
    parser.add_argument("--stable-success-frames", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-videos", action="store_true")
    return parser


def main() -> None:
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
