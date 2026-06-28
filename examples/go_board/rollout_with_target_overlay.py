#!/usr/bin/env python
"""Run a Go-board policy with the target marker overlaid on live overhead frames."""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec - local/private gRPC transport mirrors LeRobot async inference.
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from detect_board_state import (
    BoardState,
    GO_COLUMNS,
    board_delta_to_jsonable,
    board_state_from_image,
    curved_grid_points_from_corners,
    delta_between_board_states,
    inverse_transform_row_col,
    transform_board_state,
)
from process_overhead_recording import _draw_target_marker, _marker_radius
from recording_dashboard import load_dashboard_config

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
import lerobot.policies  # noqa: F401  # Registers built-in policy config choices such as ACT.
from lerobot.processor import (
    ObservationProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies import BaseStrategy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)


DEFAULT_CROP_LEFT_RIGHT_RATIO = 0.10


@dataclass
class GripperCommandGuard:
    """Optional rollout-time filter for noisy gripper targets."""

    deadband: float = 0.0
    max_step: float = 0.0
    last_value: float | None = None

    @property
    def enabled(self) -> bool:
        return self.deadband > 0 or self.max_step > 0

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        key = "gripper.pos"
        if not self.enabled or key not in action:
            return action
        target = float(action[key])
        if self.last_value is None:
            self.last_value = target
            return action

        guarded = target
        delta = target - self.last_value
        if self.deadband > 0 and abs(delta) < self.deadband:
            guarded = self.last_value
        elif self.max_step > 0 and abs(delta) > self.max_step:
            guarded = self.last_value + self.max_step * (1.0 if delta > 0 else -1.0)

        self.last_value = guarded
        if guarded == target:
            return action
        updated = dict(action)
        updated[key] = guarded
        return updated


class ActionTraceRecorder:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.started_at = time.time()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(
        self,
        *,
        tick: int,
        observation: dict[str, Any],
        raw_action: dict[str, float] | None,
        processed_action: dict[str, float] | None,
    ) -> None:
        if self.path is None:
            return
        record = {
            "tick": tick,
            "timestamp": time.time(),
            "elapsed_s": time.time() - self.started_at,
            "observed_gripper": _float_or_none(observation.get("gripper.pos")),
            "raw_action": _jsonable_action(raw_action),
            "processed_action": _jsonable_action(processed_action),
            "raw_gripper": _float_or_none((raw_action or {}).get("gripper.pos")),
            "processed_gripper": _float_or_none((processed_action or {}).get("gripper.pos")),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")


class ModelMetricsRecorder:
    def __init__(self, path: Path | None, *, source: str) -> None:
        self.path = path
        self.source = source
        self.latencies_s: list[float] = []
        self.started_at = time.time()

    def record_call(self, latency_s: float) -> None:
        self.latencies_s.append(float(latency_s))

    def write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        call_count = len(self.latencies_s)
        total_s = sum(self.latencies_s)
        payload = {
            "schema_version": 1,
            "source": self.source,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "model_call_count": call_count,
            "avg_model_latency_s": (total_s / call_count) if call_count else None,
            "min_model_latency_s": min(self.latencies_s) if call_count else None,
            "max_model_latency_s": max(self.latencies_s) if call_count else None,
            "total_model_latency_s": total_s,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable_action(action: dict[str, Any] | None) -> dict[str, float] | None:
    if action is None:
        return None
    return {key: float(value) for key, value in action.items()}


def crop_left_right_and_resize(image: np.ndarray, crop_ratio: float, size: int) -> np.ndarray:
    if crop_ratio < 0 or crop_ratio >= 0.5:
        raise ValueError("crop_ratio must be >= 0 and < 0.5")
    if crop_ratio > 0:
        width = image.shape[1]
        left = int(width * crop_ratio)
        right = int(width * (1.0 - crop_ratio))
        image = image[:, left:right]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def policy_image_size(policy_config: PreTrainedConfig) -> int:
    for feature in policy_config.input_features.values():
        if len(feature.shape) == 3:
            return int(feature.shape[-1])
    return 224


def _coord_from_row_col(row: int, col: int, board_size: int) -> str:
    columns = GO_COLUMNS if board_size == 19 else "".join(chr(ord("A") + i) for i in range(board_size))
    return f"{columns[col]}{row + 1}"


DONE_ENV_STATE_NAMES = ("done",)
DEFAULT_DONE_STABLE_FRAMES = 5


def _done_env_feature() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(DONE_ENV_STATE_NAMES),),
        "names": list(DONE_ENV_STATE_NAMES),
    }


def _add_done_env_feature(dataset_features: dict[str, Any]) -> None:
    dataset_features[OBS_ENV_STATE] = _done_env_feature()


def _board_detection_kwargs_from_dashboard(config: Any) -> dict[str, Any]:
    return {
        "size": int(config.board.size),
        "corners": np.array(config.board.corners_tl_tr_br_bl, dtype=np.float32),
        "board_pixels": int(config.board.board_pixels),
        "sample_radius_ratio": float(config.board.sample_radius_ratio),
        "black_l_threshold": float(config.board.black_l_threshold),
        "white_l_threshold": float(config.board.white_l_threshold),
        "white_s_threshold": float(config.board.white_s_threshold),
        "stone_min_radius_ratio": float(config.board.stone_min_radius_ratio),
        "stone_max_radius_ratio": float(config.board.stone_max_radius_ratio),
        "stone_min_circularity": float(config.board.stone_min_circularity),
        "stone_max_snap_distance_ratio": float(config.board.stone_max_snap_distance_ratio),
        "black_grid_min_edge_score": float(config.board.black_grid_min_edge_score),
        "overlay_fisheye_k": float(config.board.overlay_fisheye_k),
        "camera_to_robot_rotation_degrees": int(config.board.camera_to_robot_rotation_degrees),
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


def _delta_done_for_target(delta: dict[str, Any], target_coord: str, target_color: str) -> bool:
    added = delta.get("added") or []
    removed = delta.get("removed") or []
    changed = delta.get("changed") or []
    if len(added) != 1 or removed or changed:
        return False
    stone = added[0]
    return (
        str(stone.get("coord", "")).upper() == target_coord.upper()
        and str(stone.get("color", "")).lower() == target_color.lower()
    )


class TargetOverlayObservationStep(ObservationProcessorStep):
    def __init__(
        self,
        camera_name: str,
        board_size: int,
        corners_tl_tr_br_bl: list[list[float]],
        camera_to_robot_rotation_degrees: int,
        overlay_fisheye_k: float,
        target_row: int,
        target_col: int,
        target_color: str,
        image_size: int,
        crop_left_right_ratio: float,
        preview_dir: Path | None = None,
        recording_dir: Path | None = None,
        board_detection_kwargs: dict[str, Any] | None = None,
        done_stable_frames: int = DEFAULT_DONE_STABLE_FRAMES,
    ) -> None:
        self.camera_name = camera_name
        self.board_size = board_size
        self.corners = np.array(corners_tl_tr_br_bl, dtype=np.float32)
        self.rotation = camera_to_robot_rotation_degrees
        self.overlay_fisheye_k = overlay_fisheye_k
        self.target_row = target_row
        self.target_col = target_col
        self.target_color = target_color
        self.image_size = image_size
        self.crop_left_right_ratio = crop_left_right_ratio
        self.preview_dir = preview_dir
        self.recording_dir = recording_dir
        self.started_at: float | None = None
        self.sample_index = 0
        self.board_detection_kwargs = board_detection_kwargs
        self.target_coord = _coord_from_row_col(self.target_row, self.target_col, self.board_size)
        self.baseline_state: BoardState | None = None
        self.latest_task_state: dict[str, Any] | None = None
        self.done_stable_frames = max(1, int(done_stable_frames))
        self.done_candidate_count = 0
        self.done_latched = False

    def observation(self, observation: RobotObservation) -> RobotObservation:
        frame = observation.get(self.camera_name)
        if frame is None:
            return observation
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected camera '{self.camera_name}' to be a numpy array, got {type(frame)}.")

        raw_target_frame = frame.copy()
        self._update_done_state(observation, raw_target_frame)

        camera_row, camera_col = inverse_transform_row_col(
            self.target_row,
            self.target_col,
            self.board_size,
            self.rotation,
        )
        points = curved_grid_points_from_corners(
            frame.shape,
            self.corners,
            self.board_size,
            self.overlay_fisheye_k,
        )
        if not self.done_latched:
            center = points[camera_row][camera_col]
            radius = _marker_radius(points, camera_row, camera_col)
            target_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            target_frame_bgr = _draw_target_marker(target_frame_bgr, center, radius, self.target_color)
            observation[self.camera_name] = cv2.cvtColor(target_frame_bgr, cv2.COLOR_BGR2RGB)
        for name, value in list(observation.items()):
            if isinstance(value, np.ndarray) and value.ndim == 3:
                observation[name] = crop_left_right_and_resize(
                    value, self.crop_left_right_ratio, self.image_size
                )
        self._write_preview_frames(observation)
        self._write_recording_sample(observation)
        return observation

    def transform_features(self, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return features

    def _update_done_state(self, observation: RobotObservation, target_frame_rgb: np.ndarray) -> None:
        task_state = {
            "index": self.sample_index,
            "done": False,
            "target": {
                "coord": self.target_coord,
                "row": self.target_row,
                "col": self.target_col,
                "color": self.target_color,
            },
            "reason": "board detector unavailable",
            "delta": None,
        }
        try:
            if self.board_detection_kwargs is not None:
                board_frame_bgr = cv2.cvtColor(target_frame_rgb, cv2.COLOR_RGB2BGR)
                current_state = _detect_board_state_from_frame(board_frame_bgr, self.board_detection_kwargs)
                if self.baseline_state is None:
                    self.baseline_state = current_state
                    task_state["reason"] = "baseline captured"
                else:
                    delta = board_delta_to_jsonable(delta_between_board_states(self.baseline_state, current_state))
                    candidate_done = _delta_done_for_target(delta, self.target_coord, self.target_color)
                    if not self.done_latched:
                        self.done_candidate_count = self.done_candidate_count + 1 if candidate_done else 0
                        self.done_latched = self.done_candidate_count >= self.done_stable_frames
                    task_state["done"] = self.done_latched
                    task_state["delta"] = {
                        "added": len(delta.get("added") or []),
                        "removed": len(delta.get("removed") or []),
                        "changed": len(delta.get("changed") or []),
                    }
                    task_state["candidate_done"] = candidate_done
                    task_state["stable_count"] = self.done_candidate_count
                    task_state["stable_required"] = self.done_stable_frames
                    task_state["reason"] = (
                        f"target condition stable for {self.done_candidate_count}/{self.done_stable_frames} frames"
                        if candidate_done and not self.done_latched
                        else "target occupied with correct colour and no other board changes"
                        if self.done_latched
                        else "target is not the only board change"
                    )
        except Exception as exc:  # noqa: BLE001 - task telemetry should not stop policy rollout.
            task_state["reason"] = f"board: {exc}"
        self.latest_task_state = task_state
        done_state = np.array([1.0 if task_state["done"] else 0.0], dtype=np.float32)
        observation["done"] = float(done_state[0])
        observation["environment_state"] = done_state
        observation[OBS_ENV_STATE] = done_state

    def _write_preview_frames(self, observation: RobotObservation) -> None:
        if self.preview_dir is None:
            return
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        for name, value in observation.items():
            if not isinstance(value, np.ndarray) or value.ndim != 3:
                continue
            frame = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            tmp_path = self.preview_dir / f"{name}.jpg.tmp"
            final_path = self.preview_dir / f"{name}.jpg"
            tmp_path.write_bytes(encoded.tobytes())
            tmp_path.replace(final_path)

    def _write_recording_sample(self, observation: RobotObservation) -> None:
        if self.recording_dir is None:
            return
        sample_started_at = time.time()
        if self.started_at is None:
            self.started_at = sample_started_at
        frames_dir = self.recording_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        camera_files: dict[str, str] = {}
        joint_values: dict[str, float] = {}
        for name, value in observation.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                camera_dir = frames_dir / name
                camera_dir.mkdir(parents=True, exist_ok=True)
                frame = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                relative = Path("frames") / name / f"{self.sample_index:06d}.jpg"
                cv2.imwrite(str(self.recording_dir / relative), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                camera_files[name] = str(relative)
            elif name.endswith(".pos"):
                joint_values[name.removesuffix(".pos")] = float(value)
        sample = {
            "index": self.sample_index,
            "timestamp": sample_started_at,
            "elapsed_s": sample_started_at - self.started_at,
            "cameras": camera_files,
            "telemetry": {"joints": joint_values},
        }
        if self.latest_task_state is not None:
            sample["done"] = bool(self.latest_task_state["done"])
            sample["task_state"] = self.latest_task_state
        telemetry_path = self.recording_dir / "telemetry.jsonl"
        with telemetry_path.open("a", encoding="utf-8") as telemetry_file:
            telemetry_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
        self.sample_index += 1


class TimedBaseStrategy(BaseStrategy):
    """Base rollout strategy with per-stage timing logs for Go policy debugging."""

    def __init__(
        self,
        config: BaseStrategyConfig,
        warn_threshold_s: float = 0.05,
        action_trace: ActionTraceRecorder | None = None,
        gripper_guard: GripperCommandGuard | None = None,
        done_check: Any | None = None,
        model_metrics: ModelMetricsRecorder | None = None,
    ) -> None:
        super().__init__(config)
        self.warn_threshold_s = warn_threshold_s
        self.action_trace = action_trace or ActionTraceRecorder(None)
        self.gripper_guard = gripper_guard or GripperCommandGuard()
        self.done_check = done_check
        self.model_metrics = model_metrics or ModelMetricsRecorder(None, source="local")

    def _timed_send_next_action(
        self,
        tick: int,
        obs_processed: dict,
        obs_raw: dict,
        ctx,
        interpolator,
    ) -> tuple[dict | None, dict[str, float]]:
        timings: dict[str, float] = {}
        engine = ctx.policy.inference
        features = ctx.data.dataset_features
        ordered_keys = ctx.data.ordered_action_keys

        if interpolator.needs_new_action():
            stage_start = time.perf_counter()
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            timings["build_dataset_frame"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            action_tensor = engine.get_action(obs_frame)
            timings["policy_get_action"] = time.perf_counter() - stage_start
            self.model_metrics.record_call(timings["policy_get_action"])
            if action_tensor is not None:
                stage_start = time.perf_counter()
                interpolator.add(action_tensor.cpu())
                timings["interpolator_add"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        interp = interpolator.get()
        timings["interpolator_get"] = time.perf_counter() - stage_start
        if interp is None:
            return None, timings

        if len(interp) != len(ordered_keys):
            raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
        action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
        guarded_action_dict = self.gripper_guard.apply(action_dict)

        stage_start = time.perf_counter()
        processed = ctx.processors.robot_action_processor((guarded_action_dict, obs_raw))
        timings["action_processor"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        ctx.hardware.robot_wrapper.send_action(processed)
        timings["robot_send_action"] = time.perf_counter() - stage_start
        self.action_trace.write(
            tick=tick,
            observation=obs_raw,
            raw_action=action_dict,
            processed_action=processed,
        )
        return guarded_action_dict, timings

    def _log_loop_timings(self, tick: int, total_s: float, timings: dict[str, float], target_fps: float) -> None:
        line = " | ".join(f"{name}={value * 1000:.1f}ms" for name, value in timings.items())
        if total_s >= self.warn_threshold_s or total_s > (1.0 / target_fps):
            logger.warning("Go rollout timing tick=%d total=%.1fms | %s", tick, total_s * 1000, line)
        else:
            logger.info("Go rollout timing tick=%d total=%.1fms | %s", tick, total_s * 1000, line)

    def run(self, ctx) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        tick = 0
        engine.resume()
        logger.info("Base strategy control loop started with Go timing instrumentation")

        try:
            while not ctx.runtime.shutdown_event.is_set():
                tick += 1
                loop_start = time.perf_counter()
                timings: dict[str, float] = {}

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                stage_start = time.perf_counter()
                obs = robot.get_observation()
                timings["robot_get_observation"] = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                timings["observation_processors"] = time.perf_counter() - stage_start
                if self.done_check is not None and self.done_check():
                    logger.info("Target done condition reached; stopping rollout before timeout")
                    break

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict, action_timings = self._timed_send_next_action(tick, obs_processed, obs, ctx, interpolator)
                timings.update(action_timings)

                stage_start = time.perf_counter()
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                timings["telemetry_log"] = time.perf_counter() - stage_start

                dt = time.perf_counter() - loop_start
                self._log_loop_timings(tick, dt, timings, cfg.fps)
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                else:
                    logger.warning(
                        f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                    )
        finally:
            self.model_metrics.write()


def parse_go_coord(coord: str, board_size: int) -> tuple[int, int, str]:
    coord = coord.strip().upper()
    if len(coord) < 2:
        raise ValueError("Coordinate must look like Q16.")
    columns = GO_COLUMNS if board_size == 19 else "".join(chr(ord("A") + i) for i in range(board_size))
    col = columns.find(coord[0])
    try:
        row = int(coord[1:]) - 1
    except ValueError as exc:
        raise ValueError("Coordinate must use a letter plus row number, for example Q16.") from exc
    if col < 0 or row < 0 or row >= board_size:
        raise ValueError(f"Coordinate must be on a {board_size}x{board_size} Go board.")
    return row, col, f"{columns[col]}{row + 1}"


def camera_configs_from_dashboard(config_path: Path) -> dict[str, OpenCVCameraConfig]:
    config = load_dashboard_config(config_path)
    return {
        camera.name: OpenCVCameraConfig(
            index_or_path=camera.index_or_path,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            warmup_s=camera.warmup_s,
            fourcc=camera.fourcc,
        )
        for camera in config.cameras
    }


def _remote_observation_features(robot_observation_features: dict[str, Any], image_size: int) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key, value in robot_observation_features.items():
        if isinstance(value, tuple):
            features[key] = (image_size, image_size, 3)
        elif value is float and key.endswith(".pos"):
            features[key] = value
    return features


def _action_tensor_to_dict(action: torch.Tensor, action_keys: list[str]) -> dict[str, float]:
    flat = action.detach().cpu().flatten()
    if len(flat) != len(action_keys):
        raise ValueError(f"Remote action has {len(flat)} values, expected {len(action_keys)}.")
    return {key: float(flat[index]) for index, key in enumerate(action_keys)}


def _get_observation_with_retries(robot: Any, num_retries: int, retry_delay_s: float) -> RobotObservation:
    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
        try:
            return robot.get_observation()
        except ConnectionError as exc:
            last_error = exc
            if attempt >= num_retries:
                break
            logger.warning(
                "Robot observation read failed (%d/%d): %s; retrying in %.2fs",
                attempt + 1,
                num_retries + 1,
                exc,
                retry_delay_s,
            )
            precise_sleep(retry_delay_s)
    assert last_error is not None
    raise last_error


def _return_to_initial_position(
    robot: Any,
    initial_action: dict[str, float],
    *,
    action_keys: list[str],
    duration_s: float,
    fps: float,
    read_retries: int,
    retry_delay_s: float,
) -> None:
    if not initial_action or duration_s <= 0:
        return
    current_obs = _get_observation_with_retries(
        robot,
        num_retries=read_retries,
        retry_delay_s=retry_delay_s,
    )
    current = {key: float(current_obs.get(key, initial_action[key])) for key in action_keys if key in initial_action}
    steps = max(1, int(duration_s * max(fps, 1.0)))
    interval_s = 1.0 / max(fps, 1.0)
    for step in range(1, steps + 1):
        ratio = step / steps
        eased = ratio * ratio * (3.0 - 2.0 * ratio)
        action = {
            key: current[key] + (initial_action[key] - current[key]) * eased
            for key in current
        }
        robot.send_action(action)
        if step < steps:
            precise_sleep(interval_s)
    logger.info("Initial-position return complete: duration=%.1fs steps=%d", duration_s, steps)


def run_remote_policy_server_rollout(args: argparse.Namespace, config: Any, target_row: int, target_col: int, coord: str) -> None:
    import grpc

    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.robots import make_robot_from_config
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

    image_size = int(args.policy_image_size)
    robot_config = SO101FollowerConfig(
        port=config.robot.port,
        id=config.robot.id,
        cameras=camera_configs_from_dashboard(args.config),
        disable_torque_on_disconnect=False,
        configure_on_connect=config.robot.configure_on_connect,
        use_degrees=True,
    )
    robot = make_robot_from_config(robot_config)

    channel = grpc.insecure_channel(args.remote_policy_server, grpc_channel_options(initial_backoff="0.0500s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    action_queue: deque[torch.Tensor] = deque()
    initial_action: dict[str, float] | None = None
    action_trace = ActionTraceRecorder(_action_trace_path(args.recording_dir))
    model_metrics = ModelMetricsRecorder(_model_metrics_path(args.recording_dir), source="remote_policy_server")
    gripper_guard = GripperCommandGuard(
        deadband=args.gripper_deadband,
        max_step=args.gripper_max_step,
    )

    try:
        logger.info("Connecting to remote policy server at %s", args.remote_policy_server)
        ready_start = time.perf_counter()
        stub.Ready(services_pb2.Empty(), timeout=args.remote_timeout)

        robot.connect()
        logger.info("Local robot and cameras connected for remote policy rollout")
        action_keys = [key for key in robot.action_features if key.endswith(".pos")]
        observation_features = _remote_observation_features(robot.observation_features, image_size)
        lerobot_features = hw_to_dataset_features(observation_features, OBS_STR, use_video=False)
        _add_done_env_feature(lerobot_features)
        policy_setup = RemotePolicyConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=args.policy_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
        )
        setup_start = time.perf_counter()
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_setup)), timeout=args.remote_timeout)
        logger.info(
            "Remote policy ready: handshake=%.1fms setup/load=%.1fms server=%s policy=%s actions_per_chunk=%d",
            (setup_start - ready_start) * 1000,
            (time.perf_counter() - setup_start) * 1000,
            args.remote_policy_server,
            args.policy_path,
            args.actions_per_chunk,
        )

        overlay_processor = TargetOverlayObservationStep(
            camera_name=config.board.camera,
            board_size=config.board.size,
            corners_tl_tr_br_bl=config.board.corners_tl_tr_br_bl,
            camera_to_robot_rotation_degrees=config.board.camera_to_robot_rotation_degrees,
            overlay_fisheye_k=config.board.overlay_fisheye_k,
            target_row=target_row,
            target_col=target_col,
            target_color=args.color,
            image_size=image_size,
            crop_left_right_ratio=args.crop_left_right_ratio,
            preview_dir=args.preview_dir,
            recording_dir=args.recording_dir,
            board_detection_kwargs=_board_detection_kwargs_from_dashboard(config),
            done_stable_frames=args.done_stable_frames,
        )

        control_interval = 1.0 / args.fps
        start_time = time.perf_counter()
        tick = 0
        logger.info("Remote policy rollout loop started")
        while True:
            tick += 1
            loop_start = time.perf_counter()
            if args.duration > 0 and (loop_start - start_time) >= args.duration:
                logger.info("Duration limit reached (%.0fs)", args.duration)
                break

            obs_start = time.perf_counter()
            observation = _get_observation_with_retries(
                robot,
                num_retries=args.motor_read_retries,
                retry_delay_s=args.motor_read_retry_delay,
            )
            if initial_action is None:
                initial_action = {key: float(observation[key]) for key in action_keys if key in observation}
            observation["task"] = args.task or f"place {args.color} stone at {coord}"
            observation = overlay_processor.observation(observation)
            obs_time = time.perf_counter() - obs_start
            if args.stop_on_done and overlay_processor.done_latched:
                logger.info("Target done condition reached; stopping rollout before timeout")
                break

            if not action_queue:
                timed_observation = TimedObservation(
                    timestamp=time.time(),
                    timestep=tick,
                    observation=observation,
                    must_go=True,
                )
                serialize_start = time.perf_counter()
                observation_bytes = pickle.dumps(timed_observation)
                serialize_time = time.perf_counter() - serialize_start

                send_start = time.perf_counter()
                stub.SendObservations(
                    send_bytes_in_chunks(
                        observation_bytes,
                        services_pb2.Observation,
                        log_prefix="[GO REMOTE] Observation",
                        silent=True,
                    ),
                    timeout=args.remote_timeout,
                )
                send_time = time.perf_counter() - send_start

                actions_start = time.perf_counter()
                actions_response = stub.GetActions(services_pb2.Empty(), timeout=args.remote_timeout)
                actions_time = time.perf_counter() - actions_start
                model_metrics.record_call(actions_time)

                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_response.data) if actions_response.data else []  # nosec
                deserialize_time = time.perf_counter() - deserialize_start
                action_queue.extend(action.get_action().detach().cpu() for action in timed_actions)
                logger.info(
                    "Remote round trip tick=%d total=%.1fms obs=%.1fms serialize=%.1fms send=%.1fms wait=%.1fms deserialize=%.1fms chunk=%d",
                    tick,
                    (time.perf_counter() - loop_start) * 1000,
                    obs_time * 1000,
                    serialize_time * 1000,
                    send_time * 1000,
                    actions_time * 1000,
                    deserialize_time * 1000,
                    len(timed_actions),
                )

            if action_queue:
                action = _action_tensor_to_dict(action_queue.popleft(), action_keys)
                guarded_action = gripper_guard.apply(action)
                sent_action = robot.send_action(guarded_action)
                action_trace.write(
                    tick=tick,
                    observation=observation,
                    raw_action=action,
                    processed_action=sent_action,
                )

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning("Remote rollout loop slower than target: %.1f Hz target=%.1f Hz", 1 / dt, args.fps)
    finally:
        try:
            if initial_action:
                logger.info("Returning robot to initial position before shutdown")
                _return_to_initial_position(
                    robot,
                    initial_action,
                    action_keys=action_keys,
                    duration_s=args.return_to_initial_duration,
                    fps=args.return_to_initial_fps,
                    read_retries=args.motor_read_retries,
                    retry_delay_s=args.motor_read_retry_delay,
                )
        finally:
            model_metrics.write()
            robot.disconnect()
            channel.close()
            logger.info("Remote policy rollout shutdown complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/go_board/dashboard_config.json"))
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--coord", required=True)
    parser.add_argument("--color", default="white", choices=["white", "black"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="")
    parser.add_argument("--timing-warn-threshold", type=float, default=0.05)
    parser.add_argument("--crop-left-right-ratio", type=float, default=DEFAULT_CROP_LEFT_RIGHT_RATIO)
    parser.add_argument("--remote-policy-server", default="", help="Host:port of a LeRobot async policy server.")
    parser.add_argument("--remote-timeout", type=float, default=30.0)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument("--policy-image-size", type=int, default=224)
    parser.add_argument("--motor-read-retries", type=int, default=2)
    parser.add_argument("--motor-read-retry-delay", type=float, default=0.05)
    parser.add_argument("--return-to-initial-duration", type=float, default=3.0)
    parser.add_argument("--return-to-initial-fps", type=float, default=50.0)
    parser.add_argument(
        "--done-stable-frames",
        type=int,
        default=DEFAULT_DONE_STABLE_FRAMES,
        help="Latch done only after the target-only board condition holds for this many consecutive frames.",
    )
    parser.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop the rollout as soon as the stable target condition is detected.",
    )
    parser.add_argument(
        "--gripper-deadband",
        type=float,
        default=0.0,
        help="Keep the previous gripper command when the next target changes by less than this amount.",
    )
    parser.add_argument(
        "--gripper-max-step",
        type=float,
        default=0.0,
        help="Maximum gripper command change per control tick. Set 0 to disable.",
    )
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--display-data", action="store_true")
    return parser


def _action_trace_path(recording_dir: Path | None) -> Path | None:
    if recording_dir is None:
        return None
    return recording_dir / "actions.jsonl"


def _model_metrics_path(recording_dir: Path | None) -> Path | None:
    if recording_dir is None:
        return None
    return recording_dir / "model_metrics.json"


def main() -> None:
    args = build_parser().parse_args()
    init_logging()
    config = load_dashboard_config(args.config)
    if config.board.corners_tl_tr_br_bl is None:
        raise ValueError("dashboard_config.json must set board.corners_tl_tr_br_bl for target overlay rollout.")

    target_row, target_col, coord = parse_go_coord(args.coord, config.board.size)
    if args.remote_policy_server:
        run_remote_policy_server_rollout(args, config, target_row, target_col, coord)
        return

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path
    image_size = policy_image_size(policy_config)

    robot_config = SO101FollowerConfig(
        port=config.robot.port,
        id=config.robot.id,
        cameras=camera_configs_from_dashboard(args.config),
        disable_torque_on_disconnect=False,
        configure_on_connect=config.robot.configure_on_connect,
        use_degrees=True,
    )
    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        duration=args.duration,
        device=args.device,
        task=args.task or f"place {args.color} stone at {coord}",
        display_data=args.display_data,
    )
    overlay_step = TargetOverlayObservationStep(
        camera_name=config.board.camera,
        board_size=config.board.size,
        corners_tl_tr_br_bl=config.board.corners_tl_tr_br_bl,
        camera_to_robot_rotation_degrees=config.board.camera_to_robot_rotation_degrees,
        overlay_fisheye_k=config.board.overlay_fisheye_k,
        target_row=target_row,
        target_col=target_col,
        target_color=args.color,
        image_size=image_size,
        crop_left_right_ratio=args.crop_left_right_ratio,
        preview_dir=args.preview_dir,
        recording_dir=args.recording_dir,
        board_detection_kwargs=_board_detection_kwargs_from_dashboard(config),
        done_stable_frames=args.done_stable_frames,
    )
    overlay_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[overlay_step],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    signal_handler = ProcessSignalHandler(use_threads=True)
    context = build_rollout_context(cfg, signal_handler.shutdown_event, robot_observation_processor=overlay_processor)
    _add_done_env_feature(context.data.dataset_features)
    strategy = TimedBaseStrategy(
        cfg.strategy,
        warn_threshold_s=args.timing_warn_threshold,
        action_trace=ActionTraceRecorder(_action_trace_path(args.recording_dir)),
        model_metrics=ModelMetricsRecorder(_model_metrics_path(args.recording_dir), source="local_policy"),
        gripper_guard=GripperCommandGuard(
            deadband=args.gripper_deadband,
            max_step=args.gripper_max_step,
        ),
        done_check=(lambda: bool(args.stop_on_done and overlay_step.done_latched)),
    )
    try:
        strategy.setup(context)
        strategy.run(context)
    finally:
        strategy.teardown(context)


if __name__ == "__main__":
    main()
