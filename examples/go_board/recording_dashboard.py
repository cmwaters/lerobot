#!/usr/bin/env python
"""Local dashboard for Go-stone data collection.

The dashboard is intentionally small and dependency-light. It serves two camera
feeds, live joint telemetry, and optional end-effector pose from a fixed
overhead/robot setup.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from detect_board_state import (
    BoardState,
    GO_COLUMNS,
    Stone,
    auto_detect_board_corners,
    board_delta_to_jsonable,
    board_state_from_image,
    board_state_to_jsonable,
    delta_between_board_states,
    inverse_transform_row_col,
    transform_board_state,
)
from process_overhead_recording import overhead_processed_status, process_recording_overhead


DEFAULT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MODEL_ROLLOUTS_DIR_NAME = "model_rollouts"
MODEL_EVALUATIONS_DIR_NAME = "model_evaluations"
DEFAULT_SYNTHETIC_RECORDING_DIR = Path("outputs/go_board_mujoco_nudge_robot_recordings")
DEFAULT_EVALUATION_SEQUENCE = [
    {"coord": "A1", "color": "black"},
    {"coord": "T1", "color": "white"},
    {"coord": "A19", "color": "white"},
    {"coord": "T19", "color": "black"},
    {"coord": "K1", "color": "black"},
    {"coord": "K19", "color": "white"},
    {"coord": "A10", "color": "white"},
    {"coord": "T10", "color": "black"},
    {"coord": "B2", "color": "white"},
    {"coord": "S18", "color": "black"},
]


@dataclass
class CameraSpec:
    name: str
    index_or_path: str | int
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_s: int = 1
    fourcc: str | None = None


@dataclass
class RobotSpec:
    type: str = "so101_follower"
    port: str | None = None
    id: str = "go_follower"
    calibrate: bool = False
    configure_on_connect: bool = False
    urdf_path: str | None = None
    target_frame_name: str = "gripper_frame_link"
    rest_position: dict[str, float] = field(default_factory=dict)


@dataclass
class LeaderSpec:
    type: str = "so101_leader"
    port: str | None = None
    id: str = "so101_leader"
    calibrate: bool = False


@dataclass
class BoardSpec:
    camera: str = "overhead"
    size: int = 19
    corners_tl_tr_br_bl: list[list[float]] | None = None
    camera_to_robot_rotation_degrees: int = 0
    board_pixels: int = 1000
    overlay_fisheye_k: float = 0.0
    sample_radius_ratio: float = 0.34
    black_l_threshold: float = 80.0
    white_l_threshold: float = 165.0
    white_s_threshold: float = 75.0
    stone_min_radius_ratio: float = 0.2
    stone_max_radius_ratio: float = 0.48
    stone_min_circularity: float = 0.45
    stone_max_snap_distance_ratio: float = 0.52
    black_grid_min_edge_score: float = 0.18


@dataclass
class ModelSpec:
    policy_path: str = ""
    remote_host: str = ""
    remote_workdir: str = "~/Developer/lerobot"
    remote_policy_server: str = ""
    policy_type: str = "act"
    actions_per_chunk: int = 20
    policy_image_size: int = 224
    device: str = "cuda"
    fps: float = 10.0
    duration_s: float = 30.0
    gripper_deadband: float = 0.0
    gripper_max_step: float = 0.0
    release_local_devices: bool = True


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    cameras: list[CameraSpec] = field(default_factory=list)
    robot: RobotSpec = field(default_factory=RobotSpec)
    leader: LeaderSpec = field(default_factory=LeaderSpec)
    board: BoardSpec = field(default_factory=BoardSpec)
    model: ModelSpec = field(default_factory=ModelSpec)


@dataclass
class JointState:
    name: str
    value: float
    unit: str = "deg"
    min_value: float = -180.0
    max_value: float = 180.0


@dataclass
class EndEffectorState:
    available: bool
    x: float | None = None
    y: float | None = None
    z: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    unit: str = "m"
    source: str = "unavailable"


@dataclass
class DashboardState:
    timestamp: float
    connected: bool
    mode: str
    fps: float
    joints: list[JointState]
    leader_joints: list[JointState]
    teleop_enabled: bool
    end_effector: EndEffectorState
    note: str = ""


@dataclass
class RecordingSession:
    id: str
    path: Path
    started_at: float
    baseline: BoardState
    baseline_image: str
    sample_hz: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    samples: int = 0
    status: str = "starting"
    teleop_started: bool = False
    move_name: str = "recording"
    error: str = ""


@dataclass
class ModelRunSession:
    id: str
    coord: str
    color: str
    task: str
    policy_path: str
    remote_host: str
    remote_workdir: str
    remote_policy_server: str
    policy_type: str
    actions_per_chunk: int
    policy_image_size: int
    device: str
    fps: float
    duration_s: float
    gripper_deadband: float
    gripper_max_step: float
    started_at: float
    status: str = "starting"
    command: list[str] = field(default_factory=list)
    command_display: str = ""
    returncode: int | None = None
    error: str = ""
    log_tail: list[str] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    process: subprocess.Popen[str] | None = None
    preview_dir: Path | None = None
    rollout_dir: Path | None = None
    baseline: BoardState | None = None
    baseline_image: str | None = None
    saved_rollout_name: str = ""
    evaluation_id: str = ""
    evaluation_index: int | None = None
    stop_on_done: bool = False


@dataclass
class EvaluatorSession:
    id: str
    path: Path
    commands: list[dict[str, str]]
    payload: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    status: str = "starting"
    index: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    error: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def parse_camera_spec(value: str) -> CameraSpec:
    """Parse camera specs like 'overhead=0' or 'wrist=/dev/video2,1280x720@30'."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Camera must look like name=index_or_path[,WIDTHxHEIGHT@FPS].")

    name, rest = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Camera name cannot be empty.")

    parts = rest.split(",", 1)
    index_or_path: str | int
    raw_index = parts[0].strip()
    index_or_path = int(raw_index) if raw_index.isdigit() else raw_index

    width, height, fps = 640, 480, 30
    if len(parts) == 2:
        dims = parts[1].strip()
        if "@" in dims:
            dims, fps_str = dims.split("@", 1)
            fps = int(fps_str)
        if "x" not in dims:
            raise argparse.ArgumentTypeError("Camera dimensions must look like WIDTHxHEIGHT@FPS.")
        width_str, height_str = dims.split("x", 1)
        width, height = int(width_str), int(height_str)

    return CameraSpec(name=name, index_or_path=index_or_path, width=width, height=height, fps=fps)


def camera_spec_from_dict(raw: dict[str, Any]) -> CameraSpec:
    return CameraSpec(
        name=str(raw["name"]),
        index_or_path=raw["index_or_path"],
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 480)),
        fps=int(raw.get("fps", 30)),
        warmup_s=int(raw.get("warmup_s", 1)),
        fourcc=str(raw["fourcc"]) if raw.get("fourcc") else None,
    )


def load_dashboard_config(path: Path) -> DashboardConfig:
    raw = json.loads(path.read_text())
    robot_raw = raw.get("robot", {})
    leader_raw = raw.get("leader", {})
    board_raw = raw.get("board", {})
    model_raw = raw.get("model", {})
    return DashboardConfig(
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8766)),
        cameras=[camera_spec_from_dict(item) for item in raw.get("cameras", [])],
        robot=RobotSpec(
            type=str(robot_raw.get("type", "so101_follower")),
            port=robot_raw.get("port"),
            id=str(robot_raw.get("id", "go_follower")),
            calibrate=bool(robot_raw.get("calibrate", False)),
            configure_on_connect=bool(robot_raw.get("configure_on_connect", False)),
            urdf_path=robot_raw.get("urdf_path"),
            target_frame_name=str(robot_raw.get("target_frame_name", "gripper_frame_link")),
            rest_position={str(name): float(value) for name, value in robot_raw.get("rest_position", {}).items()},
        ),
        leader=LeaderSpec(
            type=str(leader_raw.get("type", "so101_leader")),
            port=leader_raw.get("port"),
            id=str(leader_raw.get("id", "so101_leader")),
            calibrate=bool(leader_raw.get("calibrate", False)),
        ),
        board=BoardSpec(
            camera=str(board_raw.get("camera", "overhead")),
            size=int(board_raw.get("size", 19)),
            corners_tl_tr_br_bl=board_raw.get("corners_tl_tr_br_bl"),
            camera_to_robot_rotation_degrees=int(board_raw.get("camera_to_robot_rotation_degrees", 0)),
            board_pixels=int(board_raw.get("board_pixels", 1000)),
            overlay_fisheye_k=float(board_raw.get("overlay_fisheye_k", 0.0)),
            sample_radius_ratio=float(board_raw.get("sample_radius_ratio", 0.34)),
            black_l_threshold=float(board_raw.get("black_l_threshold", 80.0)),
            white_l_threshold=float(board_raw.get("white_l_threshold", 165.0)),
            white_s_threshold=float(board_raw.get("white_s_threshold", 75.0)),
            stone_min_radius_ratio=float(board_raw.get("stone_min_radius_ratio", 0.2)),
            stone_max_radius_ratio=float(board_raw.get("stone_max_radius_ratio", 0.48)),
            stone_min_circularity=float(board_raw.get("stone_min_circularity", 0.45)),
            stone_max_snap_distance_ratio=float(board_raw.get("stone_max_snap_distance_ratio", 0.52)),
            black_grid_min_edge_score=float(board_raw.get("black_grid_min_edge_score", 0.18)),
        ),
        model=ModelSpec(
            policy_path=str(model_raw.get("policy_path", "")),
            remote_host=str(model_raw.get("remote_host", "")),
            remote_workdir=str(model_raw.get("remote_workdir", "~/Developer/lerobot")),
            remote_policy_server=str(model_raw.get("remote_policy_server", "")),
            policy_type=str(model_raw.get("policy_type", "act")),
            actions_per_chunk=int(model_raw.get("actions_per_chunk", 20)),
            policy_image_size=int(model_raw.get("policy_image_size", 224)),
            device=str(model_raw.get("device", "cuda")),
            fps=float(model_raw.get("fps", 10.0)),
            duration_s=float(model_raw.get("duration_s", 30.0)),
            gripper_deadband=float(model_raw.get("gripper_deadband", 0.0)),
            gripper_max_step=float(model_raw.get("gripper_max_step", 0.0)),
            release_local_devices=bool(model_raw.get("release_local_devices", True)),
        ),
    )


class CameraStream:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.capture: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_timestamp = 0.0
        self.latest_brightness = 0.0
        self.actual_width = 0
        self.actual_height = 0
        self.frames = 0
        self.fps_started_at = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop = threading.Event()
        self.capture = cv2.VideoCapture(self.spec.index_or_path)
        if not self.capture.isOpened():
            raise ConnectionError(f"Could not open camera {self.spec.name} at {self.spec.index_or_path!r}.")
        if self.spec.fourcc:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.spec.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def restart(self) -> dict[str, Any]:
        try:
            self.stop()
            self.latest_frame = None
            self.latest_jpeg = None
            self.latest_timestamp = 0.0
            self.latest_brightness = 0.0
            self.actual_width = 0
            self.actual_height = 0
            self.frames = 0
            self.fps_started_at = time.time()
            self.start()
            return {"ok": True, "name": self.spec.name}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            return {"ok": False, "name": self.spec.name, "error": str(exc)}

    def _loop(self) -> None:
        assert self.capture is not None
        interval = 1.0 / max(self.spec.fps, 1)
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if ok:
                self._set_frame(frame)
            time.sleep(interval * 0.25)

    def _set_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        with self.lock:
            self.latest_frame = frame
            self.latest_jpeg = encoded.tobytes()
            self.latest_timestamp = time.time()
            self.latest_brightness = brightness
            self.actual_height, self.actual_width = frame.shape[:2]
            self.frames += 1

    def jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def frame(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def status(self) -> dict[str, Any]:
        age_s = time.time() - self.latest_timestamp if self.latest_timestamp else None
        fresh = age_s is not None and age_s < 2.0
        dark = self.latest_jpeg is not None and self.latest_brightness < 18.0
        return {
            "fresh": fresh,
            "dark": dark,
            "age_s": round(age_s, 2) if age_s is not None else None,
            "brightness": round(self.latest_brightness, 1),
            "actual_width": self.actual_width,
            "actual_height": self.actual_height,
            "measured_fps": round(self.frames / max(time.time() - self.fps_started_at, 1e-6), 1),
        }


class MockCameraStream(CameraStream):
    def __init__(self, spec: CameraSpec, color: tuple[int, int, int]):
        super().__init__(spec)
        self.color = color

    def start(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"mock-camera-{self.spec.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self._make_frame()
            self._set_frame(frame)
            time.sleep(1.0 / max(self.spec.fps, 1))

    def _make_frame(self) -> np.ndarray:
        t = time.time()
        frame = np.full((self.spec.height, self.spec.width, 3), self.color, dtype=np.uint8)
        cv2.putText(
            frame,
            self.spec.name,
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 30, 35),
            2,
            cv2.LINE_AA,
        )
        cx = int(self.spec.width * (0.5 + 0.28 * math.sin(t * 0.9)))
        cy = int(self.spec.height * (0.5 + 0.22 * math.cos(t * 0.7)))
        cv2.circle(frame, (cx, cy), 32, (245, 245, 245), -1)
        cv2.circle(frame, (cx, cy), 34, (40, 45, 48), 2)
        return frame


class TelemetrySource:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def state(self) -> DashboardState:
        raise NotImplementedError


class MockTelemetrySource(TelemetrySource):
    def __init__(self):
        self.started_at = time.time()

    def state(self) -> DashboardState:
        t = time.time() - self.started_at
        ranges = [180, 90, 120, 110, 180, 100]
        centers = [0, -20, 35, 10, 0, 55]
        joints = []
        for idx, name in enumerate(DEFAULT_JOINTS):
            span = ranges[idx]
            value = centers[idx] + math.sin(t * (0.35 + idx * 0.07)) * span * 0.25
            joints.append(
                JointState(
                    name=name,
                    value=round(value, 2),
                    unit="%" if name == "gripper" else "deg",
                    min_value=0.0 if name == "gripper" else -span,
                    max_value=100.0 if name == "gripper" else span,
                )
            )

        ee = EndEffectorState(
            available=True,
            x=round(0.19 + 0.045 * math.sin(t * 0.45), 4),
            y=round(0.015 + 0.055 * math.cos(t * 0.36), 4),
            z=round(0.13 + 0.025 * math.sin(t * 0.28), 4),
            roll=round(3.0 * math.sin(t * 0.3), 2),
            pitch=round(8.0 * math.cos(t * 0.27), 2),
            yaw=round(20.0 * math.sin(t * 0.19), 2),
            source="mock",
        )
        return DashboardState(
            timestamp=time.time(),
            connected=True,
            mode="mock",
            fps=30.0,
            joints=joints,
            leader_joints=joints,
            teleop_enabled=False,
            end_effector=ee,
            note="Mock telemetry. Add --so101-port to read the follower arm.",
        )


class SO101TelemetrySource(TelemetrySource):
    def __init__(
        self,
        port: str,
        robot_id: str,
        calibrate: bool,
        configure_on_connect: bool,
        leader_port: str | None,
        leader_id: str,
        leader_calibrate: bool,
        urdf_path: str | None,
        target_frame_name: str,
        rest_position: dict[str, float],
    ):
        self.port = port
        self.robot_id = robot_id
        self.calibrate = calibrate
        self.configure_on_connect = configure_on_connect
        self.leader_port = leader_port
        self.leader_id = leader_id
        self.leader_calibrate = leader_calibrate
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.rest_position = rest_position
        self.robot: Any | None = None
        self.leader: Any | None = None
        self.kinematics: Any | None = None
        self.joint_names: list[str] = DEFAULT_JOINTS.copy()
        self.leader_joint_names: list[str] = DEFAULT_JOINTS.copy()
        self.last_state = self._unavailable_state("SO-101 telemetry not connected yet.")
        self.teleop_enabled = False
        self.teleop_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bus_lock = threading.RLock()
        self._samples = 0
        self._fps_started_at = time.time()
        self.rest_move_duration_s = 2.5
        self.teleop_start_duration_s = 2.0
        self.ramp_hz = 20.0

    def start(self) -> None:
        self._stop = threading.Event()
        self._samples = 0
        self._fps_started_at = time.time()
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            cfg = SO101FollowerConfig(port=self.port, id=self.robot_id, cameras={}, use_degrees=True)
            self.robot = SO101Follower(cfg)
            if self.configure_on_connect:
                self.robot.connect(calibrate=self.calibrate)
            else:
                self.robot.bus.connect()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self.robot = None
            self.last_state = self._unavailable_state(f"Follower telemetry unavailable on {self.port}: {exc}")
            return
        self.joint_names = list(self.robot.bus.motors.keys())

        if self.leader_port:
            result = self.reconnect_leader()
            if not result["ok"]:
                self.last_state = self._unavailable_state(
                    f"Follower connected. Leader telemetry unavailable on {self.leader_port}: {result['error']}"
                )

        if self.urdf_path:
            try:
                from lerobot.model.kinematics import RobotKinematics

                self.kinematics = RobotKinematics(
                    urdf_path=self.urdf_path,
                    target_frame_name=self.target_frame_name,
                    joint_names=self.joint_names,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                self.kinematics = None
                self.last_state.note = f"Robot connected. FK unavailable: {exc}"

        self._thread = threading.Thread(target=self._loop, name="so101-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self.teleop_enabled = False
            self.teleop_error = ""
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.robot is not None and self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)
        if self.leader is not None and self.leader.bus.is_connected:
            self.leader.bus.disconnect(disable_torque=False)
        self.robot = None
        self.leader = None

    def set_teleop_enabled(self, enabled: bool) -> dict[str, Any]:
        if not enabled:
            with self._lock:
                self.teleop_enabled = False
                self.teleop_error = ""
            return {"ok": True, "enabled": False}

        try:
            with self._lock:
                self.teleop_enabled = False
                self.teleop_error = "Moving follower to leader before teleoperation..."
            leader_joints = self._read_leader_joints()
            goals = {joint.name: joint.value for joint in leader_joints if joint.name in self.joint_names}
            if not goals:
                return {"ok": False, "error": "Leader joints are not available."}
            ramp = self._ramp_follower_to(goals, duration_s=self.teleop_start_duration_s)
            with self._lock:
                self.teleop_enabled = True
                self.teleop_error = ""
            return {"ok": True, "enabled": True, "ramp": ramp}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            with self._lock:
                self.teleop_enabled = False
                self.teleop_error = f"Teleop startup ramp failed: {exc}"
            return {"ok": False, "error": str(exc)}

    def move_follower_to_rest(self) -> dict[str, Any]:
        if self.robot is None:
            return {"ok": False, "error": "Follower is not connected."}
        if not self.rest_position:
            return {"ok": False, "error": "No robot.rest_position is configured."}

        goals = {
            name: float(value)
            for name, value in self.rest_position.items()
            if name in self.joint_names
        }
        if not goals:
            return {"ok": False, "error": "Rest position does not contain any follower joint names."}

        try:
            with self._lock:
                self.teleop_enabled = False
                self.teleop_error = "Moving follower to rest..."
            ramp = self._ramp_follower_to(goals, duration_s=self.rest_move_duration_s)
            with self._lock:
                self.teleop_error = ""
            return {"ok": True, "goals": goals, "ramp": ramp}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            with self._lock:
                self.teleop_error = f"Follower rest move failed: {exc}"
            return {"ok": False, "error": str(exc)}

    def set_rest_position_from_leader(self) -> dict[str, Any]:
        if self.leader is None:
            return {"ok": False, "error": "Leader is not connected."}

        try:
            leader_joints = self._read_leader_joints()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            return {"ok": False, "error": f"Could not read leader joints: {exc}"}

        rest_position = {
            joint.name: float(joint.value)
            for joint in leader_joints
            if joint.name in self.joint_names
        }
        if not rest_position:
            return {"ok": False, "error": "Leader joints do not match any follower joint names."}

        with self._lock:
            self.rest_position = rest_position
            self.teleop_error = ""
        return {"ok": True, "rest_position": rest_position}

    def reconnect_leader(self) -> dict[str, Any]:
        if not self.leader_port:
            return {"ok": False, "error": "No leader.port is configured."}

        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        try:
            with self._lock:
                old_leader = self.leader
                self.leader = None
                self.leader_joint_names = DEFAULT_JOINTS.copy()

            if old_leader is not None and old_leader.bus.is_connected:
                old_leader.bus.disconnect(disable_torque=False)

            leader_cfg = SO101LeaderConfig(port=self.leader_port, id=self.leader_id, use_degrees=True)
            leader = SO101Leader(leader_cfg)
            leader.bus.connect()

            with self._lock:
                self.leader = leader
                self.leader_joint_names = list(leader.bus.motors.keys())
                self.teleop_error = ""
            return {"ok": True, "port": self.leader_port, "joints": self.leader_joint_names}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            with self._lock:
                self.leader = None
                self.teleop_enabled = False
                self.teleop_error = f"Leader reconnect failed: {exc}"
            return {"ok": False, "error": str(exc)}

    def refresh_devices(self) -> dict[str, Any]:
        self.stop()
        self.start()
        follower_connected = bool(self.robot is not None and self.robot.bus.is_connected)
        leader_connected = bool(self.leader is not None and self.leader.bus.is_connected)
        leader_result = (
            {"ok": True, "port": self.leader_port, "joints": self.leader_joint_names}
            if leader_connected
            else {"ok": False, "error": self.last_state.note}
        )
        return {
            "ok": follower_connected or leader_result["ok"],
            "follower": {"ok": follower_connected, "port": self.port},
            "leader": leader_result,
        }

    def _unavailable_state(self, note: str) -> DashboardState:
        return DashboardState(
            timestamp=time.time(),
            connected=False,
            mode="so101",
            fps=0.0,
            joints=[],
            leader_joints=[],
            teleop_enabled=False,
            end_effector=EndEffectorState(available=False, source="unavailable"),
            note=note,
        )

    def _loop(self) -> None:
        assert self.robot is not None
        while not self._stop.is_set():
            try:
                with self._bus_lock:
                    obs = self.robot.get_observation()
                joints = self._joints_from_observation(obs)
                leader_joints = self._read_leader_joints()
                if self.teleop_enabled:
                    self._teleoperate_from_leader(leader_joints)
                ee = self._end_effector_from_joints(joints)
                self._samples += 1
                elapsed = max(time.time() - self._fps_started_at, 1e-6)
                note = "SO-101 follower and leader connected." if self.leader else "SO-101 telemetry connected."
                if self.teleop_enabled:
                    note = "Teleoperation enabled."
                if self.teleop_error:
                    note = self.teleop_error
                state = DashboardState(
                    timestamp=time.time(),
                    connected=True,
                    mode="so101",
                    fps=round(self._samples / elapsed, 1),
                    joints=joints,
                    leader_joints=leader_joints,
                    teleop_enabled=self.teleop_enabled,
                    end_effector=ee,
                    note=note,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                state = DashboardState(
                    timestamp=time.time(),
                    connected=False,
                    mode="so101",
                    fps=0.0,
                    joints=[],
                    leader_joints=[],
                    teleop_enabled=self.teleop_enabled,
                    end_effector=EndEffectorState(available=False, source="error"),
                    note=str(exc),
                )
            with self._lock:
                self.last_state = state
            time.sleep(1 / 30)

    def _joints_from_observation(self, obs: dict[str, Any]) -> list[JointState]:
        joints = []
        for name in self.joint_names:
            value = float(obs.get(f"{name}.pos", 0.0))
            joints.append(
                JointState(
                    name=name,
                    value=round(value, 2),
                    unit="%" if name == "gripper" else "deg",
                    min_value=0.0 if name == "gripper" else -180.0,
                    max_value=100.0 if name == "gripper" else 180.0,
                )
            )
        return joints

    def _read_leader_joints(self) -> list[JointState]:
        with self._lock:
            leader = self.leader
            leader_joint_names = self.leader_joint_names.copy()
        if leader is None:
            return []
        with self._bus_lock:
            action = leader.bus.sync_read("Present_Position")
        return [
            JointState(
                name=name,
                value=round(float(action.get(name, 0.0)), 2),
                unit="%" if name == "gripper" else "deg",
                min_value=0.0 if name == "gripper" else -180.0,
                max_value=100.0 if name == "gripper" else 180.0,
            )
            for name in leader_joint_names
        ]

    def _teleoperate_from_leader(self, leader_joints: list[JointState]) -> None:
        if self.robot is None or self.leader is None:
            self.teleop_error = "Teleoperation requested, but follower or leader is not connected."
            self.teleop_enabled = False
            return
        goals = {joint.name: joint.value for joint in leader_joints}
        try:
            with self._bus_lock:
                self.robot.bus.sync_write("Goal_Position", goals)
            self.teleop_error = ""
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self.teleop_error = f"Teleoperation write failed: {exc}"
            self.teleop_enabled = False

    def _ramp_follower_to(self, goals: dict[str, float], duration_s: float) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Follower is not connected.")
        if not goals:
            return {"steps": 0, "duration_s": 0.0}

        steps = max(1, int(duration_s * self.ramp_hz))
        interval_s = 1.0 / max(self.ramp_hz, 1.0)
        with self._bus_lock:
            obs = self.robot.get_observation()
            start = {
                name: float(obs.get(f"{name}.pos", goals[name]))
                for name in goals
            }
            for step in range(1, steps + 1):
                ratio = step / steps
                eased = ratio * ratio * (3.0 - 2.0 * ratio)
                intermediate = {
                    name: start[name] + (target - start[name]) * eased
                    for name, target in goals.items()
                }
                self.robot.bus.sync_write("Goal_Position", intermediate)
                if step < steps:
                    time.sleep(interval_s)
        return {"steps": steps, "duration_s": round(duration_s, 2), "goals": goals}

    def _end_effector_from_joints(self, joints: list[JointState]) -> EndEffectorState:
        if self.kinematics is None:
            return EndEffectorState(available=False, source="provide --urdf-path for FK")
        joint_values = np.array([joint.value for joint in joints], dtype=np.float64)
        transform = self.kinematics.forward_kinematics(joint_values)
        roll, pitch, yaw = rotation_matrix_to_rpy_deg(transform[:3, :3])
        return EndEffectorState(
            available=True,
            x=round(float(transform[0, 3]), 4),
            y=round(float(transform[1, 3]), 4),
            z=round(float(transform[2, 3]), 4),
            roll=round(roll, 2),
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            source="forward_kinematics",
        )

    def state(self) -> DashboardState:
        with self._lock:
            return self.last_state


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
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
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


class DashboardApp:
    def __init__(
        self,
        cameras: list[CameraStream],
        telemetry: TelemetrySource,
        board: BoardSpec,
        model: ModelSpec,
        annotation_dir: Path,
        config_path: Path | None = None,
        recording_dir: Path | None = None,
        model_rollout_dir: Path | None = None,
        model_evaluation_dir: Path | None = None,
        synthetic_recording_dir: Path | None = None,
    ):
        self.cameras = {camera.spec.name: camera for camera in cameras}
        self.telemetry = telemetry
        self.board = board
        self.model = model
        self.annotation_dir = annotation_dir
        self.recording_dir = recording_dir if recording_dir is not None else annotation_dir.parent / "recordings"
        self.model_rollout_dir = (
            model_rollout_dir if model_rollout_dir is not None else annotation_dir.parent / MODEL_ROLLOUTS_DIR_NAME
        )
        self.model_evaluation_dir = (
            model_evaluation_dir
            if model_evaluation_dir is not None
            else annotation_dir.parent / MODEL_EVALUATIONS_DIR_NAME
        )
        self.synthetic_recording_dir = synthetic_recording_dir or DEFAULT_SYNTHETIC_RECORDING_DIR
        self.config_path = config_path
        self.board_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.board_baseline: BoardState | None = None
        self.board_current: BoardState | None = None
        self.board_delta: dict[str, Any] | None = None
        self.board_live_state: BoardState | None = None
        self.board_message = "No board snapshots captured yet."
        self.active_recording: RecordingSession | None = None
        self.recording_message = "Ready to record."
        self.active_model_run: ModelRunSession | None = None
        self.last_model_run: ModelRunSession | None = None
        self.model_message = "Ready for policy rollout."
        self.evaluator_lock = threading.Lock()
        self.active_evaluator: EvaluatorSession | None = None
        self.last_evaluator: EvaluatorSession | None = None
        self.evaluator_message = "Ready to evaluate."
        self._migrate_model_rollouts_from_recordings()

    def _migrate_model_rollouts_from_recordings(self) -> None:
        if not self.recording_dir.is_dir():
            return
        for metadata_path in sorted(self.recording_dir.glob("*/metadata.json")):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("run_type") != "model_inference":
                continue
            source = metadata_path.parent
            target = self._unique_recording_path(self.model_rollout_dir / source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)

    def start(self) -> None:
        for camera in self.cameras.values():
            camera.start()
        self.telemetry.start()

    def stop(self) -> None:
        self.stop_evaluator()
        self.stop_model_run()
        self.telemetry.stop()
        for camera in self.cameras.values():
            camera.stop()

    def set_teleop_enabled(self, enabled: bool) -> dict[str, Any]:
        if not hasattr(self.telemetry, "set_teleop_enabled"):
            return {"ok": False, "error": "Telemetry source does not support teleoperation."}
        result = self.telemetry.set_teleop_enabled(enabled)
        if isinstance(result, dict):
            return result
        return {"ok": True, "enabled": enabled}

    def move_follower_to_rest(self) -> dict[str, Any]:
        if not hasattr(self.telemetry, "move_follower_to_rest"):
            return {"ok": False, "error": "Telemetry source does not support follower rest moves."}
        return self.telemetry.move_follower_to_rest()

    def set_rest_position_from_leader(self) -> dict[str, Any]:
        if not hasattr(self.telemetry, "set_rest_position_from_leader"):
            return {"ok": False, "error": "Telemetry source does not support setting rest from leader."}
        result = self.telemetry.set_rest_position_from_leader()
        if result.get("ok") and isinstance(result.get("rest_position"), dict):
            self._persist_rest_position(result["rest_position"])
        return result

    def _persist_rest_position(self, rest_position: dict[str, float]) -> None:
        if self.config_path is None:
            return
        raw = json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.is_file() else {}
        raw.setdefault("robot", {})["rest_position"] = rest_position
        self.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    def reconnect_leader(self) -> dict[str, Any]:
        if not hasattr(self.telemetry, "reconnect_leader"):
            return {"ok": False, "error": "Telemetry source does not support leader reconnect."}
        return self.telemetry.reconnect_leader()

    def refresh_devices(self) -> dict[str, Any]:
        with self.model_lock:
            if self.active_model_run is not None:
                return {"ok": False, "error": "A model rollout is active."}
        camera_results = [camera.restart() for camera in self.cameras.values()]
        if hasattr(self.telemetry, "refresh_devices"):
            telemetry_result = self.telemetry.refresh_devices()
        else:
            telemetry_result = {"ok": False, "error": "Telemetry source does not support refresh."}
        ok = all(item["ok"] for item in camera_results) and bool(telemetry_result.get("ok"))
        return {"ok": ok, "cameras": camera_results, "telemetry": telemetry_result}

    def start_model_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.model_lock:
            if self.active_model_run is not None:
                raise ValueError("A model rollout is already active.")
        with self.evaluator_lock:
            if self.active_evaluator is not None and not payload.get("evaluation_id"):
                raise ValueError("An evaluator session is active.")
        with self.recording_lock:
            if self.active_recording is not None:
                raise ValueError("Stop the recording before starting a model rollout.")

        coord = self._normalize_go_coord(str(payload.get("coord", "")))
        color = str(payload.get("color", "white")).strip().lower() or "white"
        if color not in {"white", "black"}:
            raise ValueError("Stone color must be 'white' or 'black'.")

        policy_path = str(payload.get("policy_path") or self.model.policy_path).strip()
        if not policy_path:
            raise ValueError("Set a policy path before starting rollout.")
        remote_host = str(payload.get("remote_host") if payload.get("remote_host") is not None else self.model.remote_host).strip()
        remote_workdir = str(
            payload.get("remote_workdir") if payload.get("remote_workdir") is not None else self.model.remote_workdir
        ).strip() or "~/Developer/lerobot"
        remote_policy_server = str(
            payload.get("remote_policy_server")
            if payload.get("remote_policy_server") is not None
            else self.model.remote_policy_server
        ).strip()
        policy_type = str(payload.get("policy_type") or self.model.policy_type).strip() or "act"
        actions_per_chunk = int(payload.get("actions_per_chunk") or self.model.actions_per_chunk)
        policy_image_size = int(payload.get("policy_image_size") or self.model.policy_image_size)
        device = str(payload.get("device") or self.model.device).strip() or "cpu"
        fps = float(payload.get("fps") or self.model.fps)
        duration_s = float(payload.get("duration_s") or self.model.duration_s)
        gripper_deadband = float(payload.get("gripper_deadband") or self.model.gripper_deadband)
        gripper_max_step = float(payload.get("gripper_max_step") or self.model.gripper_max_step)
        stop_on_done = bool(payload.get("stop_on_done", False))
        if actions_per_chunk <= 0:
            raise ValueError("Actions per chunk must be positive.")
        if policy_image_size <= 0:
            raise ValueError("Policy image size must be positive.")
        if fps <= 0:
            raise ValueError("FPS must be positive.")
        if duration_s <= 0:
            raise ValueError("Duration must be positive.")
        if gripper_deadband < 0:
            raise ValueError("Gripper deadband must be non-negative.")
        if gripper_max_step < 0:
            raise ValueError("Gripper max step must be non-negative.")

        task = f"place {color} stone at {coord}"
        preview_dir = Path("examples/go_board/runtime/model_preview") / time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir, baseline, baseline_image = self._create_model_rollout_dir(
            run_id=preview_dir.name,
            coord=coord,
            color=color,
            parent_dir=Path(str(payload.get("rollout_parent_dir"))) if payload.get("rollout_parent_dir") else None,
        )
        run = ModelRunSession(
            id=preview_dir.name,
            coord=coord,
            color=color,
            task=task,
            policy_path=policy_path,
            remote_host=remote_host,
            remote_workdir=remote_workdir,
            remote_policy_server=remote_policy_server,
            policy_type=policy_type,
            actions_per_chunk=actions_per_chunk,
            policy_image_size=policy_image_size,
            device=device,
            fps=fps,
            duration_s=duration_s,
            gripper_deadband=gripper_deadband,
            gripper_max_step=gripper_max_step,
            started_at=time.time(),
            preview_dir=preview_dir,
            rollout_dir=rollout_dir,
            baseline=baseline,
            baseline_image=baseline_image,
            evaluation_id=str(payload.get("evaluation_id", "")),
            evaluation_index=int(payload["evaluation_index"]) if payload.get("evaluation_index") is not None else None,
            stop_on_done=stop_on_done,
        )
        run.command = self._model_run_command(run)
        run.command_display = self._command_display(run.command)

        with self.model_lock:
            self.active_model_run = run
            self.last_model_run = run
            self.model_message = f"Starting policy rollout for {coord}..."
        run.thread = threading.Thread(target=self._model_run_loop, args=(run,), name=f"go-policy-{run.id}", daemon=True)
        run.thread.start()
        return {"ok": True, "model_run": self._model_run_status(run)}

    def _empty_board_state(self) -> BoardState:
        corners = self.board.corners_tl_tr_br_bl or []
        return BoardState(
            board_size=self.board.size,
            corners_tl_tr_br_bl=corners,
            stones=[],
            occupied={},
            summary={"black": 0, "white": 0, "total": 0},
        )

    def _create_model_rollout_dir(
        self,
        run_id: str,
        coord: str,
        color: str,
        parent_dir: Path | None = None,
    ) -> tuple[Path, BoardState, str | None]:
        root = parent_dir or self.model_rollout_dir
        rollout_dir = self._unique_recording_path(
            root / f"{run_id}_model_{color}_to_{coord.lower()}"
        )
        rollout_dir.mkdir(parents=True, exist_ok=False)
        (rollout_dir / "frames").mkdir()
        for camera_name in self.cameras:
            (rollout_dir / "frames" / camera_name).mkdir(parents=True, exist_ok=True)

        baseline = self._empty_board_state()
        baseline_image = None
        try:
            baseline = self._detect_board_state_from_camera()
        except Exception:
            baseline = self._empty_board_state()
        board_camera = self.cameras.get(self.board.camera)
        baseline_frame = board_camera.frame() if board_camera is not None else None
        if baseline_frame is not None:
            baseline_image = "baseline.jpg"
            cv2.imwrite(str(rollout_dir / baseline_image), baseline_frame)

        metadata = self._model_rollout_metadata(
            run=None,
            rollout_dir=rollout_dir,
            status="recording",
            baseline=baseline,
            baseline_image=baseline_image,
            final=None,
            final_image=None,
            delta=None,
            move_name=f"model_{color}_to_{coord.lower()}",
            samples=0,
        )
        (rollout_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return rollout_dir, baseline, baseline_image

    def stop_model_run(self) -> dict[str, Any]:
        with self.model_lock:
            run = self.active_model_run
            if run is None:
                return {"ok": True, "model_run": self._model_run_status(self.last_model_run)}
            run.stop_event.set()
            process = run.process
            self.model_message = "Stopping policy rollout..."

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
        return {"ok": True, "model_run": self._model_run_status(run)}

    def _model_run_loop(self, run: ModelRunSession) -> None:
        release_local_devices = (not run.remote_host or bool(run.remote_policy_server)) and self.model.release_local_devices
        try:
            self.set_teleop_enabled(False)
            if release_local_devices:
                self._stop_dashboard_devices_for_model()
            with self.model_lock:
                run.status = "running"
                self.model_message = f"Policy rollout running: {run.task}."
            run.process = subprocess.Popen(
                run.command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert run.process.stdout is not None
            for line in run.process.stdout:
                with self.model_lock:
                    run.log_tail.append(line.rstrip())
                    run.log_tail = run.log_tail[-80:]
                if run.stop_event.is_set():
                    break
            if run.stop_event.is_set() and run.process.poll() is None:
                run.process.terminate()
            run.returncode = run.process.wait(timeout=5.0)
            with self.model_lock:
                if run.stop_event.is_set():
                    run.status = "stopped"
                    self.model_message = "Policy rollout stopped."
                elif run.returncode == 0:
                    run.status = "complete"
                    self.model_message = f"Policy rollout finished for {run.coord}."
                else:
                    run.status = "error"
                    run.error = f"Rollout exited with code {run.returncode}."
                    self.model_message = run.error
        except Exception as exc:
            with self.model_lock:
                run.status = "error"
                run.error = str(exc)
                self.model_message = f"Policy rollout error: {exc}"
        finally:
            if release_local_devices:
                self._restart_dashboard_devices_after_model(run)
            self._finalize_model_rollout(run)
            with self.model_lock:
                if self.active_model_run is run:
                    self.active_model_run = None
                self.last_model_run = run

    def _stop_dashboard_devices_for_model(self) -> None:
        self.telemetry.stop()
        for camera in self.cameras.values():
            camera.stop()

    def _restart_dashboard_devices_after_model(self, run: ModelRunSession) -> None:
        try:
            for camera in self.cameras.values():
                camera.start()
            self.telemetry.start()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            with self.model_lock:
                run.log_tail.append(f"Dashboard device restart failed: {exc}")
                self.model_message = f"{self.model_message} Dashboard refresh failed: {exc}"

    def _model_run_command(self, run: ModelRunSession) -> list[str]:
        config_path = self.config_path or Path("examples/go_board/dashboard_config.json")
        rollout_args = [
            f"--config={config_path}",
            f"--policy-path={run.policy_path}",
            f"--coord={run.coord}",
            f"--color={run.color}",
            f"--task={run.task}",
            f"--duration={run.duration_s}",
            f"--fps={run.fps}",
            f"--device={run.device}",
            "--timing-warn-threshold=0.05",
            f"--gripper-deadband={run.gripper_deadband}",
            f"--gripper-max-step={run.gripper_max_step}",
            "--done-stable-frames=5",
        ]
        if run.remote_policy_server:
            rollout_args.extend(
                [
                    f"--remote-policy-server={run.remote_policy_server}",
                    f"--policy-type={run.policy_type}",
                    f"--actions-per-chunk={run.actions_per_chunk}",
                    f"--policy-image-size={run.policy_image_size}",
                ]
            )
        if run.stop_on_done:
            rollout_args.append("--stop-on-done")
        if run.preview_dir is not None:
            rollout_args.append(f"--preview-dir={run.preview_dir}")
        if run.rollout_dir is not None:
            rollout_args.append(f"--recording-dir={run.rollout_dir}")
        if run.remote_host and not run.remote_policy_server:
            remote_command = shlex.join(
                ["uv", "run", "python", "examples/go_board/rollout_with_target_overlay.py", *rollout_args]
            )
            remote = f"cd {shlex.quote(run.remote_workdir)} && {remote_command}"
            return ["ssh", run.remote_host, remote]
        return [sys.executable, "examples/go_board/rollout_with_target_overlay.py", *rollout_args]

    def model_preview_jpeg(self, run_id: str, camera_name: str, rotate: str = "") -> bytes | None:
        safe_run_id = Path(run_id).name
        safe_camera = Path(camera_name).stem
        preview_path = Path("examples/go_board/runtime/model_preview") / safe_run_id / f"{safe_camera}.jpg"
        if not preview_path.is_file():
            return None
        data = preview_path.read_bytes()
        if rotate != "left":
            return data
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, encoded = cv2.imencode(".jpg", rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return encoded.tobytes() if ok else None

    def _target_metadata(self, coord: str, color: str) -> dict[str, Any]:
        columns = GO_COLUMNS if self.board.size == 19 else "".join(chr(ord("A") + i) for i in range(self.board.size))
        normalized = self._normalize_go_coord(coord)
        return {
            "coord": normalized,
            "row": int(normalized[1:]) - 1,
            "col": columns.index(normalized[0]),
            "color": color,
        }

    def _model_rollout_metadata(
        self,
        run: ModelRunSession | None,
        rollout_dir: Path,
        status: str,
        baseline: BoardState,
        baseline_image: str | None,
        final: BoardState | None,
        final_image: str | None,
        delta: dict[str, Any] | None,
        move_name: str,
        samples: int,
        rest_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        camera_meta = {
            name: {
                "width": camera.actual_width or camera.spec.width,
                "height": camera.actual_height or camera.spec.height,
                "fps": camera.spec.fps,
            }
            for name, camera in self.cameras.items()
        }
        model_meta = None
        target = None
        started_at = time.time()
        if run is not None:
            started_at = run.started_at
            target = self._target_metadata(run.coord, run.color)
            model_meta = {
                "id": run.id,
                "coord": run.coord,
                "color": run.color,
                "task": run.task,
                "policy_path": run.policy_path,
                "remote_policy_server": run.remote_policy_server,
                "policy_type": run.policy_type,
                "actions_per_chunk": run.actions_per_chunk,
                "policy_image_size": run.policy_image_size,
                "device": run.device,
                "fps": run.fps,
                "duration_s": run.duration_s,
                "gripper_deadband": run.gripper_deadband,
                "gripper_max_step": run.gripper_max_step,
                "command": run.command_display,
                "returncode": run.returncode,
                "error": run.error,
                "log_tail": run.log_tail[-80:],
            }
            if run.evaluation_id:
                model_meta["evaluation_id"] = run.evaluation_id
                model_meta["evaluation_index"] = run.evaluation_index
        metadata = {
            "schema_version": 1,
            "id": rollout_dir.name,
            "name": rollout_dir.name,
            "status": status,
            "run_type": "model_inference",
            "started_at": started_at,
            "ended_at": time.time() if status in {"complete", "error", "stopped"} else None,
            "sample_hz": run.fps if run is not None else self.model.fps,
            "samples": samples,
            "move_name": move_name,
            "baseline_image": baseline_image,
            "final_image": final_image,
            "cameras": camera_meta,
            "board": {
                "camera": self.board.camera,
                "size": self.board.size,
                "corners_tl_tr_br_bl": self.board.corners_tl_tr_br_bl,
                "camera_to_robot_rotation_degrees": self.board.camera_to_robot_rotation_degrees,
                "overlay_fisheye_k": self.board.overlay_fisheye_k,
                "baseline": board_state_to_jsonable(baseline),
                "final": None if final is None else board_state_to_jsonable(final),
                "delta": delta,
                "target": target,
                "task_state": None if delta is None else self._task_state_from_delta(delta, target),
            },
            "teleop_started": False,
            "rest_result": rest_result,
            "error": run.error if run is not None else "",
            "model_run": model_meta,
        }
        if run is not None and run.evaluation_id:
            metadata["evaluation"] = {
                "id": run.evaluation_id,
                "index": run.evaluation_index,
                "stop_on_done": run.stop_on_done,
            }
        return metadata

    def _finalize_model_rollout(self, run: ModelRunSession) -> None:
        if run.rollout_dir is None or run.baseline is None:
            return
        rollout_dir = run.rollout_dir
        if not rollout_dir.is_dir():
            return

        sample_camera = self.board.camera if (rollout_dir / "frames" / self.board.camera).is_dir() else ""
        if not sample_camera:
            camera_dirs = [path for path in (rollout_dir / "frames").iterdir() if path.is_dir()]
            sample_camera = camera_dirs[0].name if camera_dirs else self.board.camera
        samples = len(list((rollout_dir / "frames" / sample_camera).glob("*.jpg")))

        final = None
        final_image = None
        delta_json = None
        move_name = f"model_{run.color}_to_{run.coord.lower()}"
        rest_result = None
        try:
            final = self._detect_board_state_from_camera()
            board_camera = self.cameras.get(self.board.camera)
            final_frame = board_camera.frame() if board_camera is not None else None
            if final_frame is not None:
                final_image = "final.jpg"
                cv2.imwrite(str(rollout_dir / final_image), final_frame)
            delta_json = board_delta_to_jsonable(delta_between_board_states(run.baseline, final))
            detected_move = self._move_name_from_delta(delta_json)
            if detected_move != "recording":
                move_name = f"model_{detected_move}"
        except Exception as exc:
            run.log_tail.append(f"Model rollout finalization warning: {exc}")

        if run.returncode == 0 and not run.stop_event.is_set():
            status = "complete"
        elif run.stop_event.is_set():
            status = "stopped"
        else:
            status = "error"

        desired_dir = rollout_dir.parent / f"{run.id}_{move_name}"
        final_dir = rollout_dir if desired_dir == rollout_dir else self._unique_recording_path(desired_dir)
        if final_dir != rollout_dir:
            rollout_dir.rename(final_dir)
            rollout_dir = final_dir
            run.rollout_dir = final_dir

        metadata = self._model_rollout_metadata(
            run=run,
            rollout_dir=rollout_dir,
            status=status,
            baseline=run.baseline,
            baseline_image=run.baseline_image,
            final=final,
            final_image=final_image,
            delta=delta_json,
            move_name=move_name,
            samples=samples,
            rest_result=rest_result,
        )
        (rollout_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        run.saved_rollout_name = rollout_dir.name

    def _rollout_camera_config(self) -> dict[str, Any]:
        return {
            name: {
                "type": "opencv",
                "index_or_path": camera.spec.index_or_path,
                "width": camera.spec.width,
                "height": camera.spec.height,
                "fps": camera.spec.fps,
            }
            for name, camera in self.cameras.items()
        }

    def _robot_type(self) -> str:
        return "so101_follower"

    def _robot_port(self) -> str:
        if hasattr(self.telemetry, "port"):
            return str(self.telemetry.port)
        raise ValueError("No SO-101 follower port is configured for rollout.")

    def _robot_id(self) -> str:
        if hasattr(self.telemetry, "robot_id"):
            return str(self.telemetry.robot_id)
        return "go_follower"

    @staticmethod
    def _command_display(command: list[str]) -> str:
        return shlex.join(command)

    def _model_run_status(self, run: ModelRunSession | None) -> dict[str, Any] | None:
        if run is None:
            return None
        return {
            "id": run.id,
            "coord": run.coord,
            "color": run.color,
            "task": run.task,
            "policy_path": run.policy_path,
            "remote_host": run.remote_host,
            "remote_workdir": run.remote_workdir,
            "remote_policy_server": run.remote_policy_server,
            "policy_type": run.policy_type,
            "actions_per_chunk": run.actions_per_chunk,
            "policy_image_size": run.policy_image_size,
            "device": run.device,
            "fps": run.fps,
            "duration_s": run.duration_s,
            "gripper_deadband": run.gripper_deadband,
            "gripper_max_step": run.gripper_max_step,
            "elapsed_s": round(time.time() - run.started_at, 2),
            "status": run.status,
            "command": run.command_display,
            "returncode": run.returncode,
            "error": run.error,
            "log_tail": run.log_tail[-80:],
            "preview": {
                "run_id": run.preview_dir.name if run.preview_dir is not None else run.id,
                "available": run.preview_dir is not None and run.preview_dir.is_dir(),
            },
            "saved_rollout": run.saved_rollout_name,
            "evaluation_id": run.evaluation_id,
            "evaluation_index": run.evaluation_index,
            "stop_on_done": run.stop_on_done,
        }

    def model_run_json(self) -> dict[str, Any]:
        with self.model_lock:
            return {
                "active": self._model_run_status(self.active_model_run),
                "last": self._model_run_status(self.last_model_run),
                "message": self.model_message,
                "defaults": asdict(self.model),
            }

    def start_evaluator(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.evaluator_lock:
            if self.active_evaluator is not None:
                raise ValueError("An evaluator session is already active.")
        with self.model_lock:
            if self.active_model_run is not None:
                raise ValueError("Stop the active model rollout before starting evaluation.")
        with self.recording_lock:
            if self.active_recording is not None:
                raise ValueError("Stop the recording before starting evaluation.")

        commands = self._parse_evaluator_commands(payload.get("commands"))
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        session_dir = self._unique_recording_path(self.model_evaluation_dir / f"{run_id}_ood_eval")
        session_dir.mkdir(parents=True, exist_ok=False)
        session = EvaluatorSession(
            id=session_dir.name,
            path=session_dir,
            commands=commands,
            payload=dict(payload),
            started_at=time.time(),
            status="running",
            message="Starting evaluation...",
        )
        self._write_evaluator_summary(session)
        with self.evaluator_lock:
            self.active_evaluator = session
            self.last_evaluator = session
            self.evaluator_message = session.message
        session.thread = threading.Thread(
            target=self._evaluator_loop,
            args=(session,),
            name=f"go-evaluator-{session.id}",
            daemon=True,
        )
        session.thread.start()
        return {"ok": True, "evaluator": self._evaluator_status(session)}

    def stop_evaluator(self) -> dict[str, Any]:
        with self.evaluator_lock:
            session = self.active_evaluator
            if session is None:
                return {"ok": True, "evaluator": self._evaluator_status(self.last_evaluator)}
            session.stop_event.set()
            self.evaluator_message = "Stopping evaluator..."
        self.stop_model_run()
        return {"ok": True, "evaluator": self._evaluator_status(session)}

    def evaluator_json(self) -> dict[str, Any]:
        with self.evaluator_lock:
            return {
                "active": self._evaluator_status(self.active_evaluator),
                "last": self._evaluator_status(self.last_evaluator),
                "message": self.evaluator_message,
                "defaults": {
                    **asdict(self.model),
                    "commands": DEFAULT_EVALUATION_SEQUENCE,
                    "duration_s": 30,
                },
            }

    def _parse_evaluator_commands(self, raw_commands: Any) -> list[dict[str, str]]:
        commands = raw_commands if isinstance(raw_commands, list) and raw_commands else DEFAULT_EVALUATION_SEQUENCE
        parsed: list[dict[str, str]] = []
        for raw in commands:
            if not isinstance(raw, dict):
                raise ValueError("Each evaluator command must be an object.")
            coord = self._normalize_go_coord(str(raw.get("coord", "")))
            color = str(raw.get("color", "white")).strip().lower() or "white"
            if color not in {"black", "white"}:
                raise ValueError("Evaluator command color must be 'black' or 'white'.")
            parsed.append({"coord": coord, "color": color})
        if len(parsed) != 10:
            raise ValueError("Evaluator sequence must contain exactly 10 commands.")
        return parsed

    def _evaluator_loop(self, session: EvaluatorSession) -> None:
        try:
            for index, command in enumerate(session.commands):
                if session.stop_event.is_set():
                    session.status = "stopped"
                    break
                session.index = index
                session.message = f"Running {index + 1}/{len(session.commands)}: {command['color']} to {command['coord']}."
                with self.evaluator_lock:
                    self.evaluator_message = session.message
                run_payload = dict(session.payload)
                run_payload.update(
                    {
                        "coord": command["coord"],
                        "color": command["color"],
                        "duration_s": 30,
                        "stop_on_done": True,
                        "rollout_parent_dir": str(session.path),
                        "evaluation_id": session.id,
                        "evaluation_index": index,
                    }
                )
                self.start_model_run(run_payload)
                with self.model_lock:
                    run = self.active_model_run
                while run is not None:
                    if session.stop_event.is_set():
                        self.stop_model_run()
                    with self.model_lock:
                        if self.active_model_run is not run:
                            break
                    time.sleep(0.2)

                rest_result = self.move_follower_to_rest()
                attempt = self._evaluator_attempt_from_run(index, command, run, rest_result)
                session.attempts.append(attempt)
                self._write_evaluator_summary(session)
                with self.evaluator_lock:
                    self.evaluator_message = (
                        f"Completed {len(session.attempts)}/{len(session.commands)}: "
                        f"{self._evaluator_success_count(session)}/{len(session.attempts)} succeeded."
                    )
                if session.stop_event.is_set():
                    session.status = "stopped"
                    break
            else:
                session.status = "complete"
                session.message = (
                    f"Evaluation complete: {self._evaluator_success_count(session)}/{len(session.commands)} succeeded."
                )
        except Exception as exc:
            session.status = "error"
            session.error = str(exc)
            session.message = f"Evaluator error: {exc}"
        finally:
            session.ended_at = time.time()
            self._write_evaluator_summary(session)
            with self.evaluator_lock:
                if self.active_evaluator is session:
                    self.active_evaluator = None
                self.last_evaluator = session
                self.evaluator_message = session.message or self.evaluator_message

    def _evaluator_attempt_from_run(
        self,
        index: int,
        command: dict[str, str],
        run: ModelRunSession | None,
        rest_result: dict[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        rollout_name = ""
        rollout_path = run.rollout_dir if run is not None else None
        if rollout_path is not None and (rollout_path / "metadata.json").is_file():
            metadata = json.loads((rollout_path / "metadata.json").read_text(encoding="utf-8"))
            rollout_name = rollout_path.name
        board = metadata.get("board", {})
        task_state = board.get("task_state") or self._task_state_from_delta(board.get("delta"), board.get("target"))
        timed_out = bool(run is not None and run.returncode == 0 and not task_state.get("done") and run.duration_s >= 30)
        attempt = {
            "index": index,
            "coord": command["coord"],
            "color": command["color"],
            "success": bool(task_state.get("done")),
            "status": "success" if task_state.get("done") else "failed",
            "reason": task_state.get("reason", ""),
            "timed_out": timed_out,
            "rollout": rollout_name,
            "returncode": None if run is None else run.returncode,
            "rest": rest_result,
        }
        if rollout_path is not None and metadata:
            metadata["evaluation_attempt"] = attempt
            metadata["rest_result"] = rest_result
            (rollout_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return attempt

    @staticmethod
    def _evaluator_success_count(session: EvaluatorSession) -> int:
        return sum(1 for attempt in session.attempts if attempt.get("success"))

    def _write_evaluator_summary(self, session: EvaluatorSession) -> None:
        payload = {
            "schema_version": 1,
            "id": session.id,
            "status": session.status,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "commands": session.commands,
            "attempts": session.attempts,
            "successes": self._evaluator_success_count(session),
            "failures": len(session.attempts) - self._evaluator_success_count(session),
            "message": session.message,
            "error": session.error,
        }
        (session.path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _evaluator_status(self, session: EvaluatorSession | None) -> dict[str, Any] | None:
        if session is None:
            return None
        return {
            "id": session.id,
            "directory": str(session.path),
            "status": session.status,
            "index": session.index,
            "total": len(session.commands),
            "elapsed_s": round(time.time() - session.started_at, 2),
            "commands": session.commands,
            "attempts": session.attempts,
            "successes": self._evaluator_success_count(session),
            "failures": len(session.attempts) - self._evaluator_success_count(session),
            "message": session.message,
            "error": session.error,
        }

    def _normalize_go_coord(self, value: str) -> str:
        text = value.strip().upper()
        if len(text) < 2:
            raise ValueError("Coordinate must look like Q16.")
        columns = GO_COLUMNS if self.board.size == 19 else "".join(chr(ord("A") + i) for i in range(self.board.size))
        col = columns.find(text[0])
        try:
            row = int(text[1:])
        except ValueError as exc:
            raise ValueError("Coordinate must use a letter plus row number, for example Q16.") from exc
        if col < 0 or row < 1 or row > self.board.size:
            raise ValueError(f"Coordinate must be on a {self.board.size}x{self.board.size} Go board.")
        return f"{columns[col]}{row}"

    def start_recording(self) -> dict[str, Any]:
        with self.model_lock:
            if self.active_model_run is not None:
                raise ValueError("Stop the model rollout before starting a recording.")
        with self.recording_lock:
            if self.active_recording is not None:
                raise ValueError("A recording is already active.")

        baseline = self._detect_board_state_from_camera()
        board_camera = self.cameras.get(self.board.camera)
        if board_camera is None:
            raise ValueError(f"Board camera '{self.board.camera}' is not configured.")
        baseline_frame = board_camera.frame()
        if baseline_frame is None:
            raise ValueError(f"Board camera '{self.board.camera}' has no frame yet.")

        started_at = time.time()
        recording_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
        session_dir = self._unique_recording_path(self.recording_dir / f"{recording_id}_recording")
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "frames").mkdir()
        for camera_name in self.cameras:
            (session_dir / "frames" / camera_name).mkdir(parents=True, exist_ok=True)

        baseline_image = "baseline.jpg"
        cv2.imwrite(str(session_dir / baseline_image), baseline_frame)
        session = RecordingSession(
            id=recording_id,
            path=session_dir,
            started_at=started_at,
            baseline=baseline,
            baseline_image=baseline_image,
            sample_hz=10.0,
            status="recording",
        )
        self._write_recording_metadata(session, status="recording")

        with self.board_lock:
            self.board_baseline = baseline
            self.board_current = None
            self.board_delta = None
            self.board_message = f"Recording baseline captured with {baseline.summary['total']} stones."
        with self.recording_lock:
            self.active_recording = session
            self.recording_message = "Recording started. Moving follower to leader before teleoperation."

        session.thread = threading.Thread(target=self._recording_loop, args=(session,), name="go-recording", daemon=True)
        session.thread.start()
        threading.Thread(target=self._delayed_recording_teleop, args=(session,), name="go-recording-teleop", daemon=True).start()
        return {"ok": True, "recording": self._recording_status(session), "board": self.board_json()}

    def stop_recording(self) -> dict[str, Any]:
        with self.recording_lock:
            session = self.active_recording
        if session is None:
            raise ValueError("No recording is active.")

        self.set_teleop_enabled(False)
        rest_result = self.move_follower_to_rest()
        with self.recording_lock:
            self.recording_message = "Follower sent to rest. Stopping recording..."
        session.stop_event.set()
        if session.thread is not None:
            session.thread.join(timeout=5.0)

        current = self._detect_board_state_from_camera()
        board_camera = self.cameras.get(self.board.camera)
        final_frame = board_camera.frame() if board_camera is not None else None
        final_image = "final.jpg"
        if final_frame is not None:
            cv2.imwrite(str(session.path / final_image), final_frame)

        delta = delta_between_board_states(session.baseline, current)
        delta_json = board_delta_to_jsonable(delta)
        move_name = self._move_name_from_delta(delta_json)
        session.move_name = move_name
        session.status = "complete"

        final_dir = self._unique_recording_path(session.path.parent / f"{session.id}_{move_name}")
        if final_dir != session.path:
            session.path.rename(final_dir)
            session.path = final_dir

        self._write_recording_metadata(
            session,
            status="complete",
            final=current,
            final_image=final_image if final_frame is not None else None,
            delta=delta_json,
            rest_result=rest_result,
        )
        self._start_overhead_processing(session.path)

        with self.board_lock:
            self.board_current = current
            self.board_delta = delta_json
            self.board_message = self._summarize_delta(delta_json)
        with self.recording_lock:
            self.active_recording = None
            self.recording_message = f"Saved recording {session.path.name}."
        return {
            "ok": True,
            "recording": self._recording_summary_from_metadata(session.path / "metadata.json"),
            "board": self.board_json(),
            "rest": rest_result,
        }

    def list_recordings(self) -> dict[str, Any]:
        items = []
        if self.recording_dir.is_dir():
            for metadata_path in sorted(self.recording_dir.glob("*/metadata.json")):
                try:
                    data = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if data.get("run_type") == "model_inference":
                        continue
                    items.append(self._recording_summary_from_metadata(metadata_path))
                except Exception:
                    continue
        return {
            "ok": True,
            "directory": str(self.recording_dir),
            "active": None if self.active_recording is None else self._recording_status(self.active_recording),
            "message": self.recording_message,
            "recordings": items,
        }

    def list_model_rollouts(self) -> dict[str, Any]:
        items = []
        if self.model_rollout_dir.is_dir():
            for metadata_path in sorted(self.model_rollout_dir.glob("*/metadata.json")):
                try:
                    items.append(self._recording_summary_from_metadata(metadata_path))
                except Exception:
                    continue
        return {
            "ok": True,
            "directory": str(self.model_rollout_dir),
            "active": self._model_run_status(self.active_model_run),
            "last": self._model_run_status(self.last_model_run),
            "message": self.model_message,
            "model_rollouts": items,
        }

    def list_synthetic_recordings(self) -> dict[str, Any]:
        items = []
        if self.synthetic_recording_dir.is_dir():
            for metadata_path in sorted(self.synthetic_recording_dir.glob("*/metadata.json")):
                try:
                    item = self._recording_summary_from_metadata(metadata_path)
                    item["kind"] = "synthetic_recording"
                    item["source"] = "synthetic"
                    item["readonly"] = True
                    items.append(item)
                except Exception:
                    continue
        return {
            "ok": True,
            "directory": str(self.synthetic_recording_dir),
            "synthetic_recordings": items,
        }

    def delete_model_rollout(self, rollout_id: str) -> dict[str, Any]:
        with self.model_lock:
            if self.active_model_run is not None:
                raise ValueError("Cannot delete a model rollout while one is active.")

        rollout_path = self._model_rollout_path_by_name(rollout_id)
        shutil.rmtree(rollout_path)

        with self.model_lock:
            self.model_message = f"Deleted model rollout {rollout_path.name}."
        return {
            "ok": True,
            "deleted": rollout_path.name,
            "message": self.model_message,
            "model_rollouts": self.list_model_rollouts()["model_rollouts"],
        }

    def delete_recordings(self) -> dict[str, Any]:
        with self.recording_lock:
            if self.active_recording is not None:
                raise ValueError("Cannot delete recordings while a recording is active.")
            self.recording_message = "Deleting saved recordings..."

        deleted: list[str] = []
        if self.recording_dir.is_dir():
            for path in self.recording_dir.iterdir():
                if not path.is_dir():
                    continue
                shutil.rmtree(path)
                deleted.append(path.name)

        with self.recording_lock:
            self.recording_message = f"Deleted {len(deleted)} recording{'s' if len(deleted) != 1 else ''}."
        return {
            "ok": True,
            "deleted": deleted,
            "count": len(deleted),
            "recordings": [],
            "message": self.recording_message,
        }

    def delete_recording(self, recording_id: str) -> dict[str, Any]:
        with self.recording_lock:
            if self.active_recording is not None:
                raise ValueError("Cannot delete a recording while a recording is active.")

        recording_path = self._recording_path_by_name(recording_id)
        shutil.rmtree(recording_path)

        with self.recording_lock:
            self.recording_message = f"Deleted recording {recording_path.name}."
        return {
            "ok": True,
            "deleted": recording_path.name,
            "message": self.recording_message,
            "recordings": self.list_recordings()["recordings"],
        }

    def get_recording(self, recording_id: str) -> dict[str, Any]:
        recording_path = self._recording_path_by_name(recording_id)
        metadata_path = recording_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(recording_id)
        metadata = json.loads(metadata_path.read_text())
        metadata["id"] = recording_path.name
        metadata["name"] = recording_path.name
        metadata["overhead_processed"] = overhead_processed_status(recording_path)
        self._ensure_task_state(metadata)
        metadata["task_trace"] = self._load_task_trace(recording_path)
        return {"ok": True, "recording": metadata}

    def get_model_rollout(self, rollout_id: str) -> dict[str, Any]:
        rollout_path = self._model_rollout_path_by_name(rollout_id)
        metadata_path = rollout_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(rollout_id)
        metadata = json.loads(metadata_path.read_text())
        metadata["id"] = rollout_path.name
        metadata["name"] = rollout_path.name
        metadata["overhead_processed"] = overhead_processed_status(rollout_path)
        self._ensure_task_state(metadata)
        metadata["task_trace"] = self._load_task_trace(rollout_path)
        return {"ok": True, "model_rollout": metadata}

    def get_synthetic_recording(self, recording_id: str) -> dict[str, Any]:
        recording_path = self._synthetic_recording_path_by_name(recording_id)
        metadata_path = recording_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(recording_id)
        metadata = json.loads(metadata_path.read_text())
        metadata["id"] = recording_path.name
        metadata["name"] = recording_path.name
        metadata["kind"] = "synthetic_recording"
        metadata["source"] = "synthetic"
        metadata["readonly"] = True
        metadata["cameras"] = self._recording_cameras(metadata, recording_path)
        metadata["overhead_processed"] = overhead_processed_status(recording_path)
        self._ensure_task_state(metadata)
        metadata["task_trace"] = self._load_task_trace(recording_path)
        return {"ok": True, "synthetic_recording": metadata}

    def recording_frame_jpeg(self, recording_id: str, camera: str, frame: int) -> bytes:
        recording_path = self._recording_path_by_name(recording_id)
        camera_name = Path(camera).name
        frame_name = f"{max(0, int(frame)):06d}.jpg"
        processed_path = recording_path / "overhead_processed" / frame_name
        if camera_name == self.board.camera and processed_path.is_file():
            return processed_path.read_bytes()
        frame_path = recording_path / "frames" / camera_name / frame_name
        if not frame_path.is_file():
            raise FileNotFoundError(str(frame_path))
        return frame_path.read_bytes()

    def model_rollout_frame_jpeg(self, rollout_id: str, camera: str, frame: int) -> bytes:
        rollout_path = self._model_rollout_path_by_name(rollout_id)
        camera_name = Path(camera).name
        frame_name = f"{max(0, int(frame)):06d}.jpg"
        processed_path = rollout_path / "overhead_processed" / frame_name
        if camera_name == self.board.camera and processed_path.is_file():
            return processed_path.read_bytes()
        frame_path = rollout_path / "frames" / camera_name / frame_name
        if not frame_path.is_file():
            raise FileNotFoundError(str(frame_path))
        return frame_path.read_bytes()

    def synthetic_recording_frame_jpeg(self, recording_id: str, camera: str, frame: int) -> bytes:
        recording_path = self._synthetic_recording_path_by_name(recording_id)
        camera_name = Path(camera).name
        frame_name = f"{max(0, int(frame)):06d}.jpg"
        processed_path = recording_path / "overhead_processed" / frame_name
        if camera_name == self.board.camera and processed_path.is_file():
            return processed_path.read_bytes()
        frame_path = recording_path / "frames" / camera_name / frame_name
        if not frame_path.is_file():
            raise FileNotFoundError(str(frame_path))
        return frame_path.read_bytes()

    def _recording_loop(self, session: RecordingSession) -> None:
        telemetry_path = session.path / "telemetry.jsonl"
        interval = 1.0 / max(session.sample_hz, 1.0)
        sample_index = 0
        try:
            with telemetry_path.open("a", encoding="utf-8") as telemetry_file:
                while not session.stop_event.is_set():
                    sample_started_at = time.time()
                    telemetry_state = self.telemetry.state()
                    camera_files: dict[str, str] = {}
                    for camera_name, camera in self.cameras.items():
                        frame = camera.frame()
                        if frame is None:
                            continue
                        relative = Path("frames") / camera_name / f"{sample_index:06d}.jpg"
                        cv2.imwrite(str(session.path / relative), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                        camera_files[camera_name] = str(relative)
                    sample = {
                        "index": sample_index,
                        "timestamp": sample_started_at,
                        "elapsed_s": sample_started_at - session.started_at,
                        "cameras": camera_files,
                        "telemetry": asdict(telemetry_state),
                    }
                    telemetry_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    telemetry_file.flush()
                    sample_index += 1
                    session.samples = sample_index
                    time.sleep(max(0.0, interval - (time.time() - sample_started_at)))
        except Exception as exc:
            session.status = "error"
            session.error = str(exc)
            with self.recording_lock:
                self.recording_message = f"Recording error: {exc}"

    def _delayed_recording_teleop(self, session: RecordingSession) -> None:
        with self.recording_lock:
            active = self.active_recording is session and not session.stop_event.is_set()
        if not active:
            return
        result = self.set_teleop_enabled(True)
        session.teleop_started = bool(result.get("ok"))
        with self.recording_lock:
            self.recording_message = "Recording and teleoperation active." if session.teleop_started else result.get("error", "Could not start teleoperation.")

    def _write_recording_metadata(
        self,
        session: RecordingSession,
        status: str,
        final: BoardState | None = None,
        final_image: str | None = None,
        delta: dict[str, Any] | None = None,
        rest_result: dict[str, Any] | None = None,
    ) -> None:
        camera_meta = {
            name: {
                "width": camera.actual_width or camera.spec.width,
                "height": camera.actual_height or camera.spec.height,
                "fps": camera.spec.fps,
            }
            for name, camera in self.cameras.items()
        }
        metadata = {
            "schema_version": 1,
            "id": session.id,
            "name": session.path.name,
            "status": status,
            "started_at": session.started_at,
            "ended_at": time.time() if status == "complete" else None,
            "sample_hz": session.sample_hz,
            "samples": session.samples,
            "move_name": session.move_name,
            "baseline_image": session.baseline_image,
            "final_image": final_image,
            "cameras": camera_meta,
            "board": {
                "camera": self.board.camera,
                "size": self.board.size,
                "corners_tl_tr_br_bl": self.board.corners_tl_tr_br_bl,
                "camera_to_robot_rotation_degrees": self.board.camera_to_robot_rotation_degrees,
                "overlay_fisheye_k": self.board.overlay_fisheye_k,
                "baseline": board_state_to_jsonable(session.baseline),
                "final": None if final is None else board_state_to_jsonable(final),
                "delta": delta,
                "task_state": None if delta is None else self._task_state_from_delta(delta),
            },
            "teleop_started": session.teleop_started,
            "rest_result": rest_result,
            "error": session.error,
        }
        (session.path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def _recording_status(self, session: RecordingSession) -> dict[str, Any]:
        return {
            "id": session.id,
            "name": session.path.name,
            "status": session.status,
            "samples": session.samples,
            "elapsed_s": round(time.time() - session.started_at, 2),
            "move_name": session.move_name,
            "teleop_started": session.teleop_started,
            "error": session.error,
        }

    def _recording_summary_from_metadata(self, metadata_path: Path) -> dict[str, Any]:
        data = json.loads(metadata_path.read_text())
        board = data.get("board", {})
        delta = board.get("delta") or {}
        task_state = board.get("task_state") or self._task_state_from_delta(delta, board.get("target"))
        move_name = data.get("move_name")
        if not move_name:
            target = board.get("target") or {}
            target_color = target.get("color")
            target_coord = target.get("coord")
            if target_color and target_coord:
                move_name = f"{target_color}_to_{str(target_coord).lower()}"
            else:
                move_name = "recording"
        return {
            "id": metadata_path.parent.name,
            "name": data.get("name", metadata_path.parent.name),
            "status": data.get("status", "unknown"),
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "samples": data.get("samples", 0),
            "move_name": move_name,
            "delta": delta,
            "task_state": task_state,
            "cameras": self._recording_cameras(data, metadata_path.parent),
            "overhead_processed": overhead_processed_status(metadata_path.parent),
            "run_type": data.get("run_type"),
            "synthetic": data.get("synthetic"),
        }

    def _recording_cameras(self, metadata: dict[str, Any], recording_path: Path) -> dict[str, Any]:
        cameras = metadata.get("cameras")
        if isinstance(cameras, dict) and cameras:
            return cameras
        sample_hz = metadata.get("sample_hz", 10)
        inferred: dict[str, Any] = {}
        frames_dir = recording_path / "frames"
        if not frames_dir.is_dir():
            return inferred
        for camera_dir in sorted(path for path in frames_dir.iterdir() if path.is_dir()):
            first = next(iter(sorted(camera_dir.glob("*.jpg"))), None)
            width = height = 0
            if first is not None:
                image = cv2.imread(str(first), cv2.IMREAD_COLOR)
                if image is not None:
                    height, width = image.shape[:2]
            inferred[camera_dir.name] = {
                "name": camera_dir.name,
                "width": width,
                "height": height,
                "fps": sample_hz,
            }
        return inferred

    def _recording_path_by_name(self, recording_id: str) -> Path:
        safe_name = Path(recording_id).name
        path = self.recording_dir / safe_name
        if not path.is_dir():
            raise FileNotFoundError(safe_name)
        return path

    def _model_rollout_path_by_name(self, rollout_id: str) -> Path:
        safe_name = Path(rollout_id).name
        path = self.model_rollout_dir / safe_name
        if not path.is_dir():
            raise FileNotFoundError(safe_name)
        return path

    def _synthetic_recording_path_by_name(self, recording_id: str) -> Path:
        safe_name = Path(recording_id).name
        path = self.synthetic_recording_dir / safe_name
        if not path.is_dir():
            raise FileNotFoundError(safe_name)
        return path

    @staticmethod
    def _unique_recording_path(path: Path) -> Path:
        if not path.exists():
            return path
        for index in range(2, 1000):
            candidate = path.with_name(f"{path.name}_{index}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not find unique recording path for {path}")

    def _start_overhead_processing(self, recording_path: Path) -> None:
        def worker() -> None:
            try:
                process_recording_overhead(recording_path)
            except Exception as exc:  # noqa: BLE001 - processing should not break recording.
                with self.recording_lock:
                    self.recording_message = f"Saved recording {recording_path.name}; overhead overlay error: {exc}"

        threading.Thread(
            target=worker,
            name=f"go-overhead-overlay-{recording_path.name}",
            daemon=True,
        ).start()

    def _ensure_task_state(self, metadata: dict[str, Any]) -> None:
        board = metadata.setdefault("board", {})
        if board.get("task_state") is None:
            board["task_state"] = self._task_state_from_delta(board.get("delta") or {}, board.get("target"))

    @staticmethod
    def _load_task_trace(recording_path: Path) -> list[dict[str, Any]]:
        telemetry_path = recording_path / "telemetry.jsonl"
        if not telemetry_path.is_file():
            return []
        trace: list[dict[str, Any]] = []
        for fallback_index, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_state = sample.get("task_state")
            if not isinstance(task_state, dict):
                task_state = {"done": bool(sample.get("done")), "reason": ""}
            telemetry = sample.get("telemetry")
            if not isinstance(telemetry, dict):
                telemetry = {}
            item = {
                "index": int(sample.get("index", fallback_index)),
                "done": bool(sample.get("done", task_state.get("done", False))),
                "task_state": task_state,
                "telemetry": {
                    "joints": telemetry.get("joints") if isinstance(telemetry.get("joints"), list) else [],
                    "leader_joints": (
                        telemetry.get("leader_joints") if isinstance(telemetry.get("leader_joints"), list) else []
                    ),
                    "mode": telemetry.get("mode", ""),
                    "fps": telemetry.get("fps", None),
                },
            }
            trace.append(item)
        return trace

    @staticmethod
    def _task_state_from_delta(
        delta: dict[str, Any] | None,
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        delta = delta or {}
        added = delta.get("added") or []
        removed = delta.get("removed") or []
        changed = delta.get("changed") or []
        expected = target if isinstance(target, dict) and target.get("coord") else (added[0] if len(added) == 1 else None)
        expected_coord = str(expected.get("coord", "")).upper() if isinstance(expected, dict) else ""
        expected_color = str(expected.get("color", "")).lower() if isinstance(expected, dict) else ""
        added_stone = added[0] if len(added) == 1 and isinstance(added[0], dict) else None
        target_matches = bool(
            added_stone
            and str(added_stone.get("coord", "")).upper() == expected_coord
            and str(added_stone.get("color", "")).lower() == expected_color
        )
        no_other_changes = bool(len(added) == 1 and not removed and not changed)
        done = bool(target_matches and no_other_changes)
        if done:
            reason = "target occupied with correct colour and no other board changes"
        elif len(added) != 1:
            reason = f"expected one added stone, found {len(added)}"
        elif removed or changed:
            reason = "other board changes detected"
        elif expected_coord and expected_color:
            reason = "added stone does not match target"
        else:
            reason = "no target available"
        return {
            "done": done,
            "target": None
            if not expected_coord
            else {
                "coord": expected_coord,
                "row": expected.get("row") if isinstance(expected, dict) else None,
                "col": expected.get("col") if isinstance(expected, dict) else None,
                "color": expected_color,
            },
            "reason": reason,
            "delta": {
                "added": len(added),
                "removed": len(removed),
                "changed": len(changed),
            },
        }

    @staticmethod
    def _move_name_from_delta(delta: dict[str, Any]) -> str:
        added = delta.get("added", [])
        removed = delta.get("removed", [])
        changed = delta.get("changed", [])
        if len(added) == 1 and not removed and not changed:
            stone = added[0]
            return f"{stone['color']}_to_{str(stone['coord']).lower()}"
        if not added and not removed and not changed:
            return "no_move"
        parts = []
        if added:
            parts.append(f"{len(added)}_added")
        if removed:
            parts.append(f"{len(removed)}_removed")
        if changed:
            parts.append(f"{len(changed)}_changed")
        return "_".join(parts) if parts else "move"

    def capture_board(self, slot: str) -> dict[str, Any]:
        state = self._detect_board_state_from_camera()
        with self.board_lock:
            if slot == "baseline":
                self.board_baseline = state
                self.board_delta = None
                self.board_message = f"Baseline captured with {state.summary['total']} stones."
            elif slot == "current":
                self.board_current = state
                if self.board_baseline is not None:
                    delta = delta_between_board_states(self.board_baseline, self.board_current)
                    self.board_delta = board_delta_to_jsonable(delta)
                    self.board_message = self._summarize_delta(self.board_delta)
                else:
                    self.board_message = f"Current board captured with {state.summary['total']} stones."
            else:
                raise ValueError("slot must be 'baseline' or 'current'")

        return {"ok": True, "board": self.board_json()}

    def compute_board_delta(self) -> dict[str, Any]:
        with self.board_lock:
            if self.board_baseline is None or self.board_current is None:
                raise ValueError("Capture both baseline and current board states before computing a delta.")
            delta = delta_between_board_states(self.board_baseline, self.board_current)
            self.board_delta = board_delta_to_jsonable(delta)
            self.board_message = self._summarize_delta(self.board_delta)
        return {"ok": True, "board": self.board_json()}

    def detect_current_board(self) -> dict[str, Any]:
        state = self._detect_board_state_from_camera()
        with self.board_lock:
            self.board_live_state = state
        return {"ok": True, "state": board_state_to_jsonable(state)}

    def board_overlay_jpeg(self, rotate: str = "") -> bytes | None:
        camera = self.cameras.get(self.board.camera)
        if camera is None:
            raise ValueError(f"Board camera '{self.board.camera}' is not configured.")
        frame = camera.frame()
        if frame is None:
            return None

        corners = (
            np.array(self.board.corners_tl_tr_br_bl, dtype=np.float32)
            if self.board.corners_tl_tr_br_bl is not None
            else None
        )
        if corners is None:
            corners = auto_detect_board_corners(frame)
        raw_corners = [[float(x), float(y)] for x, y in corners.tolist()]
        raw_state = BoardState(
            board_size=self.board.size,
            corners_tl_tr_br_bl=raw_corners,
            stones=[],
            occupied={},
            summary={"black": 0, "white": 0, "total": 0},
        )
        with self.board_lock:
            live_state = self.board_live_state
        state = live_state if live_state is not None else BoardState(
            board_size=self.board.size,
            corners_tl_tr_br_bl=raw_corners,
            stones=[],
            occupied={},
            summary={"black": 0, "white": 0, "total": 0},
        )
        overlay = self._draw_board_overlay(frame, raw_state, state)
        if rotate == "left":
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return encoded.tobytes() if ok else None

    def _draw_board_overlay(self, frame: np.ndarray, raw_state: BoardState, state: BoardState) -> np.ndarray:
        overlay = frame.copy()
        corners = np.array(raw_state.corners_tl_tr_br_bl, dtype=np.float32)
        points = self._curved_grid_points(corners, frame.shape)

        for row in range(self.board.size):
            cv2.polylines(overlay, [np.array(points[row], dtype=np.int32)], False, (0, 220, 255), 2)
        for col in range(self.board.size):
            column_points = np.array([points[row][col] for row in range(self.board.size)], dtype=np.int32)
            cv2.polylines(overlay, [column_points], False, (0, 220, 255), 2)

        for robot_row in range(self.board.size):
            for robot_col in range(self.board.size):
                camera_row, camera_col = inverse_transform_row_col(
                    robot_row,
                    robot_col,
                    self.board.size,
                    self.board.camera_to_robot_rotation_degrees,
                )
                cv2.circle(overlay, points[camera_row][camera_col], 2, (255, 255, 255), -1)

        for stone in state.stones:
            camera_row, camera_col = inverse_transform_row_col(
                stone.row,
                stone.col,
                self.board.size,
                self.board.camera_to_robot_rotation_degrees,
            )
            point = points[camera_row][camera_col]
            fill = (20, 20, 20) if stone.color == "black" else (245, 245, 245)
            outline = (255, 255, 255) if stone.color == "black" else (20, 20, 20)
            label_color = (255, 255, 255) if stone.color == "black" else (20, 20, 20)
            cv2.circle(overlay, point, 18, outline, 3)
            cv2.circle(overlay, point, 14, fill, -1)
            cv2.putText(
                overlay,
                stone.coord,
                (point[0] + 16, point[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                2,
                cv2.LINE_AA,
            )

        return overlay

    @staticmethod
    def _bilinear(corners: np.ndarray, row_t: float, col_t: float) -> np.ndarray:
        top = corners[0] * (1.0 - col_t) + corners[1] * col_t
        bottom = corners[3] * (1.0 - col_t) + corners[2] * col_t
        return top * (1.0 - row_t) + bottom * row_t

    def _curved_grid_points(self, corners: np.ndarray, image_shape: tuple[int, ...]) -> list[list[tuple[int, int]]]:
        corners = corners.astype(np.float32)
        corner_offsets = np.array(
            [self._curve_overlay_point(point, image_shape) - point for point in corners],
            dtype=np.float32,
        )
        points: list[list[tuple[int, int]]] = []
        for row in range(self.board.size):
            row_t = row / (self.board.size - 1)
            point_row = []
            for col in range(self.board.size):
                col_t = col / (self.board.size - 1)
                base = self._bilinear(corners, row_t, col_t)
                radial_offset = self._curve_overlay_point(base, image_shape) - base
                anchored_offset = radial_offset - self._bilinear(corner_offsets, row_t, col_t)
                point = base + anchored_offset
                point_row.append(tuple(np.round(point).astype(int)))
            points.append(point_row)
        return points

    def _curve_overlay_point(self, point: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
        k = float(self.board.overlay_fisheye_k)
        x = float(point[0])
        y = float(point[1])
        if abs(k) < 1e-9:
            return np.array([x, y], dtype=np.float32)

        height, width = image_shape[:2]
        cx = width / 2.0
        cy = height / 2.0
        scale = max(width, height) / 2.0
        dx = (x - cx) / scale
        dy = (y - cy) / scale
        factor = 1.0 + k * (dx * dx + dy * dy)
        curved_x = cx + dx * factor * scale
        curved_y = cy + dy * factor * scale
        return np.array([curved_x, curved_y], dtype=np.float32)

    def update_board_tuning(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("reset"):
            self._restore_saved_board_tuning()
            return {"ok": True, "board": self.board_json()}

        tuning_fields = {
            "sample_radius_ratio": (0.12, 0.5),
            "black_l_threshold": (30.0, 140.0),
            "white_l_threshold": (120.0, 230.0),
            "white_s_threshold": (25.0, 150.0),
            "stone_min_radius_ratio": (0.1, 0.45),
            "stone_max_radius_ratio": (0.2, 0.7),
            "stone_min_circularity": (0.25, 0.95),
            "stone_max_snap_distance_ratio": (0.15, 0.85),
            "black_grid_min_edge_score": (0.12, 0.78),
            "overlay_fisheye_k": (-0.75, 0.75),
        }
        for name, (minimum, maximum) in tuning_fields.items():
            if name not in payload:
                continue
            value = float(payload[name])
            setattr(self.board, name, float(np.clip(value, minimum, maximum)))
        if self.board.stone_min_radius_ratio > self.board.stone_max_radius_ratio:
            self.board.stone_min_radius_ratio, self.board.stone_max_radius_ratio = (
                self.board.stone_max_radius_ratio,
                self.board.stone_min_radius_ratio,
            )

        if "corners_tl_tr_br_bl" in payload:
            corners = payload["corners_tl_tr_br_bl"]
            if not isinstance(corners, list) or len(corners) != 4:
                raise ValueError("corners_tl_tr_br_bl must contain four [x, y] points.")
            parsed_corners = []
            for point in corners:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each corner must be [x, y].")
                parsed_corners.append([float(point[0]), float(point[1])])
            self.board.corners_tl_tr_br_bl = parsed_corners

        if payload.get("persist"):
            self._persist_board_tuning()

        return {"ok": True, "board": self.board_json()}

    def _restore_saved_board_tuning(self) -> None:
        defaults = BoardSpec()
        if self.config_path is None or not self.config_path.is_file():
            self.board.sample_radius_ratio = defaults.sample_radius_ratio
            self.board.black_l_threshold = defaults.black_l_threshold
            self.board.white_l_threshold = defaults.white_l_threshold
            self.board.white_s_threshold = defaults.white_s_threshold
            self.board.stone_min_radius_ratio = defaults.stone_min_radius_ratio
            self.board.stone_max_radius_ratio = defaults.stone_max_radius_ratio
            self.board.stone_min_circularity = defaults.stone_min_circularity
            self.board.stone_max_snap_distance_ratio = defaults.stone_max_snap_distance_ratio
            self.board.black_grid_min_edge_score = defaults.black_grid_min_edge_score
            self.board.overlay_fisheye_k = defaults.overlay_fisheye_k
            return
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        board = raw.get("board", {})
        self.board.sample_radius_ratio = float(board.get("sample_radius_ratio", defaults.sample_radius_ratio))
        self.board.black_l_threshold = float(board.get("black_l_threshold", defaults.black_l_threshold))
        self.board.white_l_threshold = float(board.get("white_l_threshold", defaults.white_l_threshold))
        self.board.white_s_threshold = float(board.get("white_s_threshold", defaults.white_s_threshold))
        self.board.stone_min_radius_ratio = float(board.get("stone_min_radius_ratio", defaults.stone_min_radius_ratio))
        self.board.stone_max_radius_ratio = float(board.get("stone_max_radius_ratio", defaults.stone_max_radius_ratio))
        self.board.stone_min_circularity = float(board.get("stone_min_circularity", defaults.stone_min_circularity))
        self.board.stone_max_snap_distance_ratio = float(
            board.get("stone_max_snap_distance_ratio", defaults.stone_max_snap_distance_ratio)
        )
        self.board.black_grid_min_edge_score = float(
            board.get("black_grid_min_edge_score", defaults.black_grid_min_edge_score)
        )
        self.board.overlay_fisheye_k = float(board.get("overlay_fisheye_k", defaults.overlay_fisheye_k))
        if "corners_tl_tr_br_bl" in board:
            self.board.corners_tl_tr_br_bl = board["corners_tl_tr_br_bl"]

    def _persist_board_tuning(self) -> None:
        if self.config_path is None:
            return
        raw = json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.is_file() else {}
        board = raw.setdefault("board", {})
        board["sample_radius_ratio"] = self.board.sample_radius_ratio
        board["black_l_threshold"] = self.board.black_l_threshold
        board["white_l_threshold"] = self.board.white_l_threshold
        board["white_s_threshold"] = self.board.white_s_threshold
        board["stone_min_radius_ratio"] = self.board.stone_min_radius_ratio
        board["stone_max_radius_ratio"] = self.board.stone_max_radius_ratio
        board["stone_min_circularity"] = self.board.stone_min_circularity
        board["stone_max_snap_distance_ratio"] = self.board.stone_max_snap_distance_ratio
        board["black_grid_min_edge_score"] = self.board.black_grid_min_edge_score
        board["overlay_fisheye_k"] = self.board.overlay_fisheye_k
        board["corners_tl_tr_br_bl"] = self.board.corners_tl_tr_br_bl
        self.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    def save_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        camera = self.cameras.get(self.board.camera)
        if camera is None:
            raise ValueError(f"Board camera '{self.board.camera}' is not configured.")
        frame = camera.frame()
        if frame is None:
            raise ValueError(f"Board camera '{self.board.camera}' has no frame yet.")

        raw_stones = payload.get("stones", [])
        if not isinstance(raw_stones, list):
            raise ValueError("stones must be a list.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        label = str(payload.get("label", "")).strip()
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        suffix = f"_{slug[:36]}" if slug else ""
        stem = f"{timestamp}{suffix}"

        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.annotation_dir / f"{stem}.jpg"
        json_path = self.annotation_dir / f"{stem}.json"
        if image_path.exists() or json_path.exists():
            stem = f"{stem}_{int(time.time() * 1000) % 100000}"
            image_path = self.annotation_dir / f"{stem}.jpg"
            json_path = self.annotation_dir / f"{stem}.json"

        stones = self._validated_annotation_stones(raw_stones)
        cv2.imwrite(str(image_path), frame)
        annotation = {
            "schema_version": 1,
            "created_at": time.time(),
            "label": label,
            "image": image_path.name,
            "camera": self.board.camera,
            "board_size": self.board.size,
            "corners_tl_tr_br_bl": self.board.corners_tl_tr_br_bl,
            "camera_to_robot_rotation_degrees": self.board.camera_to_robot_rotation_degrees,
            "overlay_fisheye_k": self.board.overlay_fisheye_k,
            "stones": stones,
            "occupied": {stone["coord"]: stone["color"] for stone in stones},
            "summary": {
                "black": sum(stone["color"] == "black" for stone in stones),
                "white": sum(stone["color"] == "white" for stone in stones),
                "total": len(stones),
            },
        }
        json_path.write_text(json.dumps(annotation, indent=2) + "\n")
        return {"ok": True, "annotation": annotation, "json_path": str(json_path), "image_path": str(image_path)}

    def list_annotations(self) -> dict[str, Any]:
        items = []
        if self.annotation_dir.is_dir():
            for path in sorted(self.annotation_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                image_name = data.get("image")
                occupied = data.get("occupied")
                corners = data.get("corners_tl_tr_br_bl")
                if not isinstance(image_name, str) or not isinstance(occupied, dict) or corners is None:
                    continue
                if not (self.annotation_dir / image_name).is_file():
                    continue
                items.append(
                    {
                        "file": path.name,
                        "image": image_name,
                        "label": data.get("label", ""),
                        "summary": data.get("summary", {}),
                        "created_at": data.get("created_at"),
                    }
                )
        return {"ok": True, "directory": str(self.annotation_dir), "annotations": items}

    def get_annotation(self, filename: str) -> dict[str, Any]:
        safe_name = Path(filename).name
        if not safe_name.endswith(".json"):
            raise ValueError("Annotation requires a .json file.")

        json_path = self.annotation_dir / safe_name
        if not json_path.is_file():
            raise FileNotFoundError(safe_name)

        data = json.loads(json_path.read_text())
        image_name = data.get("image")
        occupied = data.get("occupied")
        corners = data.get("corners_tl_tr_br_bl")
        if not isinstance(image_name, str) or not isinstance(occupied, dict) or corners is None:
            raise ValueError(f"{safe_name} is not a board annotation.")
        image = cv2.imread(str(self.annotation_dir / Path(image_name).name))
        image_shape = None if image is None else {"width": int(image.shape[1]), "height": int(image.shape[0])}
        return {"ok": True, "file": safe_name, "annotation": data, "image_shape": image_shape}

    def update_annotation_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = str(payload.get("file", ""))
        loaded = self.get_annotation(filename)
        safe_name = str(loaded["file"])
        data = dict(loaded["annotation"])

        if "corners_tl_tr_br_bl" in payload:
            corners = payload["corners_tl_tr_br_bl"]
            if not isinstance(corners, list) or len(corners) != 4:
                raise ValueError("corners_tl_tr_br_bl must contain four [x, y] points.")
            parsed_corners = []
            for point in corners:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each corner must be [x, y].")
                parsed_corners.append([float(point[0]), float(point[1])])
            data["corners_tl_tr_br_bl"] = parsed_corners

        if "overlay_fisheye_k" in payload:
            data["overlay_fisheye_k"] = float(np.clip(float(payload["overlay_fisheye_k"]), -0.75, 0.75))

        json_path = self.annotation_dir / safe_name
        json_path.write_text(json.dumps(data, indent=2) + "\n")
        return {"ok": True, "file": safe_name, "annotation": data}

    def delete_annotation(self, filename: str) -> dict[str, Any]:
        loaded = self.get_annotation(filename)
        safe_name = str(loaded["file"])
        data = loaded["annotation"]

        json_path = self.annotation_dir / safe_name
        image_name = Path(str(data.get("image", ""))).name
        image_path = self.annotation_dir / image_name if image_name else None

        deleted: list[str] = []
        if image_path is not None and image_path.is_file():
            image_path.unlink()
            deleted.append(image_path.name)
        if json_path.is_file():
            json_path.unlink()
            deleted.append(json_path.name)

        return {
            "ok": True,
            "file": safe_name,
            "deleted": deleted,
            "annotations": self.list_annotations()["annotations"],
            "message": f"Deleted snapshot {safe_name}.",
        }

    def annotation_overlay_jpeg(
        self,
        filename: str,
        rotate: str = "",
        corners_override: list[list[float]] | None = None,
        overlay_fisheye_k_override: float | None = None,
    ) -> bytes | None:
        loaded = self.get_annotation(filename)
        data = loaded["annotation"]
        image_name = data.get("image")
        occupied = data.get("occupied")
        corners = corners_override if corners_override is not None else data.get("corners_tl_tr_br_bl")

        image_path = self.annotation_dir / Path(image_name).name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path.name)

        board_size = int(data.get("board_size", self.board.size))
        rotation = int(data.get("camera_to_robot_rotation_degrees", self.board.camera_to_robot_rotation_degrees))
        corners_list = [[float(x), float(y)] for x, y in corners]
        stones = []
        columns = GO_COLUMNS if board_size == 19 else "".join(chr(ord("A") + i) for i in range(board_size))
        for coord, color in occupied.items():
            coord = str(coord)
            color = str(color)
            if color not in {"black", "white"} or coord[0] not in columns:
                continue
            row = int(coord[1:]) - 1
            col = columns.index(coord[0])
            if row < 0 or row >= board_size or col < 0 or col >= board_size:
                continue
            stones.append(
                Stone(
                    coord=coord,
                    row=row,
                    col=col,
                    color=color,
                    confidence=1.0,
                    board_xy=(0.0, 0.0),
                    image_xy=(0.0, 0.0),
                )
            )
        state = BoardState(
            board_size=board_size,
            corners_tl_tr_br_bl=corners_list,
            stones=stones,
            occupied={stone.coord: stone.color for stone in stones},
            summary={
                "black": sum(stone.color == "black" for stone in stones),
                "white": sum(stone.color == "white" for stone in stones),
                "total": len(stones),
            },
        )
        raw_state = BoardState(
            board_size=board_size,
            corners_tl_tr_br_bl=corners_list,
            stones=[],
            occupied={},
            summary={"black": 0, "white": 0, "total": 0},
        )

        previous_size = self.board.size
        previous_rotation = self.board.camera_to_robot_rotation_degrees
        previous_curve = self.board.overlay_fisheye_k
        try:
            self.board.size = board_size
            self.board.camera_to_robot_rotation_degrees = rotation
            self.board.overlay_fisheye_k = (
                float(overlay_fisheye_k_override)
                if overlay_fisheye_k_override is not None
                else float(data.get("overlay_fisheye_k", self.board.overlay_fisheye_k))
            )
            overlay = self._draw_board_overlay(image, raw_state, state)
        finally:
            self.board.size = previous_size
            self.board.camera_to_robot_rotation_degrees = previous_rotation
            self.board.overlay_fisheye_k = previous_curve

        if rotate == "left":
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return encoded.tobytes() if ok else None

    def _validated_annotation_stones(self, raw_stones: list[Any]) -> list[dict[str, Any]]:
        columns = GO_COLUMNS if self.board.size == 19 else "".join(chr(ord("A") + i) for i in range(self.board.size))
        stones_by_coord: dict[str, dict[str, Any]] = {}
        for raw in raw_stones:
            if not isinstance(raw, dict):
                raise ValueError("Each stone must be an object.")
            row = int(raw.get("row"))
            col = int(raw.get("col"))
            color = str(raw.get("color"))
            if row < 0 or row >= self.board.size or col < 0 or col >= self.board.size:
                raise ValueError(f"Stone out of range: row={row} col={col}.")
            if color not in {"black", "white"}:
                raise ValueError(f"Invalid stone color: {color}.")
            coord = f"{columns[col]}{row + 1}"
            stones_by_coord[coord] = {"coord": coord, "row": row, "col": col, "color": color}
        return sorted(stones_by_coord.values(), key=lambda stone: (stone["row"], stone["col"]))

    def _detect_board_state_from_camera(self) -> BoardState:
        camera = self.cameras.get(self.board.camera)
        if camera is None:
            raise ValueError(f"Board camera '{self.board.camera}' is not configured.")
        frame = camera.frame()
        if frame is None:
            raise ValueError(f"Board camera '{self.board.camera}' has no frame yet.")

        corners = (
            np.array(self.board.corners_tl_tr_br_bl, dtype=np.float32)
            if self.board.corners_tl_tr_br_bl is not None
            else None
        )
        state = board_state_from_image(
            frame,
            corners=corners,
            size=self.board.size,
            board_pixels=self.board.board_pixels,
            sample_radius_ratio=self.board.sample_radius_ratio,
            black_l_threshold=self.board.black_l_threshold,
            white_l_threshold=self.board.white_l_threshold,
            white_s_threshold=self.board.white_s_threshold,
            stone_min_radius_ratio=self.board.stone_min_radius_ratio,
            stone_max_radius_ratio=self.board.stone_max_radius_ratio,
            stone_min_circularity=self.board.stone_min_circularity,
            stone_max_snap_distance_ratio=self.board.stone_max_snap_distance_ratio,
            black_grid_min_edge_score=self.board.black_grid_min_edge_score,
            overlay_fisheye_k=self.board.overlay_fisheye_k,
        )
        return transform_board_state(state, self.board.camera_to_robot_rotation_degrees)

    def board_json(self) -> dict[str, Any]:
        with self.board_lock:
            return {
                "camera": self.board.camera,
                "baseline": None
                if self.board_baseline is None
                else board_state_to_jsonable(self.board_baseline),
                "current": None if self.board_current is None else board_state_to_jsonable(self.board_current),
                "delta": self.board_delta,
                "message": self.board_message,
                "has_corners": self.board.corners_tl_tr_br_bl is not None,
                "corners_tl_tr_br_bl": self.board.corners_tl_tr_br_bl,
                "camera_to_robot_rotation_degrees": self.board.camera_to_robot_rotation_degrees,
                "tuning": {
                    "overlay_fisheye_k": self.board.overlay_fisheye_k,
                    "sample_radius_ratio": self.board.sample_radius_ratio,
                    "black_l_threshold": self.board.black_l_threshold,
                    "white_l_threshold": self.board.white_l_threshold,
                    "white_s_threshold": self.board.white_s_threshold,
                    "stone_min_radius_ratio": self.board.stone_min_radius_ratio,
                    "stone_max_radius_ratio": self.board.stone_max_radius_ratio,
                    "stone_min_circularity": self.board.stone_min_circularity,
                    "stone_max_snap_distance_ratio": self.board.stone_max_snap_distance_ratio,
                    "black_grid_min_edge_score": self.board.black_grid_min_edge_score,
                },
            }

    @staticmethod
    def _summarize_delta(delta: dict[str, Any]) -> str:
        added = delta["added"]
        removed = delta["removed"]
        changed = delta["changed"]
        if len(added) == 1 and not removed and not changed:
            stone = added[0]
            return f"Added {stone['color']} stone at {stone['coord']}."
        parts = []
        if added:
            parts.append("added " + ", ".join(f"{s['color']} {s['coord']}" for s in added))
        if removed:
            parts.append("removed " + ", ".join(f"{s['color']} {s['coord']}" for s in removed))
        if changed:
            parts.append("changed " + ", ".join(str(item["coord"]) for item in changed))
        return "; ".join(parts) if parts else "No board-state delta detected."

    def state_json(self) -> dict[str, Any]:
        state = asdict(self.telemetry.state())
        state["cameras"] = [
            {
                "name": name,
                "width": camera.spec.width,
                "height": camera.spec.height,
                "fps": camera.spec.fps,
                **camera.status(),
            }
            for name, camera in self.cameras.items()
        ]
        state["board"] = self.board_json()
        with self.recording_lock:
            active = self.active_recording
            message = self.recording_message
        state["recording"] = {
            "active": None if active is None else self._recording_status(active),
            "message": message,
        }
        state["model_run"] = self.model_run_json()
        state["evaluator"] = self.evaluator_json()
        return state


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Go Recording Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --line: #d8dee3;
      --text: #182026;
      --muted: #66727c;
      --teal: #0f766e;
      --amber: #b7791f;
      --red: #b42318;
      --blue: #2364aa;
    }
    * { box-sizing: border-box; }
    html,
    body {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .app {
      height: 100vh;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .mark {
      width: 28px;
      height: 28px;
      border: 2px solid #1f2933;
      background:
        linear-gradient(90deg, transparent 46%, #1f2933 47%, #1f2933 53%, transparent 54%),
        linear-gradient(0deg, transparent 46%, #1f2933 47%, #1f2933 53%, transparent 54%),
        #dfb15b;
    }
    h1 {
      font-size: 16px;
      line-height: 1;
      margin: 0;
      font-weight: 700;
      white-space: nowrap;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--red);
    }
    .dot.connected { background: var(--teal); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 14px;
      padding: 14px;
      min-height: 0;
      overflow: hidden;
    }
    .feeds {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      min-height: 0;
      overflow: hidden;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
      overflow: hidden;
    }
    .panel-title {
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
    }
    .panel-meta {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-weight: 600;
    }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      line-height: 1.4;
      background: #f7fafb;
    }
    .badge.ok { color: var(--teal); border-color: #9fd8d2; background: #eefaf8; }
    .badge.warn { color: var(--amber); border-color: #e8c780; background: #fff8e8; }
    .badge.bad { color: var(--red); border-color: #e7a39e; background: #fff3f2; }
    .camera {
      display: grid;
      grid-template-rows: 40px minmax(0, 1fr);
      height: 100%;
      min-height: 0;
    }
    .camera img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #101820;
      display: block;
    }
    .camera-frame {
      position: relative;
      min-height: 0;
      background: #101820;
    }
    .camera-frame img {
      position: absolute;
      inset: 0;
    }
    .camera-warning {
      position: absolute;
      left: 10px;
      bottom: 10px;
      max-width: calc(100% - 20px);
      padding: 6px 8px;
      border-radius: 6px;
      background: rgba(255, 248, 232, 0.94);
      color: #7a4d00;
      border: 1px solid rgba(232, 199, 128, 0.9);
      font-size: 12px;
      display: none;
    }
    .camera-warning.visible { display: block; }
    aside {
      display: grid;
      grid-auto-rows: max-content;
      gap: 14px;
      min-height: 0;
      overflow-y: auto;
      padding-right: 2px;
    }
    .controls {
      padding: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .controls > button,
    .controls > a {
      flex: 1;
    }
    .session-nav {
      padding: 0 10px 10px;
      display: grid;
      gap: 8px;
    }
    .session-nav a,
    .session-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
    }
    button {
      appearance: none;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      border-radius: 8px;
      padding: 9px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    button.active {
      color: #ffffff;
      border-color: var(--red);
      background: var(--red);
    }
    button:disabled {
      color: var(--muted);
      cursor: not-allowed;
      background: #edf1f3;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      min-width: 0;
    }
    .metric span {
      color: var(--muted);
      font-size: 11px;
      display: block;
    }
    .metric strong {
      font-size: 18px;
      line-height: 1.35;
      display: block;
      overflow-wrap: anywhere;
    }
    .ee {
      padding: 10px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .workspace {
      width: 100%;
      height: 180px;
      display: block;
      border-top: 1px solid var(--line);
      background: #fbfcfd;
    }
    .joints {
      padding: 10px;
      display: grid;
      gap: 10px;
      overflow: auto;
    }
    .joint {
      display: grid;
      gap: 5px;
    }
    .joint-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 12px;
    }
    .bar {
      position: relative;
      height: 8px;
      background: #e8edf0;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar .value-fill {
      height: 100%;
      width: 50%;
      background: var(--blue);
      border-radius: inherit;
    }
    .bar .delta-band {
      position: absolute;
      top: 0;
      height: 100%;
      min-width: 3px;
      opacity: 0.34;
      border-radius: inherit;
    }
    .bar .target-marker {
      position: absolute;
      top: -3px;
      width: 3px;
      height: 14px;
      border-radius: 999px;
      transform: translateX(-50%);
    }
    .delta-band.ok,
    .target-marker.ok { background: var(--teal); }
    .delta-band.warn,
    .target-marker.warn { background: var(--amber); }
    .delta-band.bad,
    .target-marker.bad { background: var(--red); }
    .joint-values {
      display: inline-flex;
      align-items: baseline;
      justify-content: flex-end;
      gap: 7px;
      white-space: nowrap;
    }
    .joint-delta.ok { color: var(--teal); }
    .joint-delta.warn { color: var(--amber); }
    .joint-delta.bad { color: var(--red); }
    .note {
      color: var(--muted);
      font-size: 12px;
      padding: 0 10px 10px;
      min-height: 28px;
    }
    .board-actions {
      padding: 10px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .recording-actions {
      padding: 10px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .recording-actions button.danger {
      border-color: #e7a39e;
      color: var(--red);
    }
    .recording-actions button.recording {
      background: var(--red);
      border-color: var(--red);
      color: #fff;
    }
    .recording-actions input[type="range"] {
      grid-column: 1 / -1;
      width: 100%;
    }
    .replay-state {
      margin: 0 10px 8px;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f7fafc;
      color: var(--muted);
      font-size: 12px;
      display: none;
      justify-content: space-between;
      gap: 8px;
    }
    .replay-state.visible {
      display: flex;
    }
    .replay-state strong {
      color: var(--ink);
      text-transform: capitalize;
    }
    .recordings {
      margin: 0;
      padding: 0 10px 10px;
      list-style: none;
      display: grid;
      gap: 5px;
      max-height: 135px;
      overflow: auto;
    }
    .recordings button {
      width: 100%;
      text-align: left;
      padding: 6px 8px;
      font-size: 12px;
    }
    .recording-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 6px;
      align-items: stretch;
    }
    .recording-row button.delete-one {
      width: auto;
      min-width: 64px;
      text-align: center;
      border-color: #e7a39e;
      color: var(--red);
    }
    .recording-row.selected > button:first-child {
      border-color: var(--blue);
      background: #f2f7ff;
      color: var(--blue);
    }
    .replay-delta {
      margin: 0 10px 10px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      display: none;
      gap: 6px;
      font-size: 12px;
    }
    .replay-delta.visible { display: grid; }
    .replay-delta.warn {
      border-color: #e8c780;
      background: #fff8e8;
    }
    .replay-delta-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
    }
    .replay-delta ul {
      margin: 0;
      padding-left: 18px;
    }
    .recording-board-block {
      margin: 0 10px 10px;
      display: grid;
      gap: 6px;
    }
    .recording-board-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .board-title-actions {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .icon-button {
      width: 24px;
      height: 24px;
      padding: 0;
      display: none;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      font-size: 15px;
      line-height: 1;
    }
    .icon-button.visible {
      display: inline-flex;
    }
    .sample-board {
      --grid-pad: 6%;
      --grid-line: 1px;
      --grid-span: calc(100% - 2 * var(--grid-pad) - var(--grid-line));
      position: relative;
      width: 100%;
      max-width: 240px;
      aspect-ratio: 1;
      justify-self: center;
      background: #d7a85d;
      border: 2px solid #5f3d1f;
      border-radius: 6px;
      overflow: visible;
    }
    .sample-board::before {
      content: "";
      position: absolute;
      inset: var(--grid-pad);
      border: var(--grid-line) solid #1f2933;
      background:
        repeating-linear-gradient(
          90deg,
          #1f2933 0 var(--grid-line),
          transparent var(--grid-line) calc((100% - var(--grid-line)) / 18)
        ),
        repeating-linear-gradient(
          0deg,
          #1f2933 0 var(--grid-line),
          transparent var(--grid-line) calc((100% - var(--grid-line)) / 18)
        );
      pointer-events: none;
    }
    .sample-marker {
      position: absolute;
      left: calc(var(--grid-pad) + var(--grid-line) / 2 + var(--col) * var(--grid-span) / 18);
      top: calc(var(--grid-pad) + var(--grid-line) / 2 + var(--row) * var(--grid-span) / 18);
      width: var(--size, 16px);
      height: var(--size, 16px);
      transform: translate(-50%, -50%);
      z-index: 2;
      border-radius: 999px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(24, 32, 38, 0.55);
      color: #fff;
      background: var(--blue);
      font-size: 10px;
      font-weight: 800;
      line-height: 1;
      box-shadow: 0 1px 3px rgba(0,0,0,.25);
    }
    .sample-marker.added {
      background: var(--teal);
    }
    .sample-marker.removed {
      background: var(--red);
      text-decoration: line-through;
    }
    .sample-marker.changed {
      background: var(--amber);
    }
    .sample-marker.white-stone {
      color: #182026;
      background: radial-gradient(circle at 35% 30%, #fff, #d8dde0 75%);
    }
    .sample-marker.black-stone {
      color: #fff;
      background: radial-gradient(circle at 35% 30%, #34383b, #050607 68%);
      border-color: rgba(255,255,255,.55);
    }
    .sample-marker::after {
      content: attr(data-label);
      position: absolute;
      left: 50%;
      top: -5px;
      transform: translate(-50%, -100%);
      z-index: 5;
      display: none;
      white-space: nowrap;
      background: #182026;
      color: #fff;
      border-radius: 4px;
      padding: 2px 5px;
      font-size: 11px;
      font-weight: 700;
    }
    .sample-marker:hover::after {
      display: block;
    }
    .align-actions {
      padding: 10px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .model-form {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .model-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .model-grid .wide {
      grid-column: 1 / -1;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    input,
    select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 9px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
    }
    .model-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .model-log {
      margin: 0 10px 10px;
      max-height: 110px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #101820;
      color: #dbe7ec;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.4;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .delta-list {
      margin: 0;
      padding: 0 10px 10px;
      list-style: none;
      display: grid;
      gap: 4px;
      color: var(--text);
      font-size: 12px;
    }
    .delta-list li {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 7px;
      background: #fbfcfd;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .feeds { grid-template-columns: 1fr; }
      aside { grid-template-rows: auto auto auto; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark"></div>
        <h1>Go Recording Dashboard</h1>
      </div>
      <div class="status">
        <div id="status-dot" class="dot"></div>
        <span id="status-text">Connecting</span>
      </div>
    </header>
    <main>
      <section id="feeds" class="feeds"></section>
      <aside>
        <section class="panel">
          <div class="panel-title">
            <span>Session</span>
            <span id="mode">-</span>
          </div>
          <div class="metrics">
            <div class="metric"><span>Telemetry</span><strong id="fps">-</strong></div>
            <div class="metric"><span>Cameras</span><strong id="camera-count">-</strong></div>
            <div class="metric"><span>Age</span><strong id="age">-</strong></div>
          </div>
          <div class="controls">
            <button id="refresh-devices-button" type="button">Refresh Devices</button>
            <a class="session-link" href="/evaluate">Evaluate</a>
            <a class="session-link" href="/annotate">Annotate</a>
          </div>
          <div id="note" class="note"></div>
        </section>
        <section class="panel">
          <div class="panel-title">
            <span>Recording</span>
            <span id="recording-state">idle</span>
          </div>
          <div class="recording-actions">
            <button id="record-button" type="button">Record</button>
            <button id="stop-record-button" type="button">Stop</button>
            <button id="replay-button" type="button">Play</button>
            <input id="replay-slider" type="range" min="0" max="0" value="0" />
          </div>
          <div id="recording-task-state" class="replay-state"></div>
          <div id="recording-note" class="note"></div>
          <div id="recording-delta" class="replay-delta"></div>
          <div class="recording-board-block">
            <div class="recording-board-title">
              <span id="recording-board-title">Sample Distribution</span>
              <span class="board-title-actions">
                <span id="recording-board-count">0 samples</span>
                <button id="clear-selected-recording" class="icon-button" type="button" title="Back to collected view">×</button>
              </span>
            </div>
            <div id="recording-samples-board" class="sample-board"></div>
          </div>
          <ul id="recordings" class="recordings"></ul>
          <div class="recording-board-title">
            <span>Synthetic</span>
            <span id="synthetic-recording-count">0 runs</span>
          </div>
          <ul id="synthetic-recordings" class="recordings"></ul>
        </section>
        <section class="panel">
          <div class="panel-title">
            <span>Model Rollout</span>
            <span id="model-state">idle</span>
          </div>
          <div class="model-form">
            <div class="model-grid">
              <label>Coord
                <input id="model-coord" value="Q16" inputmode="text" autocomplete="off" />
              </label>
              <label>Color
                <select id="model-color">
                  <option value="white">White</option>
                  <option value="black">Black</option>
                </select>
              </label>
              <label class="wide">Policy Path
                <input id="model-policy-path" autocomplete="off" />
              </label>
              <label class="wide">Remote Policy Server
                <input id="model-remote-policy-server" autocomplete="off" placeholder="desktop:8080" />
              </label>
              <label>Policy Type
                <input id="model-policy-type" autocomplete="off" />
              </label>
              <label>Action Chunk
                <input id="model-actions-per-chunk" type="number" min="1" step="1" />
              </label>
              <label>Image Size
                <input id="model-policy-image-size" type="number" min="1" step="1" />
              </label>
              <label>Remote Host
                <input id="model-remote-host" autocomplete="off" />
              </label>
              <label>Remote Dir
                <input id="model-remote-workdir" autocomplete="off" />
              </label>
              <label>Device
                <input id="model-device" autocomplete="off" />
              </label>
              <label>Duration
                <input id="model-duration" type="number" min="1" step="1" />
              </label>
              <label>FPS
                <input id="model-fps" type="number" min="1" step="1" />
              </label>
              <label>Grip Deadband
                <input id="model-gripper-deadband" type="number" min="0" step="0.1" />
              </label>
              <label>Grip Max Step
                <input id="model-gripper-max-step" type="number" min="0" step="0.1" />
              </label>
            </div>
            <div class="model-actions">
              <button id="start-model-button" type="button">Start</button>
              <button id="stop-model-button" type="button">Stop</button>
            </div>
          </div>
          <div id="model-note" class="note"></div>
          <pre id="model-log" class="model-log"></pre>
          <div class="recording-board-title">
            <span>Saved Rollouts</span>
            <span id="model-rollout-count">0 runs</span>
          </div>
          <ul id="model-rollouts" class="recordings"></ul>
        </section>
        <section class="panel">
          <div class="panel-title">
            <span>Teleop & Alignment</span>
            <span><span id="teleop-state">off</span> · <span id="alignment-state">-</span></span>
          </div>
          <div class="align-actions">
            <button id="teleop-button" type="button">Start Teleop</button>
            <button id="rest-button" type="button">Move Follower to Rest</button>
            <button id="set-rest-from-leader-button" type="button">Set Rest from Leader</button>
          </div>
          <div id="teleop-note" class="note"></div>
          <div id="alignment-note" class="note"></div>
        </section>
        <section class="panel">
          <div class="panel-title">
            <span>Follower Joints</span>
            <span id="joint-count">-</span>
          </div>
          <div id="joints" class="joints"></div>
        </section>
        <section class="panel">
          <div class="panel-title">
            <span>Leader Joints</span>
            <span id="leader-joint-count">-</span>
          </div>
          <div id="leader-joints" class="joints"></div>
        </section>
      </aside>
    </main>
  </div>
  <script>
    const feeds = document.getElementById('feeds');
    const goColumns = 'ABCDEFGHJKLMNOPQRST';
    const cameraImages = new Map();
    let lastTimestamp = 0;
    let boardError = '';
    let latestState = null;
    let replayRecording = null;
    let replayFrame = 0;
    let replayTimer = null;
    let recordingsLoaded = [];
    let modelRolloutsLoaded = [];
    let syntheticRecordingsLoaded = [];
    let modelDefaultsLoaded = false;
    let modelTransientNote = '';
    let modelTransientUntil = 0;

    function fmt(value, suffix = '') {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return `${Number(value).toFixed(3)}${suffix}`;
    }

    function ensureCamera(camera) {
      if (cameraImages.has(camera.name)) return;
      const panel = document.createElement('article');
      panel.className = 'panel camera';
      panel.innerHTML = `
        <div class="panel-title">
          <span>${camera.name}</span>
          <span class="panel-meta">
            <span data-role="health" class="badge">waiting</span>
            <span data-role="shape">${camera.width}x${camera.height}@${camera.fps}</span>
          </span>
        </div>
        <div class="camera-frame">
          <img alt="${camera.name} camera feed" />
          <div data-role="warning" class="camera-warning"></div>
        </div>
      `;
      const img = panel.querySelector('img');
      feeds.appendChild(panel);
      cameraImages.set(camera.name, {
        img,
        health: panel.querySelector('[data-role="health"]'),
        shape: panel.querySelector('[data-role="shape"]'),
        warning: panel.querySelector('[data-role="warning"]'),
      });
    }

    function cameraUrl(name) {
      if (replayRecording) {
        const prefix =
          replayRecording.kind === 'model_rollout'
            ? 'model_rollout_frame'
            : replayRecording.kind === 'synthetic_recording'
              ? 'synthetic_recording_frame'
              : 'recording_frame';
        return `/api/${prefix}/${encodeURIComponent(replayRecording.id)}/${encodeURIComponent(name)}.jpg?frame=${replayFrame}&t=${Date.now()}`;
      }
      const activeModel = latestState?.model_run?.active;
      const rotate = name === 'overhead' ? '&rotate=left' : '';
      if (activeModel?.preview?.available) {
        return `/api/model_preview/${encodeURIComponent(activeModel.preview.run_id)}/${encodeURIComponent(name)}.jpg?t=${Date.now()}${rotate}`;
      }
      return `/api/camera/${encodeURIComponent(name)}.jpg?t=${Date.now()}${rotate}`;
    }

    function deltaStatus(delta) {
      const abs = Math.abs(delta);
      return abs <= 3 ? 'ok' : abs <= 10 ? 'warn' : 'bad';
    }

    function buildLeaderDeltaMap(state) {
      const followerByName = new Map(state.joints.map(joint => [joint.name, joint]));
      const deltas = new Map();
      for (const leader of state.leader_joints) {
        const follower = followerByName.get(leader.name);
        if (!follower) continue;
        const followerRange = Math.max(follower.max_value - follower.min_value, 1e-6);
        const followerPct = Math.max(0, Math.min(100, ((follower.value - follower.min_value) / followerRange) * 100));
        deltas.set(leader.name, {
          delta: Number(follower.value) - Number(leader.value),
          targetPct: followerPct,
          unit: follower.unit,
        });
      }
      return deltas;
    }

    function updateJoints(joints, rootId, countId, deltaByName = new Map()) {
      const root = document.getElementById(rootId);
      root.innerHTML = '';
      for (const joint of joints) {
        const range = Math.max(joint.max_value - joint.min_value, 1e-6);
        const pct = Math.max(0, Math.min(100, ((joint.value - joint.min_value) / range) * 100));
        const delta = deltaByName.get(joint.name);
        const status = delta ? deltaStatus(delta.delta) : '';
        const bandLeft = delta ? Math.min(pct, delta.targetPct) : 0;
        const bandWidth = delta ? Math.abs(pct - delta.targetPct) : 0;
        const deltaMarkup = delta
          ? `<span class="joint-delta ${status}">${delta.delta >= 0 ? '+' : ''}${delta.delta.toFixed(2)} ${delta.unit}</span>`
          : '';
        const bandMarkup = delta
          ? `<span class="delta-band ${status}" style="left:${bandLeft}%; width:${Math.max(bandWidth, 0.8)}%"></span>
             <span class="target-marker ${status}" style="left:${delta.targetPct}%"></span>`
          : '';
        const row = document.createElement('div');
        row.className = 'joint';
        row.innerHTML = `
          <div class="joint-top">
            <strong>${joint.name}</strong>
            <span class="joint-values"><span>${Number(joint.value).toFixed(2)} ${joint.unit}</span>${deltaMarkup}</span>
          </div>
          <div class="bar">${bandMarkup}<div class="value-fill" style="width:${pct}%"></div></div>
        `;
        root.appendChild(row);
      }
      document.getElementById(countId).textContent = joints.length;
    }

    function replayTraceEntry(recording) {
      if (!recording) return null;
      const trace = Array.isArray(recording.task_trace) ? recording.task_trace : [];
      return trace.find(entry => Number(entry.index) === Number(replayFrame)) || trace[replayFrame] || null;
    }

    function replayTelemetryState(recording) {
      const entry = replayTraceEntry(recording);
      const telemetry = entry?.telemetry || {};
      const joints = Array.isArray(telemetry.joints) ? telemetry.joints : [];
      const leaderJoints = Array.isArray(telemetry.leader_joints) ? telemetry.leader_joints : [];
      return {
        timestamp: Date.now() / 1000,
        connected: false,
        mode: telemetry.mode || `${recording?.kind || 'recording'} replay`,
        fps: telemetry.fps || recording?.sample_hz || 0,
        joints,
        leader_joints: leaderJoints,
        teleop_enabled: false,
      };
    }

    function updateReplayJoints(recording) {
      const replayState = replayTelemetryState(recording);
      updateJoints(replayState.joints, 'joints', 'joint-count');
      updateJoints(replayState.leader_joints, 'leader-joints', 'leader-joint-count', buildLeaderDeltaMap(replayState));
    }

    function updateAlignment(state) {
      const followerByName = new Map(state.joints.map(joint => [joint.name, joint]));
      const leaderByName = new Map(state.leader_joints.map(joint => [joint.name, joint]));
      const sharedNames = state.joints.map(joint => joint.name).filter(name => leaderByName.has(name));
      const restButton = document.getElementById('rest-button');
      const setRestFromLeaderButton = document.getElementById('set-rest-from-leader-button');
      restButton.disabled = !state.connected || state.joints.length === 0;
      setRestFromLeaderButton.disabled = !state.connected || state.leader_joints.length === 0;

      if (sharedNames.length === 0) {
        document.getElementById('alignment-state').textContent = '-';
        document.getElementById('alignment-note').textContent = 'Leader telemetry is unavailable.';
        return;
      }

      const deltas = sharedNames.map(name => {
        const follower = followerByName.get(name);
        const leader = leaderByName.get(name);
        const delta = Number(follower.value) - Number(leader.value);
        return { name, delta, unit: follower.unit };
      });
      const maxAbs = Math.max(...deltas.map(item => Math.abs(item.delta)));
      document.getElementById('alignment-state').textContent = `${maxAbs.toFixed(1)} max`;
      document.getElementById('alignment-note').textContent =
        maxAbs <= 3
          ? 'Leader and follower are closely aligned.'
          : 'Use the colored deltas in Leader Joints to line up the leader.';
    }

    async function refreshState() {
      const response = await fetch('/api/state', { cache: 'no-store' });
      const state = await response.json();
      latestState = state;
      lastTimestamp = state.timestamp;

      document.getElementById('status-dot').classList.toggle('connected', state.connected);
      document.getElementById('status-text').textContent = state.connected ? 'Connected' : 'Disconnected';
      document.getElementById('mode').textContent = state.mode;
      document.getElementById('fps').textContent = `${Number(state.fps).toFixed(1)} Hz`;
      document.getElementById('camera-count').textContent = state.cameras.length;
      document.getElementById('age').textContent = `${Math.max(0, Date.now() / 1000 - state.timestamp).toFixed(1)} s`;
      document.getElementById('note').textContent = state.note || '';

      for (const camera of state.cameras) ensureCamera(camera);
      for (const camera of state.cameras) {
        const view = cameraImages.get(camera.name);
        view.img.src = cameraUrl(camera.name);
        const modelPreviewActive = Boolean(state.model_run?.active?.preview?.available);
        const healthClass = modelPreviewActive ? 'ok' : !camera.fresh ? 'bad' : camera.dark ? 'warn' : 'ok';
        const healthText = modelPreviewActive
          ? (camera.name === state.board?.camera ? 'model + target' : 'model')
          : !camera.fresh ? 'no frame' : camera.dark ? `dark ${camera.brightness}` : `ok ${camera.brightness}`;
        view.health.className = `badge ${healthClass}`;
        view.health.textContent = healthText;
        view.shape.textContent = camera.actual_width && camera.actual_height
          ? `${camera.actual_width}x${camera.actual_height}@${camera.measured_fps}`
          : `${camera.width}x${camera.height}@${camera.fps}`;
        const warningText = camera.dark
          ? 'This feed is very dark. Check the camera index, lens cover, exposure, or lighting.'
          : '';
        view.warning.textContent = warningText;
        view.warning.classList.toggle('visible', Boolean(warningText));
      }

      const teleopButton = document.getElementById('teleop-button');
      const canTeleop = state.connected && state.leader_joints.length > 0;
      teleopButton.disabled = !canTeleop;
      teleopButton.classList.toggle('active', state.teleop_enabled);
      teleopButton.textContent = state.teleop_enabled ? 'Stop Teleop' : 'Start Teleop';
      document.getElementById('teleop-state').textContent = state.teleop_enabled ? 'on' : 'off';
      document.getElementById('teleop-note').textContent = canTeleop
        ? 'Align the leader and follower before starting; Start Teleop sends leader joint targets to the follower.'
        : 'Connect leader and follower telemetry before teleoperation.';
      if (replayRecording) {
        updateReplayJoints(replayRecording);
      } else {
        const leaderDeltaMap = buildLeaderDeltaMap(state);
        updateJoints(state.joints, 'joints', 'joint-count');
        updateJoints(state.leader_joints, 'leader-joints', 'leader-joint-count', leaderDeltaMap);
      }
      updateAlignment(state);
      updateRecording(state.recording);
      updateModelRun(state.model_run);
    }

    function updateModelRun(modelRun) {
      const active = modelRun?.active || null;
      const last = modelRun?.last || null;
      const defaults = modelRun?.defaults || {};
      if (!modelDefaultsLoaded) {
        document.getElementById('model-policy-path').value = defaults.policy_path || '';
        document.getElementById('model-remote-policy-server').value = defaults.remote_policy_server || '';
        document.getElementById('model-policy-type').value = defaults.policy_type || 'act';
        document.getElementById('model-actions-per-chunk').value = defaults.actions_per_chunk || 20;
        document.getElementById('model-policy-image-size').value = defaults.policy_image_size || 224;
        document.getElementById('model-remote-host').value = defaults.remote_host || '';
        document.getElementById('model-remote-workdir').value = defaults.remote_workdir || '~/Developer/lerobot';
        document.getElementById('model-device').value = defaults.device || 'cuda';
        document.getElementById('model-duration').value = defaults.duration_s || 30;
        document.getElementById('model-fps').value = defaults.fps || 10;
        document.getElementById('model-gripper-deadband').value = defaults.gripper_deadband || 0;
        document.getElementById('model-gripper-max-step').value = defaults.gripper_max_step || 0;
        modelDefaultsLoaded = true;
      }

      const shown = active || last;
      document.getElementById('model-state').textContent = active ? active.status : (last?.status || 'idle');
      document.getElementById('model-note').textContent = Date.now() < modelTransientUntil
        ? modelTransientNote
        : active
        ? `${active.task} · ${Number(active.elapsed_s).toFixed(1)} s`
        : (modelRun?.message || 'Ready for policy rollout.');
      document.getElementById('model-log').textContent = shown?.log_tail?.length
        ? shown.log_tail.join('\n')
        : (shown?.command || '');
      document.getElementById('start-model-button').disabled = Boolean(active || latestState?.recording?.active);
      document.getElementById('stop-model-button').disabled = !active;
    }

    function updateRecording(recording) {
      const active = recording && recording.active;
      const recordButton = document.getElementById('record-button');
      const stopButton = document.getElementById('stop-record-button');
      recordButton.disabled = Boolean(active);
      stopButton.disabled = !active;
      recordButton.classList.toggle('recording', Boolean(active));
      if (active) {
        document.getElementById('recording-state').textContent = `rec ${active.samples}`;
        document.getElementById('recording-note').textContent =
          `${Number(active.elapsed_s).toFixed(1)} s · ${active.samples} samples · ${recording.message || ''}`;
        renderReplayTaskState(null);
        renderReplayDelta(null);
        return;
      }
      if (replayRecording) {
        document.getElementById('recording-state').textContent =
          replayRecording.kind === 'model_rollout'
            ? 'model replay'
            : replayRecording.kind === 'synthetic_recording'
              ? 'synthetic replay'
              : 'replay';
        return;
      }
      document.getElementById('recording-state').textContent = 'idle';
      document.getElementById('recording-note').textContent = recording?.message || 'Ready to record.';
      renderReplayTaskState(null);
      renderReplayDelta(null);
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      }[char]));
    }

    function formatStone(stone) {
      if (!stone) return '-';
      const confidence = stone.confidence === undefined || stone.confidence === null
        ? ''
        : ` · ${Number(stone.confidence).toFixed(3)}`;
      return `${escapeHtml(stone.color || '?')} at ${escapeHtml(stone.coord || '?')}${confidence}`;
    }

    function parseGoCoord(value) {
      const text = String(value || '').toUpperCase();
      const col = goColumns.indexOf(text[0]);
      const row = Number(text.slice(1)) - 1;
      if (col < 0 || row < 0 || row >= 19) return null;
      return { row, col, coord: text };
    }

    function stonePoint(stone) {
      if (!stone) return null;
      if (stone.coord) {
        const parsed = parseGoCoord(stone.coord);
        if (parsed) return { ...parsed, color: stone.color || '', confidence: stone.confidence };
      }
      const row = Number(stone.row);
      const col = Number(stone.col);
      if (!Number.isFinite(row) || !Number.isFinite(col) || row < 0 || row >= 19 || col < 0 || col >= 19) {
        return null;
      }
      return { row, col, coord: `${goColumns[col]}${row + 1}`, color: stone.color || '', confidence: stone.confidence };
    }

    function addBoardMarker(board, point, options = {}) {
      if (!point) return;
      const marker = document.createElement('div');
      marker.className = `sample-marker ${options.kind || ''} ${point.color ? `${point.color}-stone` : ''}`.trim();
      marker.style.setProperty('--row', point.row);
      marker.style.setProperty('--col', point.col);
      if (options.size) marker.style.setProperty('--size', `${options.size}px`);
      marker.textContent = options.text || '';
      marker.dataset.label = options.label || point.coord;
      board.appendChild(marker);
    }

    function renderCollectedBoard(recordings) {
      const board = document.getElementById('recording-samples-board');
      const title = document.getElementById('recording-board-title');
      const summary = document.getElementById('recording-board-count');
      const clearButton = document.getElementById('clear-selected-recording');
      title.textContent = 'Sample Distribution';
      clearButton.classList.remove('visible');
      board.innerHTML = '';
      const counts = new Map();
      let addedTotal = 0;
      let completedTotal = 0;
      for (const recording of recordings || []) {
        if (recording.status !== 'complete' || !recording.delta) continue;
        completedTotal += 1;
        const delta = recording.delta || {};
        const added = delta.added || [];
        for (const stone of added) {
          const point = stonePoint(stone);
          if (!point) continue;
          addedTotal += 1;
          const previous = counts.get(point.coord) || { ...point, count: 0 };
          previous.count += 1;
          counts.set(point.coord, previous);
        }
      }
      const maxCount = Math.max(1, ...Array.from(counts.values(), item => item.count));
      for (const item of counts.values()) {
        const size = 13 + Math.round((item.count / maxCount) * 13);
        addBoardMarker(board, item, {
          size,
          text: String(item.count),
          label: `${item.coord}: ${item.count} sample${item.count === 1 ? '' : 's'}`,
        });
      }
      summary.textContent = `${completedTotal} samples · ${addedTotal} moves`;
    }

    function renderSelectedSampleBoard(recording) {
      const board = document.getElementById('recording-samples-board');
      const title = document.getElementById('recording-board-title');
      const summary = document.getElementById('recording-board-count');
      const clearButton = document.getElementById('clear-selected-recording');
      board.innerHTML = '';
      const delta = recording?.board?.delta || recording?.delta || null;
      if (!recording || !delta) {
        renderCollectedBoard(recordingsLoaded);
        return;
      }
      title.textContent = recording.name || recording.id || 'Selected Sample';
      clearButton.classList.add('visible');
      const added = delta.added || [];
      const removed = delta.removed || [];
      const changed = delta.changed || [];
      for (const stone of added) {
        const point = stonePoint(stone);
        addBoardMarker(board, point, {
          kind: 'added',
          text: '+',
          label: `Added ${point?.color || stone.color || ''} ${point?.coord || stone.coord || '?'}`,
        });
      }
      for (const stone of removed) {
        const point = stonePoint(stone);
        addBoardMarker(board, point, {
          kind: 'removed',
          text: '-',
          label: `Removed ${point?.color || stone.color || ''} ${point?.coord || stone.coord || '?'}`,
        });
      }
      for (const item of changed) {
        const point = stonePoint(item);
        addBoardMarker(board, point, {
          kind: 'changed',
          text: '!',
          label: `Changed ${point?.coord || item.coord || '?'}`,
        });
      }
      summary.textContent = `${added.length} added · ${removed.length} removed`;
    }

    function renderReplayDelta(recording) {
      const root = document.getElementById('recording-delta');
      const delta = recording?.board?.delta || recording?.delta || null;
      if (!recording || !delta) {
        root.className = 'replay-delta';
        root.innerHTML = '';
        renderSelectedSampleBoard(null);
        return;
      }
      const added = delta.added || [];
      const removed = delta.removed || [];
      const changed = delta.changed || [];
      const unexpected = added.length !== 1 || removed.length > 0 || changed.length > 0;
      const rows = [];
      if (added.length) rows.push(...added.map(stone => `<li>Added ${formatStone(stone)}</li>`));
      if (removed.length) rows.push(...removed.map(stone => `<li>Removed ${formatStone(stone)}</li>`));
      if (changed.length) rows.push(...changed.map(item => `<li>Changed ${escapeHtml(item.coord || '?')}</li>`));
      if (rows.length === 0) rows.push('<li>No board delta detected</li>');
      root.className = `replay-delta visible${unexpected ? ' warn' : ''}`;
      root.innerHTML = `
        <div class="replay-delta-title">
          <span>Detected move</span>
          <span>${added.length} added</span>
        </div>
        <ul>${rows.join('')}</ul>
      `;
      renderSelectedSampleBoard(recording);
    }

    async function startRecording() {
      clearReplay();
      const response = await fetch('/api/recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start' })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('recording-note').textContent = result.error || 'Could not start recording.';
        return;
      }
      await refreshState();
      await refreshRecordings();
    }

    async function stopRecording() {
      const response = await fetch('/api/recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('recording-note').textContent = result.error || 'Could not stop recording.';
        return;
      }
      await refreshState();
      await refreshRecordings();
    }

    async function refreshRecordings() {
      const response = await fetch('/api/recordings', { cache: 'no-store' });
      const result = await response.json().catch(() => ({ recordings: [] }));
      recordingsLoaded = result.recordings || [];
      renderRecordings(recordingsLoaded);
      if (replayRecording) {
        renderSelectedSampleBoard(replayRecording);
      } else {
        renderCollectedBoard(recordingsLoaded);
      }
    }

    async function refreshModelRollouts() {
      const response = await fetch('/api/model_rollouts', { cache: 'no-store' });
      const result = await response.json().catch(() => ({ model_rollouts: [] }));
      modelRolloutsLoaded = result.model_rollouts || [];
      renderModelRollouts(modelRolloutsLoaded);
    }

    async function refreshSyntheticRecordings() {
      const response = await fetch('/api/synthetic_recordings', { cache: 'no-store' });
      const result = await response.json().catch(() => ({ synthetic_recordings: [] }));
      syntheticRecordingsLoaded = result.synthetic_recordings || [];
      renderSyntheticRecordings(syntheticRecordingsLoaded);
    }

    async function deleteRecording(id) {
      clearReplay();
      const button = document.querySelector(`[data-delete-recording="${CSS.escape(id)}"]`);
      if (button) button.disabled = true;
      const response = await fetch('/api/recordings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'delete', id })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (button) button.disabled = false;
      if (!response.ok || !result.ok) {
        document.getElementById('recording-note').textContent = result.error || 'Could not delete recording.';
        return;
      }
      recordingsLoaded = result.recordings || [];
      renderRecordings(recordingsLoaded);
      renderCollectedBoard(recordingsLoaded);
      document.getElementById('recording-note').textContent = result.message || 'Deleted recording.';
      await refreshState();
    }

    function renderRecordings(recordings) {
      const list = document.getElementById('recordings');
      list.innerHTML = '';
      if (recordings.length === 0) {
        const li = document.createElement('li');
        li.className = 'note';
        li.textContent = 'No recordings yet.';
        list.appendChild(li);
        return;
      }
      for (const item of recordings) {
        const li = document.createElement('li');
        li.className = 'recording-row';
        li.classList.toggle(
          'selected',
          replayRecording?.kind === 'recording' && (replayRecording?.id === item.id || replayRecording?.name === item.id),
        );
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `${item.name}: ${item.samples} samples, ${item.move_name}`;
        button.addEventListener('click', () => loadRecording(item.id));
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'delete-one';
        deleteButton.dataset.deleteRecording = item.id;
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', event => {
          event.stopPropagation();
          deleteRecording(item.id);
        });
        li.appendChild(button);
        li.appendChild(deleteButton);
        list.appendChild(li);
      }
    }

    function renderModelRollouts(modelRollouts) {
      const list = document.getElementById('model-rollouts');
      const count = document.getElementById('model-rollout-count');
      list.innerHTML = '';
      count.textContent = `${modelRollouts.length} run${modelRollouts.length === 1 ? '' : 's'}`;
      if (modelRollouts.length === 0) {
        const li = document.createElement('li');
        li.className = 'note';
        li.textContent = 'No saved rollouts yet.';
        list.appendChild(li);
        return;
      }
      for (const item of modelRollouts) {
        const li = document.createElement('li');
        li.className = 'recording-row';
        li.classList.toggle(
          'selected',
          replayRecording?.kind === 'model_rollout' && (replayRecording?.id === item.id || replayRecording?.name === item.id),
        );
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `${item.name}: ${item.samples} frames, ${item.move_name}`;
        button.addEventListener('click', () => loadModelRollout(item.id));
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'delete-one';
        deleteButton.dataset.deleteModelRollout = item.id;
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', event => {
          event.stopPropagation();
          deleteModelRollout(item.id);
        });
        li.appendChild(button);
        li.appendChild(deleteButton);
        list.appendChild(li);
      }
    }

    function renderSyntheticRecordings(syntheticRecordings) {
      const list = document.getElementById('synthetic-recordings');
      const count = document.getElementById('synthetic-recording-count');
      list.innerHTML = '';
      count.textContent = `${syntheticRecordings.length} run${syntheticRecordings.length === 1 ? '' : 's'}`;
      if (syntheticRecordings.length === 0) {
        const li = document.createElement('li');
        li.className = 'note';
        li.textContent = 'No synthetic recordings found.';
        list.appendChild(li);
        return;
      }
      for (const item of syntheticRecordings) {
        const li = document.createElement('li');
        li.className = 'recording-row';
        li.classList.toggle(
          'selected',
          replayRecording?.kind === 'synthetic_recording' && (replayRecording?.id === item.id || replayRecording?.name === item.id),
        );
        const button = document.createElement('button');
        button.type = 'button';
        const success = item.synthetic?.success === true ? 'success' : item.synthetic?.success === false ? 'miss' : 'synthetic';
        button.textContent = `${item.name}: ${item.samples} frames, ${item.move_name} · ${success}`;
        button.addEventListener('click', () => loadSyntheticRecording(item.id));
        li.appendChild(button);
        list.appendChild(li);
      }
    }

    function recordingCameraNames(recording) {
      const cameras = recording?.cameras || {};
      if (Array.isArray(cameras)) return cameras;
      return Object.keys(cameras);
    }

    function overheadProcessedNote(recording) {
      const status = recording?.overhead_processed || 'missing';
      if (status === 'ready') return 'overhead target overlay ready';
      if (status === 'processing') return 'processing overhead overlay...';
      if (status === 'error') return 'showing raw overhead; overlay processing failed';
      return 'showing raw overhead';
    }

    function taskStateForRecording(recording) {
      if (!recording) return null;
      const item = replayTraceEntry(recording);
      if (item) {
        return item.task_state || { done: Boolean(item.done), reason: '' };
      }
      const target = recording?.board?.task_state?.target || recording?.task_state?.target || null;
      return { done: false, target, reason: 'waiting for task completion' };
    }

    function renderReplayTaskState(recording) {
      const root = document.getElementById('recording-task-state');
      const taskState = taskStateForRecording(recording);
      if (!recording || !taskState) {
        root.className = 'replay-state';
        root.innerHTML = '';
        return;
      }
      const target = taskState.target || {};
      const targetText = target.coord
        ? `${escapeHtml(target.color || 'stone')} at ${escapeHtml(target.coord)}`
        : 'target unknown';
      root.className = 'replay-state visible';
      root.innerHTML = `
        <span>State: <strong>${taskState.done ? 'Done' : 'Not done'}</strong></span>
        <span>${targetText} · ${escapeHtml(taskState.reason || '')}</span>
      `;
    }

    async function loadRecording(id) {
      const response = await fetch(`/api/recording?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('recording-note').textContent = result.error || 'Could not load recording.';
        return;
      }
      stopReplayTimer();
      replayRecording = { ...result.recording, kind: 'recording' };
      replayFrame = 0;
      const maxFrame = Math.max(0, Number(replayRecording.samples || 1) - 1);
      const slider = document.getElementById('replay-slider');
      slider.max = String(maxFrame);
      slider.value = '0';
      for (const camera of recordingCameraNames(replayRecording)) {
        const meta = replayRecording.cameras?.[camera] || {};
        ensureCamera({
          name: camera,
          width: meta.width || 0,
          height: meta.height || 0,
          fps: meta.fps || replayRecording.sample_hz || 10,
        });
      }
      updateReplayFrame();
      renderRecordings(recordingsLoaded);
      renderModelRollouts(modelRolloutsLoaded);
      renderSyntheticRecordings(syntheticRecordingsLoaded);
      document.getElementById('recording-state').textContent = 'replay';
    }

    async function loadModelRollout(id) {
      const response = await fetch(`/api/model_rollout?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('model-note').textContent = result.error || 'Could not load model rollout.';
        return;
      }
      stopReplayTimer();
      replayRecording = { ...result.model_rollout, kind: 'model_rollout' };
      replayFrame = 0;
      const maxFrame = Math.max(0, Number(replayRecording.samples || 1) - 1);
      const slider = document.getElementById('replay-slider');
      slider.max = String(maxFrame);
      slider.value = '0';
      for (const camera of recordingCameraNames(replayRecording)) {
        const meta = replayRecording.cameras?.[camera] || {};
        ensureCamera({
          name: camera,
          width: meta.width || 0,
          height: meta.height || 0,
          fps: meta.fps || replayRecording.sample_hz || 10,
        });
      }
      updateReplayFrame();
      renderRecordings(recordingsLoaded);
      renderModelRollouts(modelRolloutsLoaded);
      renderSyntheticRecordings(syntheticRecordingsLoaded);
      document.getElementById('recording-state').textContent = 'model replay';
    }

    async function loadSyntheticRecording(id) {
      const response = await fetch(`/api/synthetic_recording?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('recording-note').textContent = result.error || 'Could not load synthetic recording.';
        return;
      }
      stopReplayTimer();
      replayRecording = { ...result.synthetic_recording, kind: 'synthetic_recording' };
      replayFrame = 0;
      const maxFrame = Math.max(0, Number(replayRecording.samples || 1) - 1);
      const slider = document.getElementById('replay-slider');
      slider.max = String(maxFrame);
      slider.value = '0';
      for (const camera of recordingCameraNames(replayRecording)) {
        const meta = replayRecording.cameras?.[camera] || {};
        ensureCamera({
          name: camera,
          width: meta.width || 0,
          height: meta.height || 0,
          fps: meta.fps || replayRecording.sample_hz || 10,
        });
      }
      updateReplayFrame();
      renderRecordings(recordingsLoaded);
      renderModelRollouts(modelRolloutsLoaded);
      renderSyntheticRecordings(syntheticRecordingsLoaded);
      document.getElementById('recording-state').textContent = 'synthetic replay';
    }

    async function deleteModelRollout(id) {
      clearReplay();
      const button = document.querySelector(`[data-delete-model-rollout="${CSS.escape(id)}"]`);
      if (button) button.disabled = true;
      const response = await fetch('/api/model_rollouts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'delete', id })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (button) button.disabled = false;
      if (!response.ok || !result.ok) {
        document.getElementById('model-note').textContent = result.error || 'Could not delete model rollout.';
        return;
      }
      modelRolloutsLoaded = result.model_rollouts || [];
      renderModelRollouts(modelRolloutsLoaded);
      document.getElementById('model-note').textContent = result.message || 'Deleted model rollout.';
      await refreshState();
    }

    function updateReplayFrame() {
      if (!replayRecording) return;
      const cameras = recordingCameraNames(replayRecording);
      for (const camera of cameras) {
        const view = cameraImages.get(camera);
        if (!view) continue;
        view.img.src = cameraUrl(camera);
        view.health.className = 'badge ok';
        view.health.textContent = 'saved';
        view.shape.textContent = `frame ${replayFrame + 1}/${Math.max(1, replayRecording.samples || 1)}`;
        view.warning.textContent = '';
        view.warning.classList.remove('visible');
      }
      document.getElementById('replay-slider').value = String(replayFrame);
      const replayText =
        `${replayRecording.name || replayRecording.id} · frame ${replayFrame + 1}/${Math.max(1, replayRecording.samples || 1)} · ${overheadProcessedNote(replayRecording)}`;
      document.getElementById('recording-note').textContent = replayText;
      if (replayRecording.kind === 'model_rollout') {
        document.getElementById('model-note').textContent = replayText;
      }
      renderReplayTaskState(replayRecording);
      renderReplayDelta(replayRecording);
      updateReplayJoints(replayRecording);
    }

    function stopReplayTimer() {
      if (!replayTimer) return;
      window.clearInterval(replayTimer);
      replayTimer = null;
      document.getElementById('replay-button').textContent = 'Play';
    }

    function clearReplay() {
      stopReplayTimer();
      replayRecording = null;
      replayFrame = 0;
      const slider = document.getElementById('replay-slider');
      slider.max = '0';
      slider.value = '0';
      document.getElementById('replay-button').textContent = 'Play';
      renderReplayTaskState(null);
      renderReplayDelta(null);
      renderRecordings(recordingsLoaded);
      renderModelRollouts(modelRolloutsLoaded);
      renderSyntheticRecordings(syntheticRecordingsLoaded);
    }

    function toggleReplay() {
      if (!replayRecording) return;
      if (replayTimer) {
        stopReplayTimer();
        return;
      }
      document.getElementById('replay-button').textContent = 'Pause';
      replayTimer = window.setInterval(() => {
        const maxFrame = Math.max(0, Number(replayRecording.samples || 1) - 1);
        replayFrame = replayFrame >= maxFrame ? 0 : replayFrame + 1;
        updateReplayFrame();
      }, Math.max(50, 1000 / Math.max(1, Number(replayRecording.sample_hz || 10))));
    }

    async function toggleRecordingFromKeyboard(event) {
      if (event.key !== 'Enter') return;
      const target = event.target;
      const tag = target && target.tagName ? target.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable) return;
      event.preventDefault();
      if (latestState?.recording?.active) {
        await stopRecording();
      } else {
        await startRecording();
      }
    }

    function updateBoard(board) {
      document.getElementById('board-camera').textContent = board.camera || '-';
      document.getElementById('board-baseline-count').textContent = board.baseline
        ? board.baseline.summary.total
        : '-';
      document.getElementById('board-current-count').textContent = board.current
        ? board.current.summary.total
        : '-';
      document.getElementById('board-added-count').textContent = board.delta
        ? board.delta.added.length
        : '-';
      document.getElementById('board-message').textContent = boardError || board.message || '';
      const list = document.getElementById('board-delta-list');
      list.innerHTML = '';
      if (!board.delta) return;
      const items = [
        ...board.delta.added.map(stone => `Added ${stone.color} at ${stone.coord}`),
        ...board.delta.removed.map(stone => `Removed ${stone.color} from ${stone.coord}`),
        ...board.delta.changed.map(item => `Changed ${item.coord}`)
      ];
      if (items.length === 0) items.push('No delta detected');
      for (const text of items) {
        const li = document.createElement('li');
        li.textContent = text;
        list.appendChild(li);
      }
    }

    async function toggleTeleop() {
      const current = document.getElementById('teleop-button').classList.contains('active');
      await fetch('/api/teleop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !current })
      });
      await refreshState();
    }

    async function moveFollowerToRest() {
      const response = await fetch('/api/follower_rest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      const note = document.getElementById('alignment-note');
      if (!response.ok || !result.ok) {
        note.textContent = result.error || 'Follower rest move failed.';
        return;
      }
      note.textContent = `Follower rest target sent: ${Object.keys(result.goals || {}).length} joints.`;
      await refreshState();
    }

    async function setRestFromLeader() {
      const response = await fetch('/api/rest_from_leader', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      const note = document.getElementById('alignment-note');
      if (!response.ok || !result.ok) {
        note.textContent = result.error || 'Could not set rest pose from leader.';
        return;
      }
      note.textContent = `Rest pose saved from leader: ${Object.keys(result.rest_position || {}).length} joints.`;
      await refreshState();
    }

    async function refreshDevices() {
      const button = document.getElementById('refresh-devices-button');
      const note = document.getElementById('note');
      button.disabled = true;
      note.textContent = 'Refreshing devices...';
      const response = await fetch('/api/refresh_devices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      button.disabled = false;
      const cameraOk = (result.cameras || []).filter(item => item.ok).length;
      const cameraTotal = (result.cameras || []).length;
      const leaderOk = Boolean(result.telemetry?.leader?.ok);
      const followerOk = Boolean(result.telemetry?.follower?.ok);
      note.textContent = response.ok
        ? `Refresh complete: cameras ${cameraOk}/${cameraTotal}, follower ${followerOk ? 'ok' : 'missing'}, leader ${leaderOk ? 'ok' : 'missing'}.`
        : (result.error || 'Device refresh failed.');
      await refreshState();
    }

    async function startModelRun() {
      const payload = {
        coord: document.getElementById('model-coord').value,
        color: document.getElementById('model-color').value,
        policy_path: document.getElementById('model-policy-path').value,
        remote_policy_server: document.getElementById('model-remote-policy-server').value,
        policy_type: document.getElementById('model-policy-type').value,
        actions_per_chunk: Number(document.getElementById('model-actions-per-chunk').value || 20),
        policy_image_size: Number(document.getElementById('model-policy-image-size').value || 224),
        remote_host: document.getElementById('model-remote-host').value,
        remote_workdir: document.getElementById('model-remote-workdir').value,
        device: document.getElementById('model-device').value,
        duration_s: Number(document.getElementById('model-duration').value || 30),
        fps: Number(document.getElementById('model-fps').value || 10),
        gripper_deadband: Number(document.getElementById('model-gripper-deadband').value || 0),
        gripper_max_step: Number(document.getElementById('model-gripper-max-step').value || 0),
      };
      const response = await fetch('/api/model_run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start', ...payload })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        modelTransientNote = result.error || 'Could not start model rollout.';
        modelTransientUntil = Date.now() + 5000;
        document.getElementById('model-note').textContent = modelTransientNote;
        return;
      }
      await refreshState();
      await refreshModelRollouts();
    }

    async function stopModelRun() {
      const response = await fetch('/api/model_run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' })
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        modelTransientNote = result.error || 'Could not stop model rollout.';
        modelTransientUntil = Date.now() + 5000;
        document.getElementById('model-note').textContent = modelTransientNote;
        return;
      }
      await refreshState();
      await refreshModelRollouts();
    }

    async function captureBoard(slot) {
      const response = await fetch('/api/board', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'capture', slot })
      });
      if (!response.ok) {
        const error = await response.json().catch(async () => ({ error: await response.text() }));
        boardError = error.error || 'Board capture failed';
        document.getElementById('board-message').textContent = boardError;
        return;
      }
      boardError = '';
      await refreshState();
    }

    async function computeBoardDelta() {
      const response = await fetch('/api/board', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'delta' })
      });
      if (!response.ok) {
        const error = await response.json().catch(async () => ({ error: await response.text() }));
        boardError = error.error || 'Board delta failed';
        document.getElementById('board-message').textContent = boardError;
        return;
      }
      boardError = '';
      await refreshState();
    }

    document.getElementById('teleop-button').addEventListener('click', toggleTeleop);
    document.getElementById('record-button').addEventListener('click', startRecording);
    document.getElementById('stop-record-button').addEventListener('click', stopRecording);
    document.getElementById('replay-button').addEventListener('click', toggleReplay);
    document.getElementById('clear-selected-recording').addEventListener('click', clearReplay);
    document.getElementById('replay-slider').addEventListener('input', event => {
      stopReplayTimer();
      replayFrame = Number(event.target.value || 0);
      updateReplayFrame();
    });
    document.addEventListener('keydown', event => {
      toggleRecordingFromKeyboard(event).catch(error => {
        document.getElementById('recording-note').textContent = error.message || 'Recording shortcut failed.';
      });
    });
    document.getElementById('refresh-devices-button').addEventListener('click', refreshDevices);
    document.getElementById('rest-button').addEventListener('click', moveFollowerToRest);
    document.getElementById('set-rest-from-leader-button').addEventListener('click', setRestFromLeader);
    document.getElementById('start-model-button').addEventListener('click', startModelRun);
    document.getElementById('stop-model-button').addEventListener('click', stopModelRun);

    setInterval(refreshState, 250);
    setInterval(refreshRecordings, 3000);
    setInterval(refreshModelRollouts, 3000);
    setInterval(refreshSyntheticRecordings, 3000);
    refreshState().catch(() => {
      document.getElementById('status-text').textContent = 'Waiting';
    });
    refreshRecordings();
    refreshModelRollouts();
    refreshSyntheticRecordings();
  </script>
</body>
</html>
"""


EVALUATOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Go Model Evaluator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --line: #d8dee3;
      --text: #182026;
      --muted: #66727c;
      --teal: #0f766e;
      --amber: #b7791f;
      --red: #b42318;
      --blue: #2364aa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 18px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 16px; line-height: 1; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 420px) minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 56px);
    }
    .panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel-title {
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
    }
    .session-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      text-decoration: none;
    }
    .form {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .wide { grid-column: 1 / -1; }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    input,
    select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 9px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
    }
    button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 9px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary { color: #fff; border-color: var(--teal); background: var(--teal); }
    button.danger { color: var(--red); border-color: #e7a39e; }
    button:disabled { color: var(--muted); cursor: not-allowed; background: #edf1f3; }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      min-width: 0;
    }
    .metric span { display: block; color: var(--muted); font-size: 11px; }
    .metric strong { display: block; font-size: 20px; line-height: 1.25; overflow-wrap: anywhere; }
    .note {
      min-height: 26px;
      padding: 0 10px 10px;
      color: var(--muted);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th,
    td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      line-height: 1.4;
      background: #f7fafb;
      white-space: nowrap;
    }
    .badge.ok { color: var(--teal); border-color: #9fd8d2; background: #eefaf8; }
    .badge.warn { color: var(--amber); border-color: #e8c780; background: #fff8e8; }
    .badge.bad { color: var(--red); border-color: #e7a39e; background: #fff3f2; }
    .sequence {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
    }
    .cmd {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfcfd;
      font-size: 12px;
      min-width: 0;
    }
    .cmd strong { display: block; font-size: 16px; }
    .cmd span { color: var(--muted); }
    .results {
      display: grid;
      gap: 14px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .sequence { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Go Model Evaluator</h1>
    <a class="session-link" href="/">Back to Recording</a>
  </header>
  <main>
    <section class="panel">
      <div class="panel-title">
        <span>Run</span>
        <span id="status">idle</span>
      </div>
      <div class="form">
        <div class="grid">
          <label class="wide">Policy Path
            <input id="policy-path" autocomplete="off" />
          </label>
          <label class="wide">Remote Policy Server
            <input id="remote-policy-server" autocomplete="off" placeholder="desktop:8080" />
          </label>
          <label>Policy Type
            <input id="policy-type" autocomplete="off" />
          </label>
          <label>Action Chunk
            <input id="actions-per-chunk" type="number" min="1" step="1" />
          </label>
          <label>Image Size
            <input id="policy-image-size" type="number" min="1" step="1" />
          </label>
          <label>Device
            <input id="device" autocomplete="off" />
          </label>
          <label>FPS
            <input id="fps" type="number" min="1" step="1" />
          </label>
          <label>Grip Deadband
            <input id="gripper-deadband" type="number" min="0" step="0.1" />
          </label>
          <label>Grip Max Step
            <input id="gripper-max-step" type="number" min="0" step="0.1" />
          </label>
        </div>
        <div class="actions">
          <button id="start" class="primary" type="button">Start Evaluation</button>
          <button id="stop" class="danger" type="button">Stop</button>
        </div>
      </div>
      <div id="message" class="note"></div>
    </section>
    <section class="results">
      <section class="panel">
        <div class="panel-title">
          <span>Summary</span>
          <span id="directory">-</span>
        </div>
        <div class="metrics">
          <div class="metric"><span>Success</span><strong id="successes">0</strong></div>
          <div class="metric"><span>Failed</span><strong id="failures">0</strong></div>
          <div class="metric"><span>Progress</span><strong id="progress">0/10</strong></div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-title">
          <span>Sequence</span>
          <span>OOD 10</span>
        </div>
        <div id="sequence" class="sequence"></div>
      </section>
      <section class="panel">
        <div class="panel-title">
          <span>Results</span>
          <span id="elapsed">-</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Move</th>
              <th>Status</th>
              <th>Reason</th>
              <th>Rollout</th>
            </tr>
          </thead>
          <tbody id="results"></tbody>
        </table>
      </section>
    </section>
  </main>
  <script>
    let defaultsLoaded = false;
    let commands = [];

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[ch]));
    }

    function payload() {
      return {
        action: 'start',
        commands,
        policy_path: document.getElementById('policy-path').value,
        remote_policy_server: document.getElementById('remote-policy-server').value,
        policy_type: document.getElementById('policy-type').value,
        actions_per_chunk: Number(document.getElementById('actions-per-chunk').value || 20),
        policy_image_size: Number(document.getElementById('policy-image-size').value || 224),
        device: document.getElementById('device').value,
        fps: Number(document.getElementById('fps').value || 10),
        gripper_deadband: Number(document.getElementById('gripper-deadband').value || 0),
        gripper_max_step: Number(document.getElementById('gripper-max-step').value || 0),
      };
    }

    function renderSequence(activeIndex, attempts) {
      const root = document.getElementById('sequence');
      const byIndex = new Map((attempts || []).map(item => [Number(item.index), item]));
      root.innerHTML = '';
      commands.forEach((command, index) => {
        const attempt = byIndex.get(index);
        const div = document.createElement('div');
        div.className = 'cmd';
        const state = attempt ? (attempt.success ? 'ok' : 'bad') : index === activeIndex ? 'warn' : '';
        div.innerHTML = `
          <strong>${index + 1}. ${escapeHtml(command.coord)}</strong>
          <span>${escapeHtml(command.color)} ${state ? `· ${state === 'ok' ? 'success' : state === 'bad' ? 'failed' : 'running'}` : ''}</span>
        `;
        root.appendChild(div);
      });
    }

    function renderResults(evaluator) {
      const tbody = document.getElementById('results');
      const attempts = evaluator?.attempts || [];
      tbody.innerHTML = '';
      for (const attempt of attempts) {
        const tr = document.createElement('tr');
        const status = attempt.success ? 'ok' : attempt.timed_out ? 'warn' : 'bad';
        tr.innerHTML = `
          <td>${Number(attempt.index) + 1}</td>
          <td>${escapeHtml(attempt.color)} to ${escapeHtml(attempt.coord)}</td>
          <td><span class="badge ${status}">${attempt.success ? 'succeeded' : attempt.timed_out ? 'timed out' : 'failed'}</span></td>
          <td>${escapeHtml(attempt.reason || '')}</td>
          <td>${escapeHtml(attempt.rollout || '')}</td>
        `;
        tbody.appendChild(tr);
      }
      if (attempts.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="5">No attempts yet.</td>';
        tbody.appendChild(tr);
      }
    }

    function update(data) {
      const evaluator = data.active || data.last || null;
      const defaults = data.defaults || {};
      if (!defaultsLoaded) {
        commands = defaults.commands || [];
        document.getElementById('policy-path').value = defaults.policy_path || '';
        document.getElementById('remote-policy-server').value = defaults.remote_policy_server || '';
        document.getElementById('policy-type').value = defaults.policy_type || 'act';
        document.getElementById('actions-per-chunk').value = defaults.actions_per_chunk || 20;
        document.getElementById('policy-image-size').value = defaults.policy_image_size || 224;
        document.getElementById('device').value = defaults.device || 'cuda';
        document.getElementById('fps').value = defaults.fps || 10;
        document.getElementById('gripper-deadband').value = defaults.gripper_deadband || 0;
        document.getElementById('gripper-max-step').value = defaults.gripper_max_step || 0;
        defaultsLoaded = true;
      }
      const active = Boolean(data.active);
      document.getElementById('status').textContent = evaluator?.status || 'idle';
      document.getElementById('message').textContent = evaluator?.message || data.message || '';
      document.getElementById('directory').textContent = evaluator?.directory || '-';
      document.getElementById('successes').textContent = evaluator?.successes ?? 0;
      document.getElementById('failures').textContent = evaluator?.failures ?? 0;
      document.getElementById('progress').textContent = `${evaluator?.attempts?.length || 0}/${evaluator?.total || 10}`;
      document.getElementById('elapsed').textContent = evaluator ? `${Number(evaluator.elapsed_s || 0).toFixed(1)} s` : '-';
      document.getElementById('start').disabled = active;
      document.getElementById('stop').disabled = !active;
      renderSequence(active ? Number(evaluator.index || 0) : -1, evaluator?.attempts || []);
      renderResults(evaluator);
    }

    async function refresh() {
      const response = await fetch('/api/evaluator', { cache: 'no-store' });
      update(await response.json());
    }

    async function start() {
      const response = await fetch('/api/evaluator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload()),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Could not start evaluator.';
        return;
      }
      await refresh();
    }

    async function stop() {
      await fetch('/api/evaluator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' }),
      });
      await refresh();
    }

    document.getElementById('start').addEventListener('click', start);
    document.getElementById('stop').addEventListener('click', stop);
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


ANNOTATION_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Go CV Annotation Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --line: #d8dee3;
      --text: #182026;
      --muted: #66727c;
      --teal: #0f766e;
      --red: #b42318;
      --blue: #2364aa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 18px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      line-height: 1;
    }
    main {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(430px, 560px);
      gap: 14px;
      padding: 14px;
      overflow: hidden;
    }
    .panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .camera-panel {
      display: grid;
      grid-template-rows: 40px minmax(0, 1fr);
      min-height: 0;
    }
    .panel-title {
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
    }
    .camera-frame {
      position: relative;
      min-height: 0;
      height: 100%;
      background: #101820;
    }
    .camera-frame img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      line-height: 1.4;
      background: #f7fafb;
      color: var(--muted);
    }
    .badge.ok { color: var(--teal); border-color: #9fd8d2; background: #eefaf8; }
    .badge.bad { color: var(--red); border-color: #e7a39e; background: #fff3f2; }
    .workbench {
      display: grid;
      grid-template-rows: auto auto auto auto auto minmax(320px, 1fr) auto;
      gap: 10px;
      padding: 0;
      min-height: 0;
      overflow: auto;
    }
    .workbench > :not(.panel-title) {
      margin-left: 10px;
      margin-right: 10px;
    }
    .workbench > :last-child {
      margin-bottom: 10px;
    }
    .tools {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      min-height: 38px;
    }
    .segmented {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    button {
      appearance: none;
      border: 0;
      border-right: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 9px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    button:last-child { border-right: 0; }
    button.active {
      color: #fff;
      background: var(--blue);
    }
    button.save {
      border: 1px solid var(--teal);
      border-radius: 8px;
      background: var(--teal);
      color: #fff;
    }
    button.secondary {
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .session-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      text-decoration: none;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-width: 0;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1.25;
    }
    .board-wrap {
      display: grid;
      place-items: center;
      min-height: 320px;
      padding: 6px;
      overflow: visible;
    }
    .board {
      --grid-pad: 5.2%;
      --grid-line: 3px;
      --grid-span: calc(100% - 2 * var(--grid-pad) - var(--grid-line));
      position: relative;
      width: min(100%, 520px);
      aspect-ratio: 1;
      background: #d7a85d;
      border: 2px solid #5f3d1f;
      border-radius: 6px;
      overflow: visible;
    }
    .board::before {
      content: "";
      position: absolute;
      inset: var(--grid-pad);
      border: var(--grid-line) solid #1f2933;
      background:
        repeating-linear-gradient(
          90deg,
          #1f2933 0 var(--grid-line),
          transparent var(--grid-line) calc((100% - var(--grid-line)) / 18)
        ),
        repeating-linear-gradient(
          0deg,
          #1f2933 0 var(--grid-line),
          transparent var(--grid-line) calc((100% - var(--grid-line)) / 18)
        );
      pointer-events: none;
    }
    .pt {
      position: absolute;
      left: calc(var(--grid-pad) + var(--grid-line) / 2 + var(--col) * var(--grid-span) / 18);
      top: calc(var(--grid-pad) + var(--grid-line) / 2 + var(--row) * var(--grid-span) / 18);
      width: calc(var(--grid-span) / 18);
      height: calc(var(--grid-span) / 18);
      transform: translate(-50%, -50%);
      border: 0;
      background: transparent;
      padding: 0;
      cursor: pointer;
      overflow: visible;
    }
    .pt::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 62%;
      height: 62%;
      transform: translate(-50%, -50%);
      border-radius: 999px;
    }
    .pt.black::before {
      background: radial-gradient(circle at 35% 30%, #34383b, #050607 68%);
      box-shadow: 0 1px 3px rgba(0,0,0,.35);
    }
    .pt.white::before {
      background: radial-gradient(circle at 35% 30%, #fff, #d8dde0 75%);
      border: 1px solid rgba(0,0,0,.2);
      box-shadow: 0 1px 3px rgba(0,0,0,.22);
    }
    .pt:hover::after {
      content: attr(data-coord);
      position: absolute;
      left: 50%;
      top: -4px;
      transform: translate(-50%, -100%);
      z-index: 5;
      background: #182026;
      color: #fff;
      border-radius: 4px;
      padding: 2px 5px;
      font-size: 11px;
      pointer-events: none;
    }
    .save-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
      font-size: 13px;
    }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
      font-size: 13px;
      background: #fff;
    }
    .preset-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto auto;
      gap: 8px;
    }
    .tuning {
      display: grid;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfcfd;
    }
    .tuning-row {
      display: grid;
      grid-template-columns: 128px minmax(0, 1fr) 48px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .tuning-row label {
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .tuning-row output {
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .tuning input[type="range"] {
      padding: 0;
    }
    .tuning-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .corner-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
    }
    .corner-box {
      display: grid;
      gap: 4px;
      font-size: 11px;
      color: var(--muted);
    }
    .corner-box strong {
      color: var(--text);
    }
    .corner-box input {
      padding: 6px;
      font-size: 12px;
    }
    .corner-dimensions {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
    }
    .corner-dimensions output {
      display: block;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
    }
    .note {
      min-height: 22px;
      color: var(--muted);
      font-size: 12px;
    }
    .saved {
      display: grid;
      gap: 6px;
      max-height: 145px;
      overflow: auto;
      padding: 0;
      margin: 0;
      list-style: none;
    }
    .saved li {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 6px;
      align-items: stretch;
    }
    .saved button {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
      background: #fbfcfd;
      color: var(--text);
      text-align: left;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .saved button:hover {
      border-color: var(--blue);
      background: #f2f7ff;
    }
    .saved button.delete-snapshot {
      width: auto;
      min-width: 68px;
      text-align: center;
      color: #a13b31;
      background: #fffafa;
    }
    .saved button.delete-snapshot:hover {
      border-color: #c84b40;
      background: #fff0ee;
    }
    @media (max-width: 980px) {
      main {
        height: auto;
        min-height: calc(100vh - 56px);
        grid-template-columns: 1fr;
        overflow: visible;
      }
      .camera-panel { min-height: 520px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Go CV Annotation Dashboard</h1>
    <span id="camera-status" class="badge">waiting</span>
  </header>
  <main>
    <section class="panel camera-panel">
      <div class="panel-title">
        <span id="camera-title">overhead</span>
        <span><span id="camera-shape">-</span> · <span id="board-orientation">0 deg</span></span>
      </div>
      <div class="camera-frame">
        <img id="camera-feed" alt="overhead camera feed" />
      </div>
    </section>
    <section class="panel workbench">
      <div class="panel-title">
        <span>Annotate</span>
        <a class="session-link" href="/">Back to Recording</a>
      </div>
      <div>
        <div class="preset-row">
          <select id="preset"></select>
          <button id="load-preset" class="secondary" type="button">Load</button>
          <button id="detect-once" class="secondary" type="button">Detect</button>
          <button id="toggle-autodetect" class="secondary" type="button">Autodetect On</button>
        </div>
      </div>
      <div class="tuning">
        <div class="tuning-row">
          <label for="tune-fisheye">Grid curve</label>
          <input id="tune-fisheye" type="range" min="-0.5" max="0.5" step="0.01" />
          <output id="tune-fisheye-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-radius">Color sample</label>
          <input id="tune-radius" type="range" min="0.18" max="0.44" step="0.01" />
          <output id="tune-radius-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-min-radius">Min radius</label>
          <input id="tune-min-radius" type="range" min="0.10" max="0.40" step="0.01" />
          <output id="tune-min-radius-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-max-radius">Max radius</label>
          <input id="tune-max-radius" type="range" min="0.25" max="0.65" step="0.01" />
          <output id="tune-max-radius-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-circularity">Min roundness</label>
          <input id="tune-circularity" type="range" min="0.30" max="0.90" step="0.01" />
          <output id="tune-circularity-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-snap">Max snap dist</label>
          <input id="tune-snap" type="range" min="0.20" max="0.80" step="0.01" />
          <output id="tune-snap-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-black-edge">Black edge req</label>
          <input id="tune-black-edge" type="range" min="0.12" max="0.78" step="0.01" />
          <output id="tune-black-edge-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-black">Black max L</label>
          <input id="tune-black" type="range" min="50" max="115" step="1" />
          <output id="tune-black-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-white">White min L</label>
          <input id="tune-white" type="range" min="145" max="190" step="1" />
          <output id="tune-white-value">-</output>
        </div>
        <div class="tuning-row">
          <label for="tune-saturation">White max S</label>
          <input id="tune-saturation" type="range" min="45" max="115" step="1" />
          <output id="tune-saturation-value">-</output>
        </div>
        <div class="corner-grid">
          <label class="corner-box"><strong>TL</strong><input id="corner-tl-x" type="number" step="1" /><input id="corner-tl-y" type="number" step="1" /></label>
          <label class="corner-box"><strong>TR</strong><input id="corner-tr-x" type="number" step="1" /><input id="corner-tr-y" type="number" step="1" /></label>
          <label class="corner-box"><strong>BR</strong><input id="corner-br-x" type="number" step="1" /><input id="corner-br-y" type="number" step="1" /></label>
          <label class="corner-box"><strong>BL</strong><input id="corner-bl-x" type="number" step="1" /><input id="corner-bl-y" type="number" step="1" /></label>
        </div>
        <div class="corner-dimensions">
          <span>top<output id="corner-width-top">-</output></span>
          <span>bottom<output id="corner-width-bottom">-</output></span>
          <span>left<output id="corner-height-left">-</output></span>
          <span>right<output id="corner-height-right">-</output></span>
        </div>
        <div class="tuning-actions">
          <button id="reset-tuning" class="secondary" type="button" onclick="resetTuning()" onpointerdown="resetTuning()">Reset Tuning</button>
          <button id="save-tuning" class="secondary" type="button">Save Tuning</button>
        </div>
      </div>
      <div class="tools">
        <div class="segmented">
          <button id="tool-black" class="active" type="button">Black</button>
          <button id="tool-white" type="button">White</button>
          <button id="tool-erase" type="button">Erase</button>
        </div>
        <button id="clear" class="secondary" type="button">Clear</button>
      </div>
      <div class="summary">
        <div class="metric"><span>Black</span><strong id="black-count">0</strong></div>
        <div class="metric"><span>White</span><strong id="white-count">0</strong></div>
        <div class="metric"><span>Total</span><strong id="total-count">0</strong></div>
      </div>
      <div class="board-wrap">
        <div id="board" class="board"></div>
      </div>
      <div>
        <div class="save-row">
          <input id="label" placeholder="snapshot label" />
          <button id="save" class="save" type="button">Save</button>
          <button id="refresh-saved" class="secondary" type="button">Refresh</button>
        </div>
        <div id="message" class="note"></div>
        <ul id="saved" class="saved"></ul>
      </div>
    </section>
  </main>
  <script>
    const columns = 'ABCDEFGHJKLMNOPQRST';
    const board = document.getElementById('board');
    const stones = new Map();
    let tool = 'black';
    let boardCamera = 'overhead';
    let selectedAnnotation = null;
    let autodetectEnabled = true;
    let oneShotOverlayUntil = 0;
    let cvTuningLoaded = false;
    let boardCornersLoaded = false;
    let liveTuningTimer = null;
    let liveTuningRequestId = 0;
    let cameraImageLoading = false;
    let cameraImageQueued = false;
    let detectionRunning = false;
    let detectionQueued = false;
    let cameraRawWidth = 1920;
    let cameraRawHeight = 1080;
    const cameraRefreshMs = 300;
    const detectionRefreshMs = 650;
    const tuningControls = {
      overlay_fisheye_k: ['tune-fisheye', 'tune-fisheye-value', 2],
      sample_radius_ratio: ['tune-radius', 'tune-radius-value', 2],
      stone_min_radius_ratio: ['tune-min-radius', 'tune-min-radius-value', 2],
      stone_max_radius_ratio: ['tune-max-radius', 'tune-max-radius-value', 2],
      stone_min_circularity: ['tune-circularity', 'tune-circularity-value', 2],
      stone_max_snap_distance_ratio: ['tune-snap', 'tune-snap-value', 2],
      black_grid_min_edge_score: ['tune-black-edge', 'tune-black-edge-value', 2],
      black_l_threshold: ['tune-black', 'tune-black-value', 0],
      white_l_threshold: ['tune-white', 'tune-white-value', 0],
      white_s_threshold: ['tune-saturation', 'tune-saturation-value', 0],
    };
    const cornerControls = [
      ['corner-tl-x', 'corner-tl-y'],
      ['corner-tr-x', 'corner-tr-y'],
      ['corner-br-x', 'corner-br-y'],
      ['corner-bl-x', 'corner-bl-y'],
    ];
    const presets = [
      {
        name: '01 empty board',
        stones: [],
      },
      {
        name: '02 single black center',
        stones: [{ coord: 'K10', color: 'black' }],
      },
      {
        name: '03 single white upper left',
        stones: [{ coord: 'D4', color: 'white' }],
      },
      {
        name: '04 four corners',
        stones: [
          { coord: 'A1', color: 'black' },
          { coord: 'T1', color: 'white' },
          { coord: 'A19', color: 'white' },
          { coord: 'T19', color: 'black' },
        ],
      },
      {
        name: '05 edge probes',
        stones: [
          { coord: 'K1', color: 'black' },
          { coord: 'A10', color: 'white' },
          { coord: 'T10', color: 'black' },
          { coord: 'K19', color: 'white' },
        ],
      },
      {
        name: '06 adjacent horizontal',
        stones: [
          { coord: 'H10', color: 'black' },
          { coord: 'J10', color: 'white' },
          { coord: 'K10', color: 'black' },
          { coord: 'L10', color: 'white' },
        ],
      },
      {
        name: '07 adjacent vertical',
        stones: [
          { coord: 'K7', color: 'white' },
          { coord: 'K8', color: 'black' },
          { coord: 'K9', color: 'white' },
          { coord: 'K10', color: 'black' },
        ],
      },
      {
        name: '08 diagonal scatter',
        stones: [
          { coord: 'C3', color: 'black' },
          { coord: 'F6', color: 'white' },
          { coord: 'K10', color: 'black' },
          { coord: 'Q16', color: 'white' },
        ],
      },
      {
        name: '09 mixed midgame',
        stones: [
          { coord: 'D4', color: 'black' },
          { coord: 'Q4', color: 'white' },
          { coord: 'K5', color: 'black' },
          { coord: 'H9', color: 'white' },
          { coord: 'K10', color: 'white' },
          { coord: 'N10', color: 'black' },
          { coord: 'C16', color: 'white' },
          { coord: 'Q16', color: 'black' },
        ],
      },
      {
        name: '10 bowl and arm side',
        stones: [
          { coord: 'B2', color: 'white' },
          { coord: 'C3', color: 'black' },
          { coord: 'D17', color: 'black' },
          { coord: 'E18', color: 'white' },
          { coord: 'R2', color: 'black' },
          { coord: 'S18', color: 'white' },
        ],
      },
    ];

    function coord(row, col) {
      return `${columns[col]}${row + 1}`;
    }

    function parseCoord(value) {
      const col = columns.indexOf(value[0]);
      const row = Number(value.slice(1)) - 1;
      if (col < 0 || row < 0 || row >= 19) throw new Error(`Invalid coordinate ${value}`);
      return { row, col, coord: value };
    }

    function stoneKey(row, col) {
      return `${row},${col}`;
    }

    function setStones(nextStones) {
      stones.clear();
      for (const stone of nextStones) {
        const parsed = stone.coord ? parseCoord(stone.coord) : stone;
        const row = Number(parsed.row);
        const col = Number(parsed.col);
        const color = String(stone.color || parsed.color);
        stones.set(stoneKey(row, col), { row, col, color, coord: coord(row, col) });
      }
      renderBoard();
    }

    function selectedPreset() {
      const index = Number(document.getElementById('preset').value);
      return presets[index] || presets[0];
    }

    function populatePresets() {
      const select = document.getElementById('preset');
      select.innerHTML = '';
      for (const [index, preset] of presets.entries()) {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${preset.name} (${preset.stones.length})`;
        select.appendChild(option);
      }
    }

    function buildBoard() {
      board.innerHTML = '';
      for (let row = 0; row < 19; row += 1) {
        for (let col = 0; col < 19; col += 1) {
          const point = document.createElement('button');
          point.type = 'button';
          point.className = 'pt';
          point.dataset.row = row;
          point.dataset.col = col;
          point.dataset.coord = coord(row, col);
          point.style.setProperty('--row', row);
          point.style.setProperty('--col', col);
          point.addEventListener('click', () => togglePoint(row, col));
          board.appendChild(point);
        }
      }
    }

    function togglePoint(row, col) {
      const key = stoneKey(row, col);
      if (tool === 'erase') {
        stones.delete(key);
      } else {
        stones.set(key, { row, col, color: tool, coord: coord(row, col) });
      }
      renderBoard();
    }

    function setTool(next) {
      tool = next;
      for (const id of ['black', 'white', 'erase']) {
        document.getElementById(`tool-${id}`).classList.toggle('active', id === tool);
      }
    }

    function renderBoard() {
      for (const point of board.querySelectorAll('.pt')) {
        const key = `${point.dataset.row},${point.dataset.col}`;
        const stone = stones.get(key);
        point.classList.toggle('black', stone?.color === 'black');
        point.classList.toggle('white', stone?.color === 'white');
      }
      const values = [...stones.values()];
      document.getElementById('black-count').textContent = values.filter(stone => stone.color === 'black').length;
      document.getElementById('white-count').textContent = values.filter(stone => stone.color === 'white').length;
      document.getElementById('total-count').textContent = values.length;
    }

    function readTuningControls() {
      const payload = {};
      for (const [name, [inputId]] of Object.entries(tuningControls)) {
        payload[name] = Number(document.getElementById(inputId).value);
      }
      payload.corners_tl_tr_br_bl = readCornerControls();
      return payload;
    }

    function setTuningControls(tuning, force = false) {
      if (!tuning || (cvTuningLoaded && !force)) return;
      for (const [name, [inputId, outputId, decimals]] of Object.entries(tuningControls)) {
        const value = Number(tuning[name]);
        if (!Number.isFinite(value)) continue;
        const input = document.getElementById(inputId);
        const output = document.getElementById(outputId);
        input.value = value;
        output.value = value.toFixed(decimals);
      }
      cvTuningLoaded = true;
    }

    function cameraToDisplayPoint(point) {
      const x = Number(point[0]);
      const y = Number(point[1]);
      if (boardCamera === 'overhead') return [y, cameraRawWidth - x];
      return [x, y];
    }

    function displayToCameraPoint(point) {
      const x = Number(point[0]);
      const y = Number(point[1]);
      if (boardCamera === 'overhead') return [cameraRawWidth - y, x];
      return [x, y];
    }

    function orderCorners(points) {
      if (!points || points.length !== 4) return points;
      const ordered = Array(4);
      const sums = points.map(point => Number(point[0]) + Number(point[1]));
      const diffs = points.map(point => Number(point[1]) - Number(point[0]));
      ordered[0] = points[sums.indexOf(Math.min(...sums))];
      ordered[2] = points[sums.indexOf(Math.max(...sums))];
      ordered[1] = points[diffs.indexOf(Math.min(...diffs))];
      ordered[3] = points[diffs.indexOf(Math.max(...diffs))];
      return ordered;
    }

    function cameraCornersToDisplayCorners(corners) {
      return orderCorners(corners.map(cameraToDisplayPoint));
    }

    function displayCornersToCameraCorners(corners) {
      return orderCorners(corners.map(displayToCameraPoint));
    }

    function readCornerControls() {
      const displayCorners = readDisplayCornerControls();
      updateCornerDimensions(displayCorners);
      return displayCornersToCameraCorners(displayCorners);
    }

    function readDisplayCornerControls() {
      return cornerControls.map(([xId, yId]) => [
        Number(document.getElementById(xId).value),
        Number(document.getElementById(yId).value),
      ]);
    }

    function setCornerControls(corners, force = false) {
      if (!corners || (boardCornersLoaded && !force)) return;
      const displayCorners = cameraCornersToDisplayCorners(corners);
      for (const [index, [xId, yId]] of cornerControls.entries()) {
        const point = displayCorners[index];
        if (!point) continue;
        document.getElementById(xId).value = Math.round(Number(point[0]));
        document.getElementById(yId).value = Math.round(Number(point[1]));
      }
      boardCornersLoaded = true;
      updateCornerDimensions(readDisplayCornerControls());
    }

    function pointDistance(a, b) {
      if (!a || !b) return NaN;
      const dx = Number(b[0]) - Number(a[0]);
      const dy = Number(b[1]) - Number(a[1]);
      return Math.hypot(dx, dy);
    }

    function updateCornerDimensions(corners = readDisplayCornerControls()) {
      const dimensions = {
        'corner-width-top': pointDistance(corners[0], corners[1]),
        'corner-width-bottom': pointDistance(corners[3], corners[2]),
        'corner-height-left': pointDistance(corners[0], corners[3]),
        'corner-height-right': pointDistance(corners[1], corners[2]),
      };
      for (const [id, value] of Object.entries(dimensions)) {
        document.getElementById(id).value = Number.isFinite(value) ? `${Math.round(value)} px` : '-';
      }
    }

    function updateTuningReadouts() {
      for (const [_name, [inputId, outputId, decimals]] of Object.entries(tuningControls)) {
        const input = document.getElementById(inputId);
        document.getElementById(outputId).value = Number(input.value).toFixed(decimals);
      }
    }

    function cameraFeedUrl(camera) {
      if (selectedAnnotation) return annotationFeedUrl();
      const rotate = camera.name === 'overhead' ? '&rotate=left' : '';
      const shouldOverlay = autodetectEnabled || Date.now() < oneShotOverlayUntil;
      if (shouldOverlay && camera.name === boardCamera) {
        return `/api/board_overlay.jpg?t=${Date.now()}${rotate}`;
      }
      return `/api/camera/${encodeURIComponent(camera.name)}.jpg?t=${Date.now()}${rotate}`;
    }

    function annotationFeedUrl() {
      const rotate = boardCamera === 'overhead' ? '&rotate=left' : '';
      const corners = encodeURIComponent(JSON.stringify(readCornerControls()));
      const curve = encodeURIComponent(String(Number(document.getElementById('tune-fisheye').value)));
      return `/api/annotation_overlay.jpg?file=${encodeURIComponent(selectedAnnotation.file)}&corners=${corners}&overlay_fisheye_k=${curve}${rotate}&t=${Date.now()}`;
    }

    function refreshCameraFrame({ force = false } = {}) {
      const feed = document.getElementById('camera-feed');
      if (cameraImageLoading && !force) {
        cameraImageQueued = true;
        return;
      }
      cameraImageLoading = true;
      cameraImageQueued = false;
      feed.onload = feed.onerror = () => {
        cameraImageLoading = false;
        if (cameraImageQueued) refreshCameraFrame();
      };
      feed.src = cameraFeedUrl({ name: boardCamera });
    }

    function updateAutodetectButton() {
      const button = document.getElementById('toggle-autodetect');
      if (selectedAnnotation) {
        button.textContent = 'Live Camera';
        button.classList.remove('active');
        return;
      }
      button.textContent = autodetectEnabled ? 'Autodetect On' : 'Autodetect Off';
      button.classList.toggle('active', autodetectEnabled);
    }

    async function applyTuning({ persist = false, detect = false } = {}) {
      if (selectedAnnotation) {
        if (persist) await saveSelectedAnnotationMetadata();
        refreshCameraFrame({ force: true });
        return;
      }
      const response = await fetch('/api/board_tuning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...readTuningControls(), persist }),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Tuning update failed';
        return;
      }
      if (result.board?.tuning) setTuningControls(result.board.tuning, true);
      if (result.board?.corners_tl_tr_br_bl) setCornerControls(result.board.corners_tl_tr_br_bl, true);
      document.getElementById('message').textContent = persist ? 'CV tuning saved.' : 'CV tuning applied.';
      if (detect) await detectCurrent({ silent: true, force: true, showOverlay: true });
      refreshCameraFrame();
    }

    async function resetTuning() {
      if (selectedAnnotation) {
        await openAnnotation(selectedAnnotation.file, { forceReload: true });
        return;
      }
      window.clearTimeout(liveTuningTimer);
      liveTuningRequestId += 1;
      const response = await fetch('/api/board_tuning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reset: true }),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Tuning reset failed';
        return;
      }
      if (result.board?.tuning) setTuningControls(result.board.tuning, true);
      if (result.board?.corners_tl_tr_br_bl) setCornerControls(result.board.corners_tl_tr_br_bl, true);
      document.getElementById('message').textContent = 'CV tuning reset to saved values.';
      await detectCurrent({ silent: true, force: true, showOverlay: true });
      refreshCameraFrame({ force: true });
    }

    function scheduleLiveTuning() {
      updateTuningReadouts();
      updateCornerDimensions();
      if (selectedAnnotation) {
        refreshCameraFrame({ force: true });
        return;
      }
      const requestId = liveTuningRequestId + 1;
      liveTuningRequestId = requestId;
      window.clearTimeout(liveTuningTimer);
      liveTuningTimer = window.setTimeout(async () => {
        try {
          const response = await fetch('/api/board_tuning', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...readTuningControls(), persist: false }),
          });
          if (!response.ok || requestId !== liveTuningRequestId) return;
          await response.json().catch(() => null);
          refreshCameraFrame();
          if (autodetectEnabled) detectCurrent({ silent: true });
        } catch (_error) {
          // The next explicit apply/save will surface any real tuning error.
        }
      }, 120);
    }

    async function refreshState() {
      const response = await fetch('/api/state', { cache: 'no-store' });
      const state = await response.json();
      boardCamera = state.board?.camera || 'overhead';
      if (selectedAnnotation) {
        document.getElementById('camera-title').textContent = selectedAnnotation.file;
        document.getElementById('board-orientation').textContent =
          `saved metadata ${selectedAnnotation.rotation ?? 0} deg`;
        document.getElementById('camera-shape').textContent = selectedAnnotation.shape
          ? `${selectedAnnotation.shape.width}x${selectedAnnotation.shape.height}`
          : 'saved image';
        const status = document.getElementById('camera-status');
        status.className = 'badge ok';
        status.textContent = 'saved';
        return;
      }
      document.getElementById('board-orientation').textContent =
        `camera to robot ${state.board?.camera_to_robot_rotation_degrees ?? 0} deg`;
      const camera = state.cameras.find(item => item.name === boardCamera) || state.cameras[0];
      if (!camera) return;
      cameraRawWidth = Number(camera.actual_width || camera.width || cameraRawWidth);
      cameraRawHeight = Number(camera.actual_height || camera.height || cameraRawHeight);
      setTuningControls(state.board?.tuning);
      setCornerControls(state.board?.corners_tl_tr_br_bl);
      document.getElementById('camera-title').textContent = camera.name;
      refreshCameraFrame();
      document.getElementById('camera-shape').textContent = camera.actual_width && camera.actual_height
        ? `${camera.actual_width}x${camera.actual_height}@${camera.measured_fps}`
        : `${camera.width}x${camera.height}@${camera.fps}`;
      const status = document.getElementById('camera-status');
      status.className = `badge ${camera.fresh ? 'ok' : 'bad'}`;
      status.textContent = camera.fresh ? `live ${camera.brightness}` : 'no frame';
    }

    async function saveAnnotation() {
      const payload = {
        label: document.getElementById('label').value,
        stones: [...stones.values()].sort((a, b) => a.row - b.row || a.col - b.col),
      };
      const response = await fetch('/api/annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Save failed';
        return;
      }
      const summary = result.annotation.summary;
      document.getElementById('message').textContent =
        `Saved ${summary.total} stones (${summary.black} black, ${summary.white} white).`;
      stones.clear();
      document.getElementById('label').value = '';
      renderBoard();
      await refreshSaved();
    }

    async function detectCurrent({ silent = false, force = false, showOverlay = false } = {}) {
      if (selectedAnnotation) {
        if (!silent) document.getElementById('message').textContent = 'Detect is disabled while viewing a saved annotation.';
        return;
      }
      if (detectionRunning && !force) {
        detectionQueued = true;
        return;
      }
      detectionRunning = true;
      detectionQueued = false;
      try {
        const response = await fetch('/api/board', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'detect' }),
        });
        const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
        if (!response.ok || !result.ok) {
          if (!silent) document.getElementById('message').textContent = result.error || 'Detection failed';
          return;
        }
        setStones(result.state.stones);
        const summary = result.state.summary;
        document.getElementById('message').textContent =
          `Live detection: ${summary.total} stones (${summary.black} black, ${summary.white} white).`;
        if (showOverlay) oneShotOverlayUntil = Date.now() + 2200;
        refreshCameraFrame();
      } catch (error) {
        if (!silent) document.getElementById('message').textContent = String(error);
      } finally {
        detectionRunning = false;
        if (detectionQueued) detectCurrent({ silent: true });
      }
    }

    async function refreshSaved() {
      const response = await fetch('/api/annotations', { cache: 'no-store' });
      const result = await response.json();
      const list = document.getElementById('saved');
      list.innerHTML = '';
      for (const item of result.annotations.slice().reverse()) {
        const li = document.createElement('li');
        const summary = item.summary || {};
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `${item.file}: ${summary.total || 0} stones, ${summary.black || 0} black, ${summary.white || 0} white`;
        button.title = 'View saved photo with metadata overlay';
        button.classList.toggle('active', selectedAnnotation?.file === item.file);
        button.addEventListener('click', () => openAnnotation(item.file));
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'delete-snapshot';
        deleteButton.textContent = 'Delete';
        deleteButton.title = 'Delete this saved snapshot and annotation';
        deleteButton.addEventListener('click', event => {
          event.stopPropagation();
          deleteAnnotation(item.file);
        });
        li.appendChild(button);
        li.appendChild(deleteButton);
        list.appendChild(li);
      }
    }

    async function openAnnotation(file, { forceReload = false } = {}) {
      if (selectedAnnotation?.file === file && !forceReload) return;
      const response = await fetch(`/api/annotation?file=${encodeURIComponent(file)}`, { cache: 'no-store' });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Could not load annotation';
        return;
      }
      const annotation = result.annotation;
      selectedAnnotation = {
        file: result.file,
        shape: result.image_shape,
        rotation: annotation.camera_to_robot_rotation_degrees || 0,
      };
      cameraRawWidth = Number(result.image_shape?.width || cameraRawWidth);
      cameraRawHeight = Number(result.image_shape?.height || cameraRawHeight);
      setTuningControls({ overlay_fisheye_k: annotation.overlay_fisheye_k ?? 0 }, true);
      setCornerControls(annotation.corners_tl_tr_br_bl, true);
      setStones(annotation.stones || []);
      document.getElementById('label').value = annotation.label || '';
      document.getElementById('message').textContent = `Viewing ${result.file}. Adjust Grid curve or corners, then Save Tuning to write metadata.`;
      updateAutodetectButton();
      refreshCameraFrame({ force: true });
      await refreshSaved();
    }

    async function saveSelectedAnnotationMetadata() {
      const response = await fetch('/api/annotation_metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file: selectedAnnotation.file,
          corners_tl_tr_br_bl: readCornerControls(),
          overlay_fisheye_k: Number(document.getElementById('tune-fisheye').value),
        }),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Could not save annotation metadata';
        return;
      }
      document.getElementById('message').textContent = `Saved metadata for ${result.file}.`;
      await openAnnotation(result.file, { forceReload: true });
    }

    async function deleteAnnotation(file) {
      if (!window.confirm(`Delete snapshot ${file}?`)) return;
      const response = await fetch('/api/annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'delete', file }),
      });
      const result = await response.json().catch(async () => ({ ok: false, error: await response.text() }));
      if (!response.ok || !result.ok) {
        document.getElementById('message').textContent = result.error || 'Could not delete snapshot';
        return;
      }
      if (selectedAnnotation?.file === file) returnToLiveCamera();
      document.getElementById('message').textContent = result.message || `Deleted ${file}.`;
      await refreshSaved();
    }

    function returnToLiveCamera() {
      selectedAnnotation = null;
      cvTuningLoaded = false;
      boardCornersLoaded = false;
      document.getElementById('message').textContent = 'Returned to live camera.';
      updateAutodetectButton();
      refreshState();
      refreshSaved();
    }

    document.getElementById('tool-black').addEventListener('click', () => setTool('black'));
    document.getElementById('tool-white').addEventListener('click', () => setTool('white'));
    document.getElementById('tool-erase').addEventListener('click', () => setTool('erase'));
    document.getElementById('clear').addEventListener('click', () => { stones.clear(); renderBoard(); });
    document.getElementById('toggle-autodetect').addEventListener('click', () => {
      if (selectedAnnotation) {
        returnToLiveCamera();
        return;
      }
      autodetectEnabled = !autodetectEnabled;
      if (!autodetectEnabled) oneShotOverlayUntil = 0;
      updateAutodetectButton();
      refreshState();
    });
    for (const [inputId] of Object.values(tuningControls)) {
      document.getElementById(inputId).addEventListener('input', scheduleLiveTuning);
    }
    for (const ids of cornerControls) {
      for (const inputId of ids) document.getElementById(inputId).addEventListener('input', scheduleLiveTuning);
    }
    document.getElementById('save-tuning').addEventListener('click', () => applyTuning({ persist: true }));
    document.getElementById('load-preset').addEventListener('click', () => {
      if (selectedAnnotation) returnToLiveCamera();
      const preset = selectedPreset();
      setStones(preset.stones);
      document.getElementById('label').value = preset.name;
      document.getElementById('message').textContent = `Loaded ${preset.name}.`;
    });
    document.getElementById('detect-once').addEventListener('click', () => {
      detectCurrent({ silent: false, force: true, showOverlay: true });
    });
    document.getElementById('save').addEventListener('click', saveAnnotation);
    document.getElementById('refresh-saved').addEventListener('click', refreshSaved);

    updateAutodetectButton();
    populatePresets();
    buildBoard();
    renderBoard();
    refreshSaved();
    setInterval(refreshState, cameraRefreshMs);
    setInterval(() => {
      if (autodetectEnabled) detectCurrent({ silent: true });
    }, detectionRefreshMs);
    refreshState();
    if (autodetectEnabled) detectCurrent({ silent: true });
  </script>
</body>
</html>
"""


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def make_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/annotate":
                self._send_bytes(ANNOTATION_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/evaluate":
                self._send_bytes(EVALUATOR_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send_bytes(json_bytes(app.state_json()), "application/json")
                return
            if parsed.path == "/api/annotations":
                self._send_bytes(json_bytes(app.list_annotations()), "application/json")
                return
            if parsed.path == "/api/recordings":
                self._send_bytes(json_bytes(app.list_recordings()), "application/json")
                return
            if parsed.path == "/api/model_rollouts":
                self._send_bytes(json_bytes(app.list_model_rollouts()), "application/json")
                return
            if parsed.path == "/api/evaluator":
                self._send_bytes(json_bytes(app.evaluator_json()), "application/json")
                return
            if parsed.path == "/api/synthetic_recordings":
                self._send_bytes(json_bytes(app.list_synthetic_recordings()), "application/json")
                return
            if parsed.path == "/api/recording":
                query = parse_qs(parsed.query)
                recording_id = str(query.get("id", [""])[0])
                try:
                    self._send_bytes(json_bytes(app.get_recording(recording_id)), "application/json")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Recording not found")
                return
            if parsed.path == "/api/model_rollout":
                query = parse_qs(parsed.query)
                rollout_id = str(query.get("id", [""])[0])
                try:
                    self._send_bytes(json_bytes(app.get_model_rollout(rollout_id)), "application/json")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Model rollout not found")
                return
            if parsed.path == "/api/synthetic_recording":
                query = parse_qs(parsed.query)
                recording_id = str(query.get("id", [""])[0])
                try:
                    self._send_bytes(json_bytes(app.get_synthetic_recording(recording_id)), "application/json")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Synthetic recording not found")
                return
            if parsed.path.startswith("/api/recording_frame/") and parsed.path.endswith(".jpg"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND, "Recording frame not found")
                    return
                _api, _frame_prefix, recording_id, camera_file = parts
                camera_name = Path(camera_file).stem
                query = parse_qs(parsed.query)
                frame_index = int(query.get("frame", ["0"])[0])
                try:
                    jpeg = app.recording_frame_jpeg(recording_id, camera_name, frame_index)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Recording frame not found")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path.startswith("/api/model_rollout_frame/") and parsed.path.endswith(".jpg"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND, "Model rollout frame not found")
                    return
                _api, _frame_prefix, rollout_id, camera_file = parts
                camera_name = Path(camera_file).stem
                query = parse_qs(parsed.query)
                frame_index = int(query.get("frame", ["0"])[0])
                try:
                    jpeg = app.model_rollout_frame_jpeg(rollout_id, camera_name, frame_index)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Model rollout frame not found")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path.startswith("/api/synthetic_recording_frame/") and parsed.path.endswith(".jpg"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND, "Synthetic recording frame not found")
                    return
                _api, _frame_prefix, recording_id, camera_file = parts
                camera_name = Path(camera_file).stem
                query = parse_qs(parsed.query)
                frame_index = int(query.get("frame", ["0"])[0])
                try:
                    jpeg = app.synthetic_recording_frame_jpeg(recording_id, camera_name, frame_index)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Synthetic recording frame not found")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path == "/api/annotation":
                query = parse_qs(parsed.query)
                filename = str(query.get("file", [""])[0])
                try:
                    self._send_bytes(json_bytes(app.get_annotation(filename)), "application/json")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Annotation not found")
                except ValueError as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if parsed.path == "/api/board_overlay.jpg":
                query = parse_qs(parsed.query)
                rotate = str(query.get("rotate", [""])[0])
                jpeg = app.board_overlay_jpeg(rotate=rotate)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Board overlay has no frame yet")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path == "/api/annotation_overlay.jpg":
                query = parse_qs(parsed.query)
                filename = str(query.get("file", [""])[0])
                rotate = str(query.get("rotate", [""])[0])
                corners_override = None
                if "corners" in query:
                    try:
                        corners_override = json.loads(str(query["corners"][0]))
                    except json.JSONDecodeError:
                        self.send_error(HTTPStatus.BAD_REQUEST, "Invalid corners JSON")
                        return
                curve_override = None
                if "overlay_fisheye_k" in query:
                    curve_override = float(query["overlay_fisheye_k"][0])
                try:
                    jpeg = app.annotation_overlay_jpeg(
                        filename=filename,
                        rotate=rotate,
                        corners_override=corners_override,
                        overlay_fisheye_k_override=curve_override,
                    )
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Annotation not found")
                    return
                except ValueError as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                if jpeg is None:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not render annotation overlay")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path.startswith("/api/model_preview/") and parsed.path.endswith(".jpg"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND, "Model preview not found")
                    return
                _api, _preview, run_id, camera_file = parts
                query = parse_qs(parsed.query)
                rotate = str(query.get("rotate", [""])[0])
                jpeg = app.model_preview_jpeg(run_id, Path(camera_file).stem, rotate=rotate)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Model preview has no frame yet")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            if parsed.path.startswith("/api/camera/") and parsed.path.endswith(".jpg"):
                camera_name = unquote(Path(parsed.path).name.removesuffix(".jpg"))
                camera = app.cameras.get(camera_name)
                if camera is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Camera not found")
                    return
                query = parse_qs(parsed.query)
                rotate = str(query.get("rotate", [""])[0])
                if rotate == "left":
                    frame = camera.frame()
                    if frame is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Camera has no frame yet")
                        return
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ok, encoded = cv2.imencode(".jpg", rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    jpeg = encoded.tobytes() if ok else None
                else:
                    jpeg = camera.jpeg()
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Camera has no frame yet")
                    return
                self._send_bytes(jpeg, "image/jpeg")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {
                "/api/teleop",
                "/api/follower_rest",
                "/api/rest_from_leader",
                "/api/refresh_devices",
                "/api/board",
                "/api/board_tuning",
                "/api/annotations",
                "/api/annotation_metadata",
                "/api/recording",
                "/api/recordings",
                "/api/model_run",
                "/api/model_rollouts",
                "/api/evaluator",
            }:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
                if parsed.path == "/api/teleop":
                    enabled = bool(payload.get("enabled", False))
                    result = app.set_teleop_enabled(enabled)
                elif parsed.path == "/api/follower_rest":
                    result = app.move_follower_to_rest()
                elif parsed.path == "/api/rest_from_leader":
                    result = app.set_rest_position_from_leader()
                elif parsed.path == "/api/refresh_devices":
                    result = app.refresh_devices()
                elif parsed.path == "/api/board":
                    action = payload.get("action")
                    if action == "capture":
                        result = app.capture_board(str(payload.get("slot", "")))
                    elif action == "delta":
                        result = app.compute_board_delta()
                    elif action == "detect":
                        result = app.detect_current_board()
                    else:
                        raise ValueError("Board action must be 'capture', 'delta', or 'detect'.")
                elif parsed.path == "/api/board_tuning":
                    result = app.update_board_tuning(payload)
                elif parsed.path == "/api/annotations":
                    action = payload.get("action")
                    if action == "delete":
                        result = app.delete_annotation(str(payload.get("file", "")))
                    elif action in {None, "", "save"}:
                        result = app.save_annotation(payload)
                    else:
                        raise ValueError("Annotations action must be 'save' or 'delete'.")
                elif parsed.path == "/api/annotation_metadata":
                    result = app.update_annotation_metadata(payload)
                elif parsed.path == "/api/recording":
                    action = payload.get("action")
                    if action == "start":
                        result = app.start_recording()
                    elif action == "stop":
                        result = app.stop_recording()
                    else:
                        raise ValueError("Recording action must be 'start' or 'stop'.")
                elif parsed.path == "/api/recordings":
                    action = payload.get("action")
                    if action == "delete_all":
                        result = app.delete_recordings()
                    elif action == "delete":
                        result = app.delete_recording(str(payload.get("id", "")))
                    else:
                        raise ValueError("Recordings action must be 'delete_all' or 'delete'.")
                elif parsed.path == "/api/model_rollouts":
                    action = payload.get("action")
                    if action == "delete":
                        result = app.delete_model_rollout(str(payload.get("id", "")))
                    else:
                        raise ValueError("Model rollout action must be 'delete'.")
                elif parsed.path == "/api/model_run":
                    action = payload.get("action")
                    if action == "start":
                        result = app.start_model_run(payload)
                    elif action == "stop":
                        result = app.stop_model_run()
                    else:
                        raise ValueError("Model action must be 'start' or 'stop'.")
                elif parsed.path == "/api/evaluator":
                    action = payload.get("action")
                    if action == "start":
                        result = app.start_evaluator(payload)
                    elif action == "stop":
                        result = app.stop_evaluator()
                    else:
                        raise ValueError("Evaluator action must be 'start' or 'stop'.")
                else:
                    result = app.save_annotation(payload)
            except Exception as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json")
                body = json_bytes({"ok": False, "error": str(exc)})
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self._send_bytes(json_bytes(result), "application/json")

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def choose_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        if port_is_free(host, port):
            return port
    raise RuntimeError(f"No free port found from {preferred} to {preferred + 49}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/go_board/dashboard_config.json"),
        help="JSON config to load. Defaults to examples/go_board/dashboard_config.json.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera_spec,
        default=[],
        help="Camera spec, e.g. overhead=0 or wrist=/dev/video2,1280x720@30. Repeat for two cameras.",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock camera feeds and mock robot telemetry.")
    parser.add_argument("--so101-port", help="Optional SO-101 follower serial port for live joint telemetry.")
    parser.add_argument("--leader-port", help="Optional SO-101 leader serial port for live teleoperation.")
    parser.add_argument("--robot-id", default="go_dashboard_follower")
    parser.add_argument("--calibrate", action="store_true", help="Allow SO-101 calibration on connect if needed.")
    parser.add_argument("--urdf-path", help="Optional URDF path for forward kinematics.")
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("examples/go_board/cv_snapshots"),
        help="Directory for annotated CV test snapshots.",
    )
    parser.add_argument(
        "--synthetic-recording-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_RECORDING_DIR,
        help="Directory containing generated synthetic recording folders.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_dashboard_config(args.config) if args.config.is_file() else DashboardConfig()

    use_mock = args.mock
    camera_specs = args.camera or config.cameras
    if use_mock and not camera_specs:
        camera_specs = [
            CameraSpec(name="overhead", index_or_path="mock", width=960, height=540, fps=30),
            CameraSpec(name="wrist", index_or_path="mock", width=960, height=540, fps=30),
        ]
    if not use_mock and not camera_specs:
        raise ValueError(
            f"No cameras configured. Add cameras to {args.config}, pass --camera, or use --mock explicitly."
        )

    if use_mock:
        colors = [(214, 185, 127), (180, 202, 204)]
        cameras: list[CameraStream] = [
            MockCameraStream(spec, colors[idx % len(colors)]) for idx, spec in enumerate(camera_specs[:2])
        ]
    else:
        cameras = [CameraStream(spec) for spec in camera_specs[:2]]

    robot_port = args.so101_port or config.robot.port
    robot_id = args.robot_id if args.robot_id != "go_dashboard_follower" else config.robot.id
    calibrate = bool(args.calibrate or config.robot.calibrate)
    configure_on_connect = bool(config.robot.configure_on_connect)
    urdf_path = args.urdf_path or config.robot.urdf_path
    target_frame_name = (
        args.target_frame_name
        if args.target_frame_name != "gripper_frame_link"
        else config.robot.target_frame_name
    )

    if robot_port and not args.mock:
        telemetry: TelemetrySource = SO101TelemetrySource(
            port=robot_port,
            robot_id=robot_id,
            calibrate=calibrate,
            configure_on_connect=configure_on_connect,
            leader_port=args.leader_port or config.leader.port,
            leader_id=config.leader.id,
            leader_calibrate=config.leader.calibrate,
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            rest_position=config.robot.rest_position,
        )
    else:
        if not args.mock:
            raise ValueError(
                f"No robot port configured. Add robot.port to {args.config}, pass --so101-port, "
                "or use --mock explicitly."
            )
        telemetry = MockTelemetrySource()

    app = DashboardApp(
        cameras=cameras,
        telemetry=telemetry,
        board=config.board,
        model=config.model,
        annotation_dir=args.annotation_dir,
        config_path=args.config,
        synthetic_recording_dir=args.synthetic_recording_dir,
    )
    app.start()

    host = args.host if args.host != "127.0.0.1" else config.host
    requested_port = args.port if args.port != 8765 else config.port
    port = choose_port(host, requested_port)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}"
    print(f"Dashboard running at {url}")

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, name="dashboard-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        server.serve_forever()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
