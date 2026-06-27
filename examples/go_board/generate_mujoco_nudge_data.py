#!/usr/bin/env python
"""Generate MuJoCo nudge-only Go-board correction recordings.

The generator focuses on board-local contact corrections: a target stone starts
near an intended Go coordinate, optional neighbor stones constrain the move, and
a simple candidate-search teacher chooses small pushes that improve target
alignment while penalizing neighbor disturbance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import cv2
import numpy as np

from mujoco_board_sim import (
    ArmSpec,
    BoardSpec,
    GO_COLUMNS,
    ROBOTSTUDIO_SO101_DIR,
    ROBOTSTUDIO_SO101_XML,
    _geom,
    _stone_body,
    intersection_xy,
    parse_go_coord,
)


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_OUTPUT_DIR = Path("outputs/go_board_mujoco_nudge_recordings")
ROBOTSTUDIO_BASE_POS = np.array([-0.335, 0.030, 0.012], dtype=np.float64)
ROBOTSTUDIO_BASE_QUAT_180_Z = "1 0 0 0"
ROBOTSTUDIO_BLUE_RGBA = "0.05 0.20 0.85 1"
WRIST_CAMERA_POS = "0 -0.050 -0.060"
WRIST_CAMERA_XYAXES = "-1 0 0 0 -0.766044 -0.642788"
WRIST_CAMERA_FLIPPED_XYAXES = "1 0 0 0 1 0"
WRIST_CAMERA_FOVY = "75"
NUDGE_TIP_RADIUS_M = 0.0045
NUDGE_TIP_POS = "-0.0079 -0.000218121 -0.0981274"
OVERHEAD_CAMERA_Z_M = 0.74
OVERHEAD_CAMERA_FOVY_DEG = 45.0
REAL_CENTER_TOUCH_JOINTS = {
    "shoulder_pan": 6.95,
    "shoulder_lift": -10.59,
    "elbow_flex": 56.66,
    "wrist_flex": 11.91,
    "wrist_roll": -9.27,
    "gripper": 1.52,
}
SIM_CENTER_TOUCH_QPOS = {
    "shoulder_pan": 0.096,
    "shoulder_lift": 0.279,
    "elbow_flex": 0.406,
    "wrist_flex": 0.232,
    "wrist_roll": 1.7,
    "gripper": -0.145,
}
SIM_CENTER_TOUCH_DEG = {
    name: math.degrees(value)
    for name, value in SIM_CENTER_TOUCH_QPOS.items()
    if name != "gripper"
}
REAL_TO_SIM_DEG_OFFSET = {
    name: SIM_CENTER_TOUCH_DEG[name] - REAL_CENTER_TOUCH_JOINTS[name]
    for name in SIM_CENTER_TOUCH_DEG
}


@dataclass(frozen=True)
class NudgeConfig:
    episodes: int = 20
    max_steps: int = 16
    seed: int = 7
    width: int = 640
    height: int = 480
    tolerance_m: float = 0.0035
    stable_success_frames: int = 2
    min_offset_m: float = 0.004
    max_offset_m: float = 0.018
    push_distance_m: float = 0.008
    min_push_distance_m: float = 0.0010
    push_distance_steps: int = 5
    candidate_count: int = 19
    approach_fan_deg: float = 120.0
    lateral_approach_weight: float = 0.35
    stop_short_margin_m: float = 0.004
    nudge_wrist_flex_deg: float = 80.0
    motion_slowdown: float = 3.0
    neighbor_penalty: float = 35.0
    action_penalty: float = 0.08
    fps: int = 10
    robot_arm: bool = False
    episode_prefix: str = "mujoco_nudge"
    stone_color: str | None = None
    no_neighbors: bool = False
    arm_start_jitter_deg: float = 12.0
    wrist_camera_pos: str = WRIST_CAMERA_POS
    wrist_camera_xyaxes: str = WRIST_CAMERA_XYAXES
    wrist_camera_fovy: str = WRIST_CAMERA_FOVY
    show_wrist_camera_marker: bool = False
    nudge_tip_radius_m: float = NUDGE_TIP_RADIUS_M


@dataclass(frozen=True)
class Scenario:
    episode_id: str
    target_coord: str
    target_row: int
    target_col: int
    target_color: str
    target_xy: tuple[float, float]
    initial_offset_xy: tuple[float, float]
    neighbor_stones: list[dict[str, Any]]


@dataclass(frozen=True)
class CandidatePush:
    direction_xy: tuple[float, float]
    distance_m: float


def go_coord(row: int, col: int, board_size: int = 19) -> str:
    columns = GO_COLUMNS if board_size == 19 else "".join(chr(ord("A") + i) for i in range(board_size))
    return f"{columns[col]}{row + 1}"


def _stone_name(index: int) -> str:
    return f"neighbor_{index:02d}"


def _validate_float_tuple(name: str, value: str, expected_count: int) -> str:
    parts = value.split()
    if len(parts) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} space-separated numbers, got {value!r}.")
    try:
        [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{name} must contain only numbers, got {value!r}.") from exc
    return " ".join(parts)


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def _format_float_tuple(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _pitch_camera_down_xyaxes(xyaxes: str, degrees: float) -> str:
    values = _float_tuple(_validate_float_tuple("wrist_camera_xyaxes", xyaxes, 6))
    x_axis = np.array(values[:3], dtype=np.float64)
    y_axis = np.array(values[3:], dtype=np.float64)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)

    radians = math.radians(degrees)
    pitched_y = math.cos(radians) * y_axis - math.sin(radians) * z_axis
    return _format_float_tuple(tuple(x_axis.tolist() + pitched_y.tolist()))


def _wrist_camera_pose_summary(pos: str, xyaxes: str) -> dict[str, tuple[float, float, float]]:
    x_axis = np.array(_float_tuple(_validate_float_tuple("wrist_camera_xyaxes", xyaxes, 6))[:3], dtype=np.float64)
    y_axis = np.array(_float_tuple(_validate_float_tuple("wrist_camera_xyaxes", xyaxes, 6))[3:], dtype=np.float64)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    facing = -z_axis
    pitch_down_deg = math.degrees(math.atan2(max(0.0, -float(facing[1])), max(1e-9, -float(facing[2]))))
    roll_flip_deg = 180.0 if float(y_axis[1]) >= 0 else 0.0
    return {
        "position_m": tuple(_float_tuple(_validate_float_tuple("wrist_camera_pos", pos, 3))),  # type: ignore[dict-item]
        "orientation_deg": (roll_flip_deg, pitch_down_deg, 0.0),
        "facing_unit": tuple(float(value) for value in facing),
    }


def _add_visual_geom(parent: ElementTree.Element, **attrs: str) -> None:
    attrs.setdefault("contype", "0")
    attrs.setdefault("conaffinity", "0")
    ElementTree.SubElement(parent, "geom", attrs)


def _add_camera_face(
    parent: ElementTree.Element,
    name: str,
    size: str,
    pos: str,
    rgba: str,
) -> None:
    _add_visual_geom(
        parent,
        name=name,
        type="box",
        size=size,
        pos=pos,
        rgba=rgba,
    )


def _add_axis_tripod(
    parent: ElementTree.Element,
    *,
    prefix: str,
    length: float,
    radius: float,
    alpha: float,
) -> None:
    _add_visual_geom(
        parent,
        name=f"{prefix}_x_axis",
        type="capsule",
        fromto=f"0 0 0 {length:.5f} 0 0",
        size=f"{radius:.5f}",
        rgba=f"1 0 0 {alpha:.3f}",
    )
    _add_visual_geom(
        parent,
        name=f"{prefix}_y_axis",
        type="capsule",
        fromto=f"0 0 0 0 {length:.5f} 0",
        size=f"{radius:.5f}",
        rgba=f"0 1 0 {alpha:.3f}",
    )
    _add_visual_geom(
        parent,
        name=f"{prefix}_z_axis",
        type="capsule",
        fromto=f"0 0 0 0 0 {length:.5f}",
        size=f"{radius:.5f}",
        rgba=f"0.2 0.45 1 {alpha:.3f}",
    )


def _add_wrist_camera_marker(gripper_body: ElementTree.Element, pos: str, xyaxes: str) -> None:
    wrist_axes_body = ElementTree.SubElement(
        gripper_body,
        "body",
        {
            "name": "wrist_camera_wrist_frame_axes",
            "pos": pos,
        },
    )
    _add_axis_tripod(
        wrist_axes_body,
        prefix="wrist_frame_at_camera",
        length=0.050,
        radius=0.0008,
        alpha=0.450,
    )

    marker_body = ElementTree.SubElement(
        gripper_body,
        "body",
        {
            "name": "wrist_camera_marker",
            "pos": pos,
            "xyaxes": xyaxes,
        },
    )
    half_x = 0.007
    half_y = 0.005
    half_z = 0.004
    face_t = 0.00045
    _add_camera_face(
        marker_body,
        "wrist_camera_view_face",
        f"{half_x:.5f} {half_y:.5f} {face_t:.5f}",
        f"0 0 {-half_z:.5f}",
        "1 0.9 0 0.95",
    )
    _add_camera_face(
        marker_body,
        "wrist_camera_back_face",
        f"{half_x:.5f} {half_y:.5f} {face_t:.5f}",
        f"0 0 {half_z:.5f}",
        "0.1 0.35 1 0.45",
    )
    _add_camera_face(
        marker_body,
        "wrist_camera_pos_x_face",
        f"{face_t:.5f} {half_y:.5f} {half_z:.5f}",
        f"{half_x:.5f} 0 0",
        "1 0.05 0.05 0.55",
    )
    _add_camera_face(
        marker_body,
        "wrist_camera_neg_x_face",
        f"{face_t:.5f} {half_y:.5f} {half_z:.5f}",
        f"{-half_x:.5f} 0 0",
        "0.55 0.02 0.02 0.45",
    )
    _add_camera_face(
        marker_body,
        "wrist_camera_pos_y_face",
        f"{half_x:.5f} {face_t:.5f} {half_z:.5f}",
        f"0 {half_y:.5f} 0",
        "0.05 1 0.05 0.55",
    )
    _add_camera_face(
        marker_body,
        "wrist_camera_neg_y_face",
        f"{half_x:.5f} {face_t:.5f} {half_z:.5f}",
        f"0 {-half_y:.5f} 0",
        "0.02 0.45 0.02 0.45",
    )
    _add_visual_geom(
        marker_body,
        name="wrist_camera_origin_dot",
        type="sphere",
        size="0.0022",
        rgba="1 1 1 1",
    )
    _add_axis_tripod(marker_body, prefix="camera_frame", length=0.032, radius=0.0013, alpha=1.0)


def _add_nudge_tip(gripper_body: ElementTree.Element, radius_m: float) -> None:
    for child in list(gripper_body):
        if child.get("name") in {"nudge_tip", "nudge_tip_collision", "nudge_tip_visual"}:
            gripper_body.remove(child)
    ElementTree.SubElement(
        gripper_body,
        "site",
        {
            "name": "nudge_tip",
            "pos": NUDGE_TIP_POS,
            "size": f"{radius_m:.5f}",
            "rgba": "0 1 1 0",
            "group": "3",
        },
    )
    ElementTree.SubElement(
        gripper_body,
        "geom",
        {
            "name": "nudge_tip_collision",
            "type": "sphere",
            "pos": NUDGE_TIP_POS,
            "size": f"{radius_m:.5f}",
            "rgba": "0 0.85 1 0",
            "group": "3",
            "mass": "0.002",
            "friction": "2.0 0.05 0.001",
            "solref": "0.002 1",
            "solimp": "0.95 0.99 0.001",
        },
    )


def robotstudio_so101_nudge_xml(
    destination: Path | None = None,
    wrist_camera_pos: str = WRIST_CAMERA_POS,
    wrist_camera_xyaxes: str = WRIST_CAMERA_XYAXES,
    wrist_camera_fovy: str = WRIST_CAMERA_FOVY,
    show_wrist_camera_marker: bool = False,
    nudge_tip_radius_m: float = NUDGE_TIP_RADIUS_M,
) -> Path:
    wrist_camera_pos = _validate_float_tuple("wrist_camera_pos", wrist_camera_pos, 3)
    wrist_camera_xyaxes = _validate_float_tuple("wrist_camera_xyaxes", wrist_camera_xyaxes, 6)
    wrist_camera_fovy = str(float(wrist_camera_fovy))
    output_path = destination or Path(tempfile.gettempdir()) / "go_board_robotstudio_so101_nudge" / "so101_nudge.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ElementTree.parse(ROBOTSTUDIO_SO101_XML)
    root = tree.getroot()
    asset_dir = ROBOTSTUDIO_SO101_DIR / "assets"
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str(asset_dir / Path(mesh_file).name))
    for compiler in root.findall("compiler"):
        compiler.attrib.pop("meshdir", None)
    for material in root.findall(".//material"):
        if material.get("rgba") == "1 0.82 0.12 1":
            material.set("rgba", ROBOTSTUDIO_BLUE_RGBA)

    base = root.find("./worldbody/body[@name='base']")
    if base is None:
        raise ValueError("RobotStudio SO-101 model is missing the base body.")
    base.set("pos", " ".join(f"{value:.6g}" for value in ROBOTSTUDIO_BASE_POS))
    base.set("quat", ROBOTSTUDIO_BASE_QUAT_180_Z)

    gripper_body = root.find(".//body[@name='gripper']")
    if gripper_body is None:
        raise ValueError("RobotStudio SO-101 model is missing the gripper body.")
    for camera in list(gripper_body.findall("camera")):
        if camera.get("name") == "wrist":
            gripper_body.remove(camera)
    ElementTree.SubElement(
        gripper_body,
        "camera",
        {
            "name": "wrist",
            "pos": wrist_camera_pos,
            "xyaxes": wrist_camera_xyaxes,
            "fovy": wrist_camera_fovy,
        },
    )
    for marker_body in list(gripper_body.findall("body")):
        if marker_body.get("name") in {"wrist_camera_marker", "wrist_camera_wrist_frame_axes"}:
            gripper_body.remove(marker_body)
    _add_nudge_tip(gripper_body, nudge_tip_radius_m)
    if show_wrist_camera_marker:
        _add_wrist_camera_marker(gripper_body, wrist_camera_pos, wrist_camera_xyaxes)

    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def build_nudge_scene_xml(
    scenario: Scenario,
    spec: BoardSpec,
    robotstudio_include_path: Path | None = None,
) -> str:
    half = spec.board_half_extent_m
    top_z = spec.board_top_z_m
    grid_z = top_z + 0.0008
    stone_z = top_z + spec.stone_half_height_m + 0.0004
    initial_xy = np.array(scenario.target_xy) + np.array(scenario.initial_offset_xy)
    bowl_y = half + 0.052
    bowl_z = 0.012

    grid_geoms: list[str] = []
    line_half = spec.grid_span_m / 2
    line_thickness = 0.00055
    for idx in range(spec.size):
        offset = (idx - (spec.size - 1) / 2) * spec.spacing_m
        grid_geoms.append(
            _geom(
                f"grid_col_{idx:02d}",
                "box",
                f"{line_thickness:.5f} {line_half:.5f} 0.00020",
                f"{offset:.5f} 0 {grid_z:.5f}",
                "0.12 0.08 0.035 1",
                contact=False,
            )
        )
        grid_geoms.append(
            _geom(
                f"grid_row_{idx:02d}",
                "box",
                f"{line_half:.5f} {line_thickness:.5f} 0.00020",
                f"0 {offset:.5f} {grid_z:.5f}",
                "0.12 0.08 0.035 1",
                contact=False,
            )
        )

    stone_bodies = [
        _stone_body("target_stone", scenario.target_color, (float(initial_xy[0]), float(initial_xy[1]), stone_z), spec)
    ]
    for idx, neighbor in enumerate(scenario.neighbor_stones):
        xy = neighbor["xy"]
        stone_bodies.append(_stone_body(_stone_name(idx), str(neighbor["color"]), (float(xy[0]), float(xy[1]), stone_z), spec))

    include_arm = robotstudio_include_path is not None
    include_xml = f'<include file="{escape(str(robotstudio_include_path))}"/>' if robotstudio_include_path else ""
    compiler_xml = "" if include_arm else '<compiler angle="radian"/>'
    pusher_xml = ""
    if not include_arm:
        pusher_xml = f"""
    <body name="pusher" mocap="true" pos="0 0 {top_z + 0.026:.5f}">
      <camera name="wrist" pos="0 -0.055 0.060" xyaxes="1 0 0 0 0.72 0.69"/>
      <geom name="pusher_geom" type="capsule" fromto="0 0 -0.018 0 0 0.018"
            size="0.0045" rgba="0.07 0.10 0.65 1" mass="0.02"
            friction="1.8 0.03 0.001"/>
    </body>
"""

    return f"""
<mujoco model="go_board_nudge_generator">
  {include_xml}
  {compiler_xml}
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.62" ambient="0.35 0.35 0.35"/>
    <rgba haze="0.85 0.88 0.92 1"/>
  </visual>
  <asset>
    <material name="board_mat" rgba="0.74 0.48 0.20 1"
              specular="0.18" shininess="0.35"/>
  </asset>
  <worldbody>
    <light name="overhead_light" pos="0 -0.25 0.8" diffuse="0.9 0.86 0.78"/>
    <camera name="overhead" pos="0 0 {OVERHEAD_CAMERA_Z_M:.5f}" xyaxes="1 0 0 0 1 0" fovy="{OVERHEAD_CAMERA_FOVY_DEG:.5f}"/>
    {pusher_xml}
    {_geom("table", "box", f"{half + 0.09:.5f} {half + 0.12:.5f} 0.01000", "0 0 -0.01000", "0.35 0.32 0.28 1", friction=spec.board_friction)}
    {_geom("board", "box", f"{half:.5f} {half:.5f} {spec.thickness_m / 2:.5f}", "0 0 0", "0.74 0.48 0.20 1", material="board_mat", friction=spec.board_friction)}
    {_geom("white_bowl", "cylinder", "0.04200 0.01200", f"-0.05800 {bowl_y:.5f} {bowl_z:.5f}", "0.82 0.82 0.78 0.45", friction=spec.board_friction)}
    {_geom("black_bowl", "cylinder", "0.04200 0.01200", f"0.05800 {bowl_y:.5f} {bowl_z:.5f}", "0.12 0.12 0.12 0.45", friction=spec.board_friction)}
    {" ".join(grid_geoms)}
    {" ".join(stone_bodies)}
  </worldbody>
</mujoco>
""".strip()


class NudgeMuJoCo:
    def __init__(
        self,
        scenario: Scenario,
        spec: BoardSpec,
        render_size: tuple[int, int],
        robot_arm: bool = False,
        wrist_camera_pos: str = WRIST_CAMERA_POS,
        wrist_camera_xyaxes: str = WRIST_CAMERA_XYAXES,
        wrist_camera_fovy: str = WRIST_CAMERA_FOVY,
        show_wrist_camera_marker: bool = False,
        nudge_tip_radius_m: float = NUDGE_TIP_RADIUS_M,
        motion_slowdown: float = NudgeConfig.motion_slowdown,
        nudge_wrist_flex_deg: float = NudgeConfig.nudge_wrist_flex_deg,
        stop_short_margin_m: float = NudgeConfig.stop_short_margin_m,
        min_push_distance_m: float = NudgeConfig.min_push_distance_m,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("MuJoCo is required. Try `uv sync --extra all` or install the mujoco extra.") from exc

        self.mujoco = mujoco
        self.scenario = scenario
        self.spec = spec
        self.arm_spec = ArmSpec()
        self.robot_arm = robot_arm
        self.motion_slowdown = motion_slowdown
        self.nudge_wrist_flex_deg = nudge_wrist_flex_deg
        self.stop_short_margin_m = stop_short_margin_m
        self.min_push_distance_m = min_push_distance_m
        self.arm_base_pos = ROBOTSTUDIO_BASE_POS.copy()
        robotstudio_include = (
            robotstudio_so101_nudge_xml(
                wrist_camera_pos=wrist_camera_pos,
                wrist_camera_xyaxes=wrist_camera_xyaxes,
                wrist_camera_fovy=wrist_camera_fovy,
                show_wrist_camera_marker=show_wrist_camera_marker,
                nudge_tip_radius_m=nudge_tip_radius_m,
            )
            if robot_arm
            else None
        )
        self.model = mujoco.MjModel.from_xml_string(
            build_nudge_scene_xml(scenario, spec, robotstudio_include_path=robotstudio_include)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size[1], width=render_size[0])
        self.pusher_z = spec.board_top_z_m + 0.026
        self.stone_z = spec.board_top_z_m + spec.stone_half_height_m + 0.0004
        self.nudge_tip_radius_m = nudge_tip_radius_m
        self.neighbor_names = [_stone_name(i) for i in range(len(scenario.neighbor_stones))]
        self.target_xy = np.array(scenario.target_xy, dtype=np.float64)
        if self.robot_arm:
            self.set_arm_joint_targets(self.arm_joint_targets_for_xy(np.array([0.0, -0.16]), z=self.pusher_z + 0.035))
        else:
            self._set_pusher(np.array([0.0, -0.18], dtype=np.float64))
        self.mujoco.mj_forward(self.model, self.data)
        if not self.robot_arm:
            self.step(80)

    def close(self) -> None:
        self.renderer.close()

    def step(self, n: int = 1, data: Any | None = None) -> None:
        self.mujoco.mj_step(self.model, data or self.data, nstep=n)

    def _set_pusher(self, xy: np.ndarray, data: Any | None = None) -> None:
        if self.robot_arm:
            return
        target = data or self.data
        target.mocap_pos[0] = np.array([float(xy[0]), float(xy[1]), self.pusher_z], dtype=np.float64)
        target.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _copy_state_to(self, target: Any) -> None:
        self._copy_data(self.data, target)
        self.mujoco.mj_forward(self.model, target)

    def _copy_data(self, source: Any, target: Any) -> None:
        target.qpos[:] = source.qpos
        target.qvel[:] = source.qvel
        target.act[:] = source.act
        target.ctrl[:] = source.ctrl
        target.mocap_pos[:] = source.mocap_pos
        target.mocap_quat[:] = source.mocap_quat
        self.mujoco.mj_forward(self.model, target)

    def stone_xy(self, data: Any | None = None) -> np.ndarray:
        target = data or self.data
        return target.body("target_stone").xpos[:2].copy()

    def set_target_stone_xy(self, data: Any, xy: np.ndarray) -> None:
        joint = data.joint("target_stone_free")
        joint.qpos[:3] = np.array([float(xy[0]), float(xy[1]), self.stone_z], dtype=np.float64)
        joint.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        joint.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, data)

    def neighbor_xy(self, data: Any | None = None) -> dict[str, list[float]]:
        target = data or self.data
        return {name: target.body(name).xpos[:2].copy().round(6).tolist() for name in self.neighbor_names}

    def target_error(self, data: Any | None = None) -> float:
        return float(np.linalg.norm(self.stone_xy(data) - self.target_xy))

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def arm_joint_targets_for_xy(
        self,
        xy: np.ndarray,
        z: float | None = None,
        gripper: float = 35.0,
    ) -> dict[str, float]:
        z = self.pusher_z if z is None else z
        base = self.arm_base_pos if self.robot_arm else np.array(self.arm_spec.base_pos, dtype=np.float64)
        shoulder_z = base[2] + 0.024 + 0.036
        dx = float(xy[0] - base[0])
        dy = float(xy[1] - base[1])
        pan = math.atan2(dy, dx)
        reach = math.hypot(dx, dy)
        height = float(z - shoulder_z)
        l1 = self.arm_spec.upper_arm_m
        l2 = self.arm_spec.forearm_m + self.arm_spec.wrist_m + self.arm_spec.gripper_m + self.arm_spec.finger_length_m
        dist = float(np.clip(math.hypot(reach, height), 0.025, l1 + l2 - 0.003))
        cos_elbow = np.clip((dist * dist - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
        elbow = -math.acos(float(cos_elbow))
        shoulder = math.atan2(height, reach) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
        wrist = -(shoulder + elbow)
        return {
            "shoulder_pan": math.degrees(pan),
            "shoulder_lift": math.degrees(shoulder),
            "elbow_flex": math.degrees(elbow),
            "wrist_flex": math.degrees(wrist),
            "wrist_roll": 0.0,
            "gripper": gripper,
        }

    def set_arm_joint_targets(self, joints: dict[str, float], data: Any | None = None) -> None:
        target = data or self.data
        for name in JOINT_NAMES[:-1]:
            actuator_name = name if self.robot_arm else f"{name}_act"
            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_value = self._joint_degrees_to_qpos(name, float(joints[name]))
            if actuator_id >= 0:
                target.ctrl[actuator_id] = qpos_value
            if joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                target.qpos[qpos_addr] = qpos_value
                dof_addr = self.model.jnt_dofadr[joint_id]
                target.qvel[dof_addr] = 0.0
        gripper_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_ACTUATOR,
            "gripper" if self.robot_arm else "gripper_act",
        )
        gripper_joint = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_JOINT,
            "gripper" if self.robot_arm else "gripper_left",
        )
        gripper_qpos = self._gripper_percent_to_qpos(float(joints["gripper"]))
        if gripper_id >= 0:
            target.ctrl[gripper_id] = gripper_qpos
        if gripper_joint >= 0:
            qpos_addr = self.model.jnt_qposadr[gripper_joint]
            dof_addr = self.model.jnt_dofadr[gripper_joint]
            target.qpos[qpos_addr] = gripper_qpos
            target.qvel[dof_addr] = 0.0
        self.mujoco.mj_forward(self.model, target)

    def _joint_degrees_to_qpos(self, name: str, value: float) -> float:
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        sim_value = value
        if self.robot_arm and name in REAL_TO_SIM_DEG_OFFSET:
            sim_value = value + REAL_TO_SIM_DEG_OFFSET[name]
        qpos = math.radians(sim_value)
        if actuator_id >= 0:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            qpos = float(np.clip(qpos, low, high))
        return qpos

    def _joint_qpos_to_real_degrees(self, name: str, qpos: float) -> float:
        degrees = math.degrees(float(qpos))
        if self.robot_arm and name in REAL_TO_SIM_DEG_OFFSET:
            degrees -= REAL_TO_SIM_DEG_OFFSET[name]
        return degrees

    def _gripper_percent_to_qpos(self, value: float) -> float:
        if self.robot_arm:
            if abs(value - REAL_CENTER_TOUCH_JOINTS["gripper"]) < 1e-6:
                return float(SIM_CENTER_TOUCH_QPOS["gripper"])
            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
            low, high = self.model.actuator_ctrlrange[actuator_id]
            return float(low + np.clip(value, 0.0, 100.0) / 100.0 * (high - low))
        return float(np.clip(value, 0.0, 100.0) / 100.0 * self.arm_spec.open_width_m)

    def current_arm_joint_targets(self, data: Any | None = None) -> dict[str, float]:
        target = data or self.data
        values: dict[str, float] = {}
        for name in JOINT_NAMES[:-1]:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                values[name] = 0.0
                continue
            values[name] = self._joint_qpos_to_real_degrees(name, float(target.qpos[self.model.jnt_qposadr[joint_id]]))
        gripper_joint = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_JOINT,
            "gripper" if self.robot_arm else "gripper_left",
        )
        if gripper_joint < 0:
            values["gripper"] = 100.0
        else:
            qpos = float(target.qpos[self.model.jnt_qposadr[gripper_joint]])
            if self.robot_arm:
                if abs(qpos - float(SIM_CENTER_TOUCH_QPOS["gripper"])) < 1e-4:
                    values["gripper"] = float(REAL_CENTER_TOUCH_JOINTS["gripper"])
                    return values
                actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
                low, high = self.model.actuator_ctrlrange[actuator_id]
                values["gripper"] = float(np.clip((qpos - low) / (high - low) * 100.0, 0.0, 100.0))
            else:
                values["gripper"] = float(np.clip(qpos / self.arm_spec.open_width_m * 100.0, 0.0, 100.0))
        return values

    def arm_joint_rows_for_xy(self, xy: np.ndarray, z: float | None = None, gripper: float = 35.0) -> list[dict[str, Any]]:
        return _joint_rows(self.arm_joint_targets_for_xy(xy, z=z, gripper=gripper))

    def current_arm_joint_rows(self) -> list[dict[str, Any]]:
        return _joint_rows(self.current_arm_joint_targets())

    def solve_site_ik(
        self,
        target_xyz: np.ndarray,
        data: Any | None = None,
        *,
        site_name: str = "nudge_tip",
        max_iterations: int = 80,
        tolerance_m: float = 0.0008,
    ) -> dict[str, Any]:
        if not self.robot_arm:
            raise ValueError("site IK requires robot_arm=True.")
        target = data or self.data
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"RobotStudio model is missing the {site_name} site.")
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        joint_ids = [self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError("RobotStudio model is missing one or more arm joints needed for IK.")
        wrist_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "wrist_flex")
        if wrist_id >= 0:
            wrist_qpos = math.radians(float(self.nudge_wrist_flex_deg))
            wrist_low, wrist_high = self.model.jnt_range[wrist_id]
            target.qpos[self.model.jnt_qposadr[wrist_id]] = float(np.clip(wrist_qpos, wrist_low, wrist_high))
            target.qvel[self.model.jnt_dofadr[wrist_id]] = 0.0
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
            joint_ids = [self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        dof_ids = [int(self.model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        qpos_ids = [int(self.model.jnt_qposadr[joint_id]) for joint_id in joint_ids]

        final_error = float("inf")
        iterations = 0
        for iterations in range(max_iterations):
            self.mujoco.mj_forward(self.model, target)
            current = target.site_xpos[site_id].copy()
            error = target_xyz - current
            final_error = float(np.linalg.norm(error))
            if final_error < tolerance_m:
                break
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            self.mujoco.mj_jacSite(self.model, target, jacp, jacr, site_id)
            jacobian = jacp[:, dof_ids]
            damping = 1e-3
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error * 0.55)
            for delta_value, joint_id, qpos_id in zip(delta, joint_ids, qpos_ids, strict=True):
                low, high = self.model.jnt_range[joint_id]
                target.qpos[qpos_id] = float(np.clip(target.qpos[qpos_id] + delta_value, low, high))

        self.mujoco.mj_forward(self.model, target)
        for name in JOINT_NAMES:
            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id >= 0 and joint_id >= 0:
                target.ctrl[actuator_id] = target.qpos[self.model.jnt_qposadr[joint_id]]
        return {
            "site_id": site_id,
            "target_xyz_m": target_xyz.round(6).tolist(),
            "actual_xyz_m": target.site_xpos[site_id].round(6).tolist(),
            "ik_error_m": final_error,
            "ik_iterations": iterations + 1,
        }

    def move_nudge_tip_to(
        self,
        data: Any,
        target_xyz: np.ndarray,
        steps: int = 16,
        after_step: Any | None = None,
    ) -> list[dict[str, Any]]:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "nudge_tip")
        self.mujoco.mj_forward(self.model, data)
        start_xyz = data.site_xpos[site_id].copy()
        trace: list[dict[str, Any]] = []
        for idx in range(max(1, steps)):
            alpha = (idx + 1) / max(1, steps)
            waypoint = start_xyz + (target_xyz - start_xyz) * alpha
            trial = self.mujoco.MjData(self.model)
            self._copy_data(data, trial)
            result = self.solve_site_ik(waypoint, trial, max_iterations=40, tolerance_m=0.0012)
            trace.append(result)
            for name in JOINT_NAMES:
                actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
                if actuator_id >= 0 and joint_id >= 0:
                    data.ctrl[actuator_id] = trial.qpos[self.model.jnt_qposadr[joint_id]]
            self.step(12, data)
            if after_step is not None:
                after_step()
        return trace

    def simulate_push(self, data: Any, push: CandidatePush, after_step: Any | None = None) -> None:
        direction = np.array(push.direction_xy, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        stone_xy = self.stone_xy(data)
        to_target = self.target_xy - stone_xy
        target_distance = float(np.linalg.norm(to_target))
        if target_distance > 1e-9:
            target_direction = to_target / target_distance
            forward_projection = max(0.0, float(np.dot(direction, target_direction)))
            if forward_projection > 1e-6:
                remaining_before_target = max(0.0, target_distance - float(self.stop_short_margin_m))
                distance = min(float(push.distance_m), remaining_before_target / forward_projection)
            else:
                distance = min(float(push.distance_m), float(self.min_push_distance_m))
        else:
            distance = 0.0
        start = stone_xy - direction * (self.spec.stone_radius_m + self.nudge_tip_radius_m + 0.002)
        end = stone_xy + direction * distance
        slow = max(1, int(round(self.motion_slowdown)))
        if self.robot_arm:
            approach = np.array([start[0], start[1], self.stone_z + 0.030], dtype=np.float64)
            pre_touch = np.array([start[0], start[1], self.stone_z], dtype=np.float64)
            push_end = np.array([end[0], end[1], self.stone_z], dtype=np.float64)
            retreat_xy = end + direction * 0.010
            retreat = np.array([retreat_xy[0], retreat_xy[1], self.stone_z + 0.030], dtype=np.float64)
            self.move_nudge_tip_to(data, approach, steps=14 * slow, after_step=after_step)
            self.move_nudge_tip_to(data, pre_touch, steps=10 * slow, after_step=after_step)
            self.move_nudge_tip_to(data, push_end, steps=24 * slow, after_step=after_step)
            self.move_nudge_tip_to(data, retreat, steps=12 * slow, after_step=after_step)
            self.step(40 * slow, data)
            if after_step is not None:
                after_step()
        else:
            self._move_pusher(data, start, steps=24 * slow)
            self._move_pusher(data, end, steps=24 * slow)
            self._move_target_stone(data, stone_xy + direction * distance, steps=18 * slow)
            self._move_pusher(data, end + direction * 0.010, steps=14 * slow)
            self.step(20 * slow, data)

    def execute_push(self, push: CandidatePush) -> None:
        self.simulate_push(self.data, push)

    def _move_pusher(self, data: Any, target_xy: np.ndarray, steps: int) -> None:
        start = data.mocap_pos[0, :2].copy()
        for idx in range(max(1, steps)):
            alpha = (idx + 1) / max(1, steps)
            xy = start + (target_xy - start) * alpha
            self._set_pusher(xy, data)
            self.step(1, data)

    def _move_arm_to_xy(self, data: Any, target_xy: np.ndarray, steps: int, z: float | None = None) -> None:
        start = self.current_arm_joint_targets(data)
        target = self.arm_joint_targets_for_xy(target_xy, z=z)
        for idx in range(max(1, steps)):
            alpha = (idx + 1) / max(1, steps)
            joints = {name: start[name] + (target[name] - start[name]) * alpha for name in JOINT_NAMES}
            self.set_arm_joint_targets(joints, data)
            self.mujoco.mj_forward(self.model, data)

    def _move_target_stone(self, data: Any, target_xy: np.ndarray, steps: int) -> None:
        start = self.stone_xy(data)
        for idx in range(max(1, steps)):
            alpha = (idx + 1) / max(1, steps)
            xy = start + (target_xy - start) * alpha
            self.set_target_stone_xy(data, xy)
            if self.robot_arm:
                self.mujoco.mj_forward(self.model, data)
            else:
                self.step(1, data)

    def score_push(self, push: CandidatePush, initial_neighbor_xy: dict[str, np.ndarray], cfg: NudgeConfig) -> float:
        trial = self.mujoco.MjData(self.model)
        self._copy_state_to(trial)
        self.simulate_push(trial, push)
        distance = self.target_error(trial)
        neighbor_motion = 0.0
        overlap = 0.0
        target_xy = self.stone_xy(trial)
        for name, initial in initial_neighbor_xy.items():
            current = trial.body(name).xpos[:2]
            neighbor_motion += float(np.linalg.norm(current - initial))
            overlap += max(0.0, self.spec.stone_radius_m * 2.05 - float(np.linalg.norm(target_xy - current)))
        action_cost = cfg.action_penalty * float(push.distance_m)
        return distance + cfg.neighbor_penalty * neighbor_motion + 60.0 * overlap + action_cost


def _scenario(rng: random.Random, episode_index: int, spec: BoardSpec, target_coords: list[str], cfg: NudgeConfig) -> Scenario:
    target_coord = rng.choice(target_coords)
    row, col, coord = parse_go_coord(target_coord, spec.size)
    target_xy = np.array(intersection_xy(row, col, spec), dtype=np.float64)
    angle = rng.uniform(0, 2 * math.pi)
    radius = rng.uniform(cfg.min_offset_m, cfg.max_offset_m)
    offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * radius
    color = cfg.stone_color or rng.choice(["black", "white"])

    neighbor_options = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    rng.shuffle(neighbor_options)
    neighbor_count = 0 if cfg.no_neighbors else rng.choices([0, 1, 2, 3, 4], weights=[1.0, 2.0, 2.0, 1.0, 0.6])[0]
    neighbors: list[dict[str, Any]] = []
    for dr, dc in neighbor_options:
        if len(neighbors) >= neighbor_count:
            break
        rr, cc = row + dr, col + dc
        if rr < 0 or rr >= spec.size or cc < 0 or cc >= spec.size:
            continue
        xy = intersection_xy(rr, cc, spec)
        neighbors.append(
            {
                "coord": go_coord(rr, cc, spec.size),
                "row": rr,
                "col": cc,
                "color": rng.choice(["black", "white"]),
                "xy": [float(xy[0]), float(xy[1])],
            }
        )

    return Scenario(
        episode_id=f"{cfg.episode_prefix}_{episode_index:05d}_{color}_to_{coord.lower()}",
        target_coord=coord,
        target_row=row,
        target_col=col,
        target_color=color,
        target_xy=(float(target_xy[0]), float(target_xy[1])),
        initial_offset_xy=(float(offset[0]), float(offset[1])),
        neighbor_stones=neighbors,
    )


def _ik_calibration_scenario(spec: BoardSpec) -> Scenario:
    row, col, coord = parse_go_coord("K10", spec.size)
    target_xy = np.array(intersection_xy(row, col, spec), dtype=np.float64)
    offset = np.array([0.0, 0.0], dtype=np.float64)
    return Scenario(
        episode_id="center_touch_calibration_white_to_k10",
        target_coord=coord,
        target_row=row,
        target_col=col,
        target_color="white",
        target_xy=(float(target_xy[0]), float(target_xy[1])),
        initial_offset_xy=(float(offset[0]), float(offset[1])),
        neighbor_stones=[],
    )


def candidate_pushes(sim: NudgeMuJoCo, cfg: NudgeConfig) -> list[CandidatePush]:
    stone_xy = sim.stone_xy()
    to_target = sim.target_xy - stone_xy
    target_distance = float(np.linalg.norm(to_target))
    base_direction = to_target / target_distance if target_distance > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
    remaining_before_target = max(0.0, target_distance - float(cfg.stop_short_margin_m))
    neighbor_vectors = []
    for xy in sim.neighbor_xy().values():
        offset = stone_xy - np.array(xy, dtype=np.float64)
        distance = float(np.linalg.norm(offset))
        if distance > 1e-9:
            neighbor_vectors.append(offset / distance / max(distance, sim.spec.stone_radius_m))
    if neighbor_vectors:
        clearance_bias = np.sum(neighbor_vectors, axis=0)
        base_direction = base_direction + cfg.lateral_approach_weight * clearance_bias
        base_direction /= max(float(np.linalg.norm(base_direction)), 1e-9)
    base_angle = math.atan2(float(base_direction[1]), float(base_direction[0]))
    fan = math.radians(max(0.0, float(cfg.approach_fan_deg)))
    pushes: list[CandidatePush] = []
    count = max(1, int(cfg.candidate_count))
    for idx in range(count):
        fraction = 0.0 if count == 1 else (idx / (count - 1)) - 0.5
        angle = base_angle + fraction * fan
        direction_array = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        forward_projection = float(np.dot(direction_array, base_direction))
        if forward_projection <= 0.05:
            continue
        max_distance = min(
            float(cfg.push_distance_m),
            max(0.0, remaining_before_target / forward_projection),
            max(float(cfg.min_push_distance_m), target_distance * 0.45),
        )
        if max_distance <= 1e-6:
            continue
        min_distance = min(float(cfg.min_push_distance_m), max_distance)
        distances = np.linspace(min_distance, max_distance, max(1, int(cfg.push_distance_steps)))
        direction = (float(direction_array[0]), float(direction_array[1]))
        for distance in distances:
            pushes.append(CandidatePush(direction_xy=direction, distance_m=float(distance)))
    if not pushes and target_distance > 1e-9:
        fallback_distance = min(float(cfg.min_push_distance_m), max(0.0, remaining_before_target))
        if fallback_distance > 1e-6:
            pushes.append(
                CandidatePush(direction_xy=(float(base_direction[0]), float(base_direction[1])), distance_m=fallback_distance)
            )
    return pushes


def _joint_rows(values: dict[str, float] | None = None, gripper: float = 100.0) -> list[dict[str, Any]]:
    values = values or {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": gripper,
    }
    return [
        {
            "name": name,
            "value": float(values[name]),
            "unit": "%" if name == "gripper" else "deg",
            "min_value": 0.0 if name == "gripper" else -180.0,
            "max_value": 100.0 if name == "gripper" else 180.0,
        }
        for name, value in values.items()
    ]


def _write_rgb_jpeg(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _overhead_target_pixel(
    image_shape: tuple[int, int, int],
    target_xy: tuple[float, float],
    spec: BoardSpec,
) -> tuple[int, int, int]:
    height, width = image_shape[:2]
    camera_to_board_m = OVERHEAD_CAMERA_Z_M - spec.board_top_z_m
    span_y_m = 2.0 * camera_to_board_m * math.tan(math.radians(OVERHEAD_CAMERA_FOVY_DEG) / 2.0)
    span_x_m = span_y_m * (width / height)
    x, y = target_xy
    px = int(round(width / 2.0 + (x / span_x_m) * width))
    py = int(round(height / 2.0 - (y / span_y_m) * height))
    stone_radius_px = int(round(spec.stone_radius_m / span_x_m * width))
    radius = int(max(5, min(10, round(stone_radius_px * 0.55))))
    return px, py, radius


def _draw_overhead_target_overlay(image_rgb: np.ndarray, scenario: Scenario, spec: BoardSpec) -> np.ndarray:
    output = image_rgb.copy()
    px, py, radius = _overhead_target_pixel(output.shape, scenario.target_xy, spec)
    center = (px, py)
    cv2.circle(output, center, radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(output, center, radius, (245, 20, 20), -1, cv2.LINE_AA)
    return output


def _randomize_arm_start(sim: NudgeMuJoCo, rng: random.Random, cfg: NudgeConfig) -> dict[str, float]:
    if not sim.robot_arm:
        return {}
    jitter = float(cfg.arm_start_jitter_deg)
    joints = {
        "shoulder_pan": rng.uniform(-jitter, jitter),
        "shoulder_lift": rng.uniform(-jitter * 0.5, jitter),
        "elbow_flex": rng.uniform(-jitter, jitter),
        "wrist_flex": rng.uniform(-jitter, jitter),
        "wrist_roll": rng.uniform(-jitter * 0.4, jitter * 0.4),
        "gripper": rng.uniform(28.0, 45.0),
    }
    sim.set_arm_joint_targets(joints)
    sim.step(40)
    return {name: round(value, 3) for name, value in joints.items()}


def generate_episode(output_root: Path, scenario: Scenario, cfg: NudgeConfig, spec: BoardSpec) -> dict[str, Any]:
    episode_dir = output_root / scenario.episode_id
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    (episode_dir / "frames" / "overhead").mkdir(parents=True, exist_ok=True)
    (episode_dir / "frames" / "wrist").mkdir(parents=True, exist_ok=True)
    (episode_dir / "overhead_processed").mkdir(parents=True, exist_ok=True)

    sim = NudgeMuJoCo(
        scenario,
        spec,
        (cfg.width, cfg.height),
        robot_arm=cfg.robot_arm,
        wrist_camera_pos=cfg.wrist_camera_pos,
        wrist_camera_xyaxes=cfg.wrist_camera_xyaxes,
        wrist_camera_fovy=cfg.wrist_camera_fovy,
        show_wrist_camera_marker=cfg.show_wrist_camera_marker,
        nudge_tip_radius_m=cfg.nudge_tip_radius_m,
        motion_slowdown=cfg.motion_slowdown,
        nudge_wrist_flex_deg=cfg.nudge_wrist_flex_deg,
        stop_short_margin_m=cfg.stop_short_margin_m,
        min_push_distance_m=cfg.min_push_distance_m,
    )
    rng = random.Random(f"{cfg.seed}:{scenario.episode_id}:arm_start")
    arm_start_joints = _randomize_arm_start(sim, rng, cfg)
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    sample_index = 0
    stable = 0
    initial_neighbor_xy = {name: np.array(xy, dtype=np.float64) for name, xy in sim.neighbor_xy().items()}

    def record_sample(action: dict[str, Any], action_joints: list[dict[str, Any]], control_step: int) -> None:
        nonlocal sample_index
        error = sim.target_error()
        done = error <= cfg.tolerance_m
        overhead_path = Path("frames") / "overhead" / f"{sample_index:06d}.jpg"
        wrist_path = Path("frames") / "wrist" / f"{sample_index:06d}.jpg"
        overhead_rgb = sim.render("overhead")
        _write_rgb_jpeg(episode_dir / overhead_path, overhead_rgb)
        _write_rgb_jpeg(
            episode_dir / "overhead_processed" / f"{sample_index:06d}.jpg",
            _draw_overhead_target_overlay(overhead_rgb, scenario, spec),
        )
        _write_rgb_jpeg(episode_dir / wrist_path, sim.render("wrist"))
        timestamp = started_at + sample_index / cfg.fps
        rows.append(
            {
                "index": sample_index,
                "timestamp": timestamp,
                "elapsed_s": sample_index / cfg.fps,
                "cameras": {"overhead": str(overhead_path), "wrist": str(wrist_path)},
                "telemetry": {
                    "timestamp": timestamp,
                    "connected": True,
                    "mode": "mujoco_contact_nudge_robot_arm" if cfg.robot_arm else "mujoco_nudge",
                    "fps": cfg.fps,
                    "joints": sim.current_arm_joint_rows() if cfg.robot_arm else _joint_rows(),
                    "leader_joints": action_joints,
                    "teleop_enabled": True,
                    "synthetic": {
                        "target_xy_m": list(scenario.target_xy),
                        "stone_xy_m": sim.stone_xy().round(6).tolist(),
                        "stone_offset_xy_m": (sim.stone_xy() - sim.target_xy).round(6).tolist(),
                        "neighbor_xy_m": sim.neighbor_xy(),
                        "target_error_m": error,
                        "done": done,
                        "stable_done_count": stable,
                        "control_step": control_step,
                        "action": action,
                    },
                    "note": "MuJoCo synthetic contact nudge sample.",
                },
            }
        )
        sample_index += 1

    try:
        for step_index in range(cfg.max_steps):
            error_before = sim.target_error()
            done = error_before <= cfg.tolerance_m
            stable = stable + 1 if done else 0
            pushes = candidate_pushes(sim, cfg)
            if not pushes:
                record_sample(
                    {"push_dx_m": 0.0, "push_dy_m": 0.0, "push_distance_m": 0.0, "stopped_before_target": True},
                    sim.current_arm_joint_rows() if cfg.robot_arm else _joint_rows(),
                    step_index,
                )
                break
            scored = [(sim.score_push(push, initial_neighbor_xy, cfg), push) for push in pushes]
            scored.sort(key=lambda item: item[0])
            best = scored[0][1]
            action = {
                "push_dx_m": best.direction_xy[0] * best.distance_m,
                "push_dy_m": best.direction_xy[1] * best.distance_m,
                "push_distance_m": best.distance_m,
            }
            action_xy = sim.stone_xy() + np.array(
                [action["push_dx_m"], action["push_dy_m"]],
                dtype=np.float64,
            )
            state_joints = sim.current_arm_joint_rows() if cfg.robot_arm else _joint_rows()
            if cfg.robot_arm:
                action_xyz = np.array([action_xy[0], action_xy[1], sim.stone_z], dtype=np.float64)
                trial = sim.mujoco.MjData(sim.model)
                sim._copy_state_to(trial)
                sim.solve_site_ik(action_xyz, trial, max_iterations=50, tolerance_m=0.0015)
                action_joints = _joint_rows(sim.current_arm_joint_targets(trial))
            else:
                action_joints = _joint_rows()

            record_sample(action, action_joints, step_index)

            if stable >= cfg.stable_success_frames:
                break
            sim.simulate_push(sim.data, best, after_step=lambda: record_sample(action, action_joints, step_index))
    finally:
        final_error = sim.target_error()
        final_stone_xy = sim.stone_xy().round(6).tolist()
        sim.close()

    with (episode_dir / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry_file:
        for row in rows:
            telemetry_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    metadata = {
        "schema_version": 1,
        "run_type": "mujoco_synthetic_nudge",
        "id": scenario.episode_id,
        "status": "complete",
        "sample_hz": cfg.fps,
        "samples": len(rows),
        "board": {
            "size": spec.size,
            "target": {
                "coord": scenario.target_coord,
                "row": scenario.target_row,
                "col": scenario.target_col,
                "color": scenario.target_color,
            },
            "delta": {
                "added": [
                    {
                        "coord": scenario.target_coord,
                        "row": scenario.target_row,
                        "col": scenario.target_col,
                        "color": scenario.target_color,
                    }
                ],
                "removed": [],
                "changed": [],
            },
            "target_xy_m": list(scenario.target_xy),
            "initial_offset_xy_m": list(scenario.initial_offset_xy),
            "neighbors": scenario.neighbor_stones,
        },
        "synthetic": {
            "generator": Path(__file__).name,
            "action_space": "so101_joint_targets" if cfg.robot_arm else "board_local_push_delta_m",
            "observation_space": (
                "rendered_overhead_wrist_plus_so101_joint_state"
                if cfg.robot_arm
                else "rendered_overhead_wrist_plus_board_local_state"
            ),
            "robot_arm": cfg.robot_arm,
            "tolerance_m": cfg.tolerance_m,
            "final_error_m": final_error,
            "final_stone_xy_m": final_stone_xy,
            "success": final_error <= cfg.tolerance_m,
            "arm_start_joints": arm_start_joints,
            "nudge_tip_radius_m": cfg.nudge_tip_radius_m,
            "stop_short_margin_m": cfg.stop_short_margin_m,
            "nudge_wrist_flex_deg": cfg.nudge_wrist_flex_deg,
            "motion_slowdown": cfg.motion_slowdown,
            "wrist_camera": {
                "pos": cfg.wrist_camera_pos,
                "xyaxes": cfg.wrist_camera_xyaxes,
                "fovy": cfg.wrist_camera_fovy,
                "marker_visible": cfg.show_wrist_camera_marker,
            },
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
                "source": "synthetic_overhead_only_overlay",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"episode": scenario.episode_id, "frames": len(rows), **metadata["synthetic"]}


def default_targets() -> list[str]:
    return ["D4", "D10", "D16", "K4", "K10", "K16", "Q4", "Q10", "Q16"]


def generate_dataset(output_root: Path, cfg: NudgeConfig, targets: list[str], force: bool = False) -> dict[str, Any]:
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec = BoardSpec()
    rng = random.Random(cfg.seed)
    episodes = []
    for index in range(cfg.episodes):
        scenario = _scenario(rng, index, spec, targets, cfg)
        episodes.append(generate_episode(output_root, scenario, cfg, spec))
    summary = {
        "schema_version": 1,
        "output_root": str(output_root),
        "config": asdict(cfg),
        "targets": targets,
        "episodes": len(episodes),
        "successes": sum(1 for episode in episodes if episode["success"]),
        "frames": sum(int(episode["frames"]) for episode in episodes),
        "episode_summaries": episodes,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_joint_values(value: str) -> dict[str, float]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--viewer-real-joints must be a JSON object.") from exc
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("--viewer-real-joints must be a JSON object.")
    missing = [name for name in JOINT_NAMES if name not in raw]
    if missing:
        raise argparse.ArgumentTypeError(f"--viewer-real-joints is missing: {', '.join(missing)}.")
    try:
        return {name: float(raw[name]) for name in JOINT_NAMES}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--viewer-real-joints values must be numbers.") from exc


def _place_arm_pre_touch(sim: NudgeMuJoCo, real_joints: dict[str, float] | None = None) -> dict[str, Any]:
    if not sim.robot_arm:
        raise ValueError("IK calibration requires --robot-arm.")
    source_joints = real_joints or REAL_CENTER_TOUCH_JOINTS
    stone_xy = sim.stone_xy()
    sim.set_arm_joint_targets(source_joints)
    sim.mujoco.mj_forward(sim.model, sim.data)
    site_id = sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_SITE, "nudge_tip")
    actual_xyz = sim.data.site_xpos[site_id].copy()
    target_xyz = np.array([sim.target_xy[0], sim.target_xy[1], sim.stone_z], dtype=np.float64)
    joints = sim.current_arm_joint_targets()
    return {
        "stone_xy_m": stone_xy.round(6).tolist(),
        "target_xy_m": sim.target_xy.round(6).tolist(),
        "pre_touch_xy_m": sim.target_xy.round(6).tolist(),
        "nudge_tip_xyz_m": actual_xyz.round(6).tolist(),
        "target_nudge_tip_xyz_m": target_xyz.round(6).tolist(),
        "nudge_tip_radius_m": sim.nudge_tip_radius_m,
        "gap_m": 0.0,
        "ik_error_m": float(np.linalg.norm(actual_xyz - target_xyz)),
        "ik_iterations": 0,
        "source_joints": source_joints,
        "joints": {name: round(value, 3) for name, value in joints.items()},
    }


def run_viewer(
    cfg: NudgeConfig,
    targets: list[str],
    episode_index: int,
    ik_calibration: bool = False,
    real_joints: dict[str, float] | None = None,
) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise ImportError(
            "The MuJoCo viewer is unavailable. Try `uv run --with mujoco python "
            "examples/go_board/generate_mujoco_nudge_data.py --viewer`."
        ) from exc

    spec = BoardSpec()
    if ik_calibration:
        scenario = _ik_calibration_scenario(spec)
    else:
        rng = random.Random(cfg.seed)
        scenario = _scenario(rng, 0, spec, targets, cfg)
        for index in range(1, max(0, episode_index) + 1):
            scenario = _scenario(rng, index, spec, targets, cfg)

    sim = NudgeMuJoCo(
        scenario,
        spec,
        (cfg.width, cfg.height),
        robot_arm=cfg.robot_arm,
        wrist_camera_pos=cfg.wrist_camera_pos,
        wrist_camera_xyaxes=cfg.wrist_camera_xyaxes,
        wrist_camera_fovy=cfg.wrist_camera_fovy,
        show_wrist_camera_marker=cfg.show_wrist_camera_marker,
        nudge_tip_radius_m=cfg.nudge_tip_radius_m,
        motion_slowdown=cfg.motion_slowdown,
        nudge_wrist_flex_deg=cfg.nudge_wrist_flex_deg,
        stop_short_margin_m=cfg.stop_short_margin_m,
        min_push_distance_m=cfg.min_push_distance_m,
    )
    pre_touch: dict[str, Any] | None = None
    if ik_calibration:
        pre_touch = _place_arm_pre_touch(sim, real_joints=real_joints)
    print(
        f"Viewer nudge demo: {scenario.episode_id}, target={scenario.target_color} "
        f"{scenario.target_coord}, neighbors={len(scenario.neighbor_stones)}, robot_arm={cfg.robot_arm}."
    )
    if pre_touch is not None:
        print(f"IK calibration pre-touch: {json.dumps(pre_touch, sort_keys=True)}")
    if cfg.robot_arm:
        pose = _wrist_camera_pose_summary(cfg.wrist_camera_pos, cfg.wrist_camera_xyaxes)
        print(
            "Wrist camera: "
            f"pos={cfg.wrist_camera_pos!r}, xyaxes={cfg.wrist_camera_xyaxes!r}, fovy={cfg.wrist_camera_fovy}."
        )
        print(
            "Wrist camera offset: "
            f"x={pose['position_m'][0]:.4f}m, y={pose['position_m'][1]:.4f}m, z={pose['position_m'][2]:.4f}m, "
            f"roll={pose['orientation_deg'][0]:.1f}deg, pitch_down={pose['orientation_deg'][1]:.1f}deg, "
            f"yaw={pose['orientation_deg'][2]:.1f}deg; facing={pose['facing_unit']}."
        )
    print("Press Space/Play in MuJoCo to step physics, or close the viewer window when done.")
    try:
        mujoco.viewer.launch(sim.model, sim.data)
    finally:
        sim.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=NudgeConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=NudgeConfig.max_steps)
    parser.add_argument("--seed", type=int, default=NudgeConfig.seed)
    parser.add_argument("--width", type=int, default=NudgeConfig.width)
    parser.add_argument("--height", type=int, default=NudgeConfig.height)
    parser.add_argument("--tolerance-m", type=float, default=NudgeConfig.tolerance_m)
    parser.add_argument("--min-offset-m", type=float, default=NudgeConfig.min_offset_m)
    parser.add_argument("--max-offset-m", type=float, default=NudgeConfig.max_offset_m)
    parser.add_argument("--candidate-count", type=int, default=NudgeConfig.candidate_count)
    parser.add_argument("--push-distance-m", type=float, default=NudgeConfig.push_distance_m)
    parser.add_argument("--min-push-distance-m", type=float, default=NudgeConfig.min_push_distance_m)
    parser.add_argument("--push-distance-steps", type=int, default=NudgeConfig.push_distance_steps)
    parser.add_argument("--approach-fan-deg", type=float, default=NudgeConfig.approach_fan_deg)
    parser.add_argument("--lateral-approach-weight", type=float, default=NudgeConfig.lateral_approach_weight)
    parser.add_argument("--stop-short-margin-m", type=float, default=NudgeConfig.stop_short_margin_m)
    parser.add_argument("--nudge-wrist-flex-deg", type=float, default=NudgeConfig.nudge_wrist_flex_deg)
    parser.add_argument("--motion-slowdown", type=float, default=NudgeConfig.motion_slowdown)
    parser.add_argument("--fps", type=int, default=NudgeConfig.fps)
    parser.add_argument("--episode-prefix", default=NudgeConfig.episode_prefix)
    parser.add_argument("--stone-color", choices=["black", "white"], default=NudgeConfig.stone_color)
    parser.add_argument("--no-neighbors", action="store_true", help="Generate exactly one target stone and no neighbor stones.")
    parser.add_argument("--arm-start-jitter-deg", type=float, default=NudgeConfig.arm_start_jitter_deg)
    parser.add_argument("--nudge-tip-radius-m", type=float, default=NudgeConfig.nudge_tip_radius_m)
    parser.add_argument("--target", action="append", help="Target coordinate to sample. Repeat to provide a set.")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists.")
    parser.add_argument(
        "--robot-arm",
        action="store_true",
        help="Include the RobotStudio SO-101 arm and record simulated joint state/action telemetry.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer for one nudge scenario.")
    parser.add_argument("--viewer-episode-index", type=int, default=0, help="Seeded scenario index to open in the viewer.")
    parser.add_argument(
        "--viewer-ik-calibration",
        action="store_true",
        help="Open a simple one-white-stone scene with the arm posed just before contact for IK calibration.",
    )
    parser.add_argument(
        "--viewer-real-joints",
        type=_parse_joint_values,
        help=(
            "JSON object of real/dashboard joint values to apply in viewer calibration mode. "
            "Must include shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, and gripper."
        ),
    )
    parser.add_argument(
        "--wrist-camera-pos",
        default=WRIST_CAMERA_POS,
        help=f"Space-separated wrist camera position in the gripper frame. Default: {WRIST_CAMERA_POS!r}.",
    )
    parser.add_argument(
        "--wrist-camera-xyaxes",
        default=WRIST_CAMERA_XYAXES,
        help=f"Space-separated MuJoCo camera xyaxes in the gripper frame. Default: {WRIST_CAMERA_XYAXES!r}.",
    )
    parser.add_argument(
        "--wrist-camera-fovy",
        default=WRIST_CAMERA_FOVY,
        help=f"Wrist camera vertical field of view in degrees. Default: {WRIST_CAMERA_FOVY}.",
    )
    parser.add_argument(
        "--wrist-camera-flip-roll-180",
        action="store_true",
        help=(
            "Flip the wrist camera 180 degrees around the local wrist-roll axis. "
            f"Equivalent to --wrist-camera-xyaxes {WRIST_CAMERA_FLIPPED_XYAXES!r}."
        ),
    )
    parser.add_argument(
        "--wrist-camera-pitch-down-deg",
        type=float,
        default=0.0,
        help=(
            "Pitch the wrist camera facing direction down toward the board by this many degrees, "
            "after applying any roll flip."
        ),
    )
    parser.add_argument(
        "--show-wrist-camera-marker",
        action="store_true",
        help=(
            "Add visual-only marker geometry at the wrist camera pose. "
            "Use for viewer inspection, not for generating training frames."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wrist_camera_xyaxes = WRIST_CAMERA_FLIPPED_XYAXES if args.wrist_camera_flip_roll_180 else args.wrist_camera_xyaxes
    if args.wrist_camera_pitch_down_deg:
        wrist_camera_xyaxes = _pitch_camera_down_xyaxes(wrist_camera_xyaxes, args.wrist_camera_pitch_down_deg)
    cfg = NudgeConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        tolerance_m=args.tolerance_m,
        min_offset_m=args.min_offset_m,
        max_offset_m=args.max_offset_m,
        candidate_count=args.candidate_count,
        push_distance_m=args.push_distance_m,
        min_push_distance_m=args.min_push_distance_m,
        push_distance_steps=args.push_distance_steps,
        approach_fan_deg=args.approach_fan_deg,
        lateral_approach_weight=args.lateral_approach_weight,
        stop_short_margin_m=args.stop_short_margin_m,
        nudge_wrist_flex_deg=args.nudge_wrist_flex_deg,
        motion_slowdown=args.motion_slowdown,
        fps=args.fps,
        robot_arm=args.robot_arm,
        episode_prefix=args.episode_prefix,
        stone_color=args.stone_color,
        no_neighbors=args.no_neighbors,
        arm_start_jitter_deg=args.arm_start_jitter_deg,
        nudge_tip_radius_m=args.nudge_tip_radius_m,
        wrist_camera_pos=_validate_float_tuple("wrist_camera_pos", args.wrist_camera_pos, 3),
        wrist_camera_xyaxes=_validate_float_tuple("wrist_camera_xyaxes", wrist_camera_xyaxes, 6),
        wrist_camera_fovy=str(float(args.wrist_camera_fovy)),
        show_wrist_camera_marker=args.show_wrist_camera_marker,
    )
    targets = args.target or default_targets()
    if args.viewer:
        run_viewer(
            cfg,
            targets,
            args.viewer_episode_index,
            ik_calibration=args.viewer_ik_calibration,
            real_joints=args.viewer_real_joints,
        )
        return
    summary = generate_dataset(args.output_dir, cfg, targets, force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
