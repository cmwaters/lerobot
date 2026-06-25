#!/usr/bin/env python
"""First MuJoCo sandbox for Go-board stone placement.

This intentionally starts below the robot layer: it builds a board, target
marker, side bowls, and rounded Go stones with ground-truth placement metrics.
The next step is to add a gripper and scripted teacher on top of this scene.
"""

from __future__ import annotations

import argparse
import math
import random
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import numpy as np


GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"
ROBOTSTUDIO_SO101_DIR = Path(__file__).resolve().parent / "assets" / "robotstudio_so101"
ROBOTSTUDIO_SO101_XML = ROBOTSTUDIO_SO101_DIR / "so101_new_calib.xml"
ROBOTSTUDIO_SO101_NORMALIZED_XML = Path(tempfile.gettempdir()) / "go_board_robotstudio_so101" / "so101_new_calib.xml"


@dataclass(frozen=True)
class BoardSpec:
    size: int = 19
    spacing_m: float = 0.022
    margin_m: float = 0.028
    thickness_m: float = 0.018
    stone_radius_m: float = 0.0108
    stone_half_height_m: float = 0.0036
    board_friction: str = "1.2 0.02 0.001"
    stone_friction: str = "0.8 0.02 0.001"

    @property
    def grid_span_m(self) -> float:
        return (self.size - 1) * self.spacing_m

    @property
    def board_half_extent_m(self) -> float:
        return self.grid_span_m / 2 + self.margin_m

    @property
    def board_top_z_m(self) -> float:
        return self.thickness_m / 2


@dataclass(frozen=True)
class ArmSpec:
    """Approximate SO-101 geometry for early MuJoCo experiments."""

    base_pos: tuple[float, float, float] = (-0.335, 0.0, 0.012)
    upper_arm_m: float = 0.105
    forearm_m: float = 0.105
    wrist_m: float = 0.055
    gripper_m: float = 0.045
    finger_length_m: float = 0.038
    open_width_m: float = 0.028


def parse_go_coord(coord: str, board_size: int = 19) -> tuple[int, int, str]:
    coord = coord.strip().upper()
    if len(coord) < 2:
        raise ValueError("Coordinate must look like K10.")
    columns = GO_COLUMNS if board_size == 19 else "".join(chr(ord("A") + i) for i in range(board_size))
    col = columns.find(coord[0])
    try:
        row = int(coord[1:]) - 1
    except ValueError as exc:
        raise ValueError("Coordinate must use a letter plus row number, for example K10.") from exc
    if col < 0 or row < 0 or row >= board_size:
        raise ValueError(f"Coordinate must be on a {board_size}x{board_size} board.")
    return row, col, f"{columns[col]}{row + 1}"


def intersection_xy(row: int, col: int, spec: BoardSpec) -> tuple[float, float]:
    center = (spec.size - 1) / 2
    x = (col - center) * spec.spacing_m
    y = (row - center) * spec.spacing_m
    return x, y


def _geom(
    name: str,
    geom_type: str,
    size: str,
    pos: str,
    rgba: str,
    *,
    material: str | None = None,
    friction: str | None = None,
    contact: bool = True,
) -> str:
    attrs = {
        "name": name,
        "type": geom_type,
        "size": size,
        "pos": pos,
        "rgba": rgba,
    }
    if material:
        attrs["material"] = material
    if friction:
        attrs["friction"] = friction
    if not contact:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
    rendered = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f"<geom {rendered}/>"


def _stone_body(name: str, color: str, pos: tuple[float, float, float], spec: BoardSpec) -> str:
    rgba = "0.96 0.94 0.86 1" if color == "white" else "0.02 0.018 0.014 1"
    size = f"{spec.stone_radius_m:.5f} {spec.stone_radius_m:.5f} {spec.stone_half_height_m:.5f}"
    x, y, z = pos
    return textwrap.dedent(
        f"""
        <body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}">
          <freejoint name="{name}_free"/>
          <geom name="{name}_geom" type="ellipsoid" size="{size}" rgba="{rgba}"
                mass="0.006" friction="{spec.stone_friction}"/>
        </body>
        """
    ).strip()


def _capsule(name: str, fromto: str, radius: float, rgba: str, *, mass: float = 0.04) -> str:
    return (
        f'<geom name="{name}" type="capsule" fromto="{fromto}" size="{radius:.5f}" '
        f'rgba="{rgba}" mass="{mass:.4f}"/>'
    )


def _box(name: str, size: str, pos: str, rgba: str, *, mass: float = 0.02) -> str:
    return f'<geom name="{name}" type="box" size="{size}" pos="{pos}" rgba="{rgba}" mass="{mass:.4f}"/>'


def robotstudio_so101_xml_for_include(destination: Path | None = None) -> Path:
    """Return a MuJoCo-include-safe copy of RobotStudio's SO-101 MJCF."""
    output_path = destination or ROBOTSTUDIO_SO101_NORMALIZED_XML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_mtime_ns = ROBOTSTUDIO_SO101_XML.stat().st_mtime_ns
    if (
        output_path.is_file()
        and output_path.stat().st_mtime_ns >= source_mtime_ns
    ):
        return output_path

    tree = ElementTree.parse(ROBOTSTUDIO_SO101_XML)
    root = tree.getroot()
    asset_dir = ROBOTSTUDIO_SO101_DIR / "assets"
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str(asset_dir / Path(mesh_file).name))
    for compiler in root.findall("compiler"):
        compiler.attrib.pop("meshdir", None)
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def _approx_so101_xml(spec: ArmSpec) -> tuple[str, str, str]:
    base_x, base_y, base_z = spec.base_pos
    upper = spec.upper_arm_m
    forearm = spec.forearm_m
    wrist = spec.wrist_m
    gripper = spec.gripper_m
    finger = spec.finger_length_m
    half_open = spec.open_width_m / 2
    blue = "0.16 0.28 0.78 1"
    dark = "0.05 0.055 0.065 1"
    metal = "0.72 0.72 0.68 1"
    pad = "0.025 0.025 0.023 1"

    arm_xml = textwrap.dedent(
        f"""
        <body name="so101_base" pos="{base_x:.5f} {base_y:.5f} {base_z:.5f}">
          <geom name="so101_base_geom" type="cylinder" size="0.040 0.014" rgba="{dark}" mass="0.180"/>
          <body name="so101_shoulder_pan_link" pos="0 0 0.024">
            <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-2.8 2.8" damping="0.35" armature="0.015"/>
            <geom name="shoulder_pan_hub" type="cylinder" size="0.028 0.020" rgba="{blue}" mass="0.090"/>
            <body name="so101_shoulder_lift_link" pos="0 0 0.036">
              <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-2.6 2.3" damping="0.45" armature="0.018"/>
              {_capsule("upper_arm_geom", f"0 0 0 {upper:.5f} 0 0", 0.014, blue, mass=0.090)}
              <body name="so101_elbow_link" pos="{upper:.5f} 0 0">
                <joint name="elbow_flex" type="hinge" axis="0 1 0" range="-2.6 2.6" damping="0.40" armature="0.014"/>
                <geom name="elbow_hub" type="sphere" size="0.024" rgba="{metal}" mass="0.050"/>
                {_capsule("forearm_geom", f"0 0 0 {forearm:.5f} 0 0", 0.012, blue, mass=0.075)}
                <body name="so101_wrist_flex_link" pos="{forearm:.5f} 0 0">
                  <joint name="wrist_flex" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="0.28" armature="0.010"/>
                  <geom name="wrist_flex_hub" type="sphere" size="0.019" rgba="{metal}" mass="0.035"/>
                  {_capsule("wrist_geom", f"0 0 0 {wrist:.5f} 0 0", 0.010, blue, mass=0.040)}
                  <body name="so101_wrist_roll_link" pos="{wrist:.5f} 0 0">
                    <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.1416 3.1416" damping="0.15" armature="0.006"/>
                    {_box("wrist_roll_block", "0.014 0.018 0.014", "0.012 0 0", metal, mass=0.030)}
                    <body name="so101_gripper_palm" pos="{gripper:.5f} 0 0">
                      <camera name="wrist" pos="-0.025 0 0.045" xyaxes="0 -1 0 0.35 0 0.94"/>
                      {_box("gripper_palm_geom", "0.018 0.024 0.010", "0 0 0", dark, mass=0.030)}
                      <body name="so101_left_finger" pos="0 {half_open:.5f} 0">
                        <joint name="gripper_left" type="slide" axis="0 1 0" range="0 0.030" damping="0.08"/>
                        {_box("left_finger_geom", f"{finger / 2:.5f} 0.0035 0.010", f"{finger / 2:.5f} 0 0", pad, mass=0.010)}
                      </body>
                      <body name="so101_right_finger" pos="0 {-half_open:.5f} 0">
                        <joint name="gripper_right" type="slide" axis="0 -1 0" range="0 0.030" damping="0.08"/>
                        {_box("right_finger_geom", f"{finger / 2:.5f} 0.0035 0.010", f"{finger / 2:.5f} 0 0", pad, mass=0.010)}
                      </body>
                      <site name="gripper_frame_link" pos="{finger:.5f} 0 0" size="0.006" rgba="0.0 1.0 0.25 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        """
    ).strip()

    equality_xml = textwrap.dedent(
        """
        <equality>
          <joint name="gripper_mirror" joint1="gripper_right" joint2="gripper_left" polycoef="0 1 0 0 0"/>
        </equality>
        """
    ).strip()

    actuator_xml = textwrap.dedent(
        """
        <actuator>
          <position name="shoulder_pan_act" joint="shoulder_pan" kp="35" ctrlrange="-2.8 2.8"/>
          <position name="shoulder_lift_act" joint="shoulder_lift" kp="45" ctrlrange="-2.6 2.3"/>
          <position name="elbow_flex_act" joint="elbow_flex" kp="38" ctrlrange="-2.6 2.6"/>
          <position name="wrist_flex_act" joint="wrist_flex" kp="28" ctrlrange="-2.4 2.4"/>
          <position name="wrist_roll_act" joint="wrist_roll" kp="12" ctrlrange="-3.1416 3.1416"/>
          <position name="gripper_act" joint="gripper_left" kp="80" ctrlrange="0 0.030"/>
        </actuator>
        """
    ).strip()

    return arm_xml, equality_xml, actuator_xml


def build_scene_xml(
    target_coord: str = "K10",
    spec: BoardSpec | None = None,
    white_stone_pos: tuple[float, float, float] | None = None,
    include_arm: bool = False,
    arm_spec: ArmSpec | None = None,
    board_center_m: tuple[float, float] = (0.0, 0.0),
    robotstudio_include_path: Path | None = None,
) -> str:
    spec = spec or BoardSpec()
    arm_spec = arm_spec or ArmSpec()
    target_row, target_col, normalized_coord = parse_go_coord(target_coord, spec.size)
    target_x, target_y = intersection_xy(target_row, target_col, spec)
    board_x, board_y = board_center_m
    target_x += board_x
    target_y += board_y
    half = spec.board_half_extent_m
    top_z = spec.board_top_z_m
    grid_z = top_z + 0.0008
    target_z = top_z + 0.0015
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
                f"{board_x + offset:.5f} {board_y:.5f} {grid_z:.5f}",
                "0.12 0.08 0.035 1",
                contact=False,
            )
        )
        grid_geoms.append(
            _geom(
                f"grid_row_{idx:02d}",
                "box",
                f"{line_half:.5f} {line_thickness:.5f} 0.00020",
                f"{board_x:.5f} {board_y + offset:.5f} {grid_z:.5f}",
                "0.12 0.08 0.035 1",
                contact=False,
            )
        )

    star_geoms = []
    for row in (3, 9, 15):
        for col in (3, 9, 15):
            x, y = intersection_xy(row, col, spec)
            x += board_x
            y += board_y
            star_geoms.append(
                _geom(
                    f"star_{row}_{col}",
                    "cylinder",
                    "0.00220 0.00025",
                    f"{x:.5f} {y:.5f} {grid_z + 0.0004:.5f}",
                    "0.06 0.04 0.02 1",
                    contact=False,
                )
            )

    target_marker = [
        _geom(
            f"target_{normalized_coord.lower()}",
            "cylinder",
            f"{spec.stone_radius_m * 1.35:.5f} 0.00035",
            f"{target_x:.5f} {target_y:.5f} {target_z:.5f}",
            "0.0 0.75 0.1 0.35",
            contact=False,
        ),
        _geom(
            "target_center",
            "cylinder",
            "0.00200 0.00050",
            f"{target_x:.5f} {target_y:.5f} {target_z + 0.0006:.5f}",
            "0.0 0.9 0.15 0.8",
            contact=False,
        ),
    ]

    stone_start_z = top_z + spec.stone_half_height_m * 4.0
    if white_stone_pos is None:
        white_stone_pos = (board_x - half - 0.055, board_y - 0.055, stone_start_z)
    stone_bodies = [
        _stone_body("white_stone", "white", white_stone_pos, spec),
        _stone_body("black_stone", "black", (board_x - half - 0.055, board_y + 0.055, stone_start_z), spec),
    ]

    left_bowl = _geom(
        "white_bowl",
        "cylinder",
        "0.042 0.012",
        f"{board_x - half - 0.060:.5f} {board_y - 0.055:.5f} {top_z:.5f}",
        "0.82 0.82 0.78 0.45",
        friction=spec.board_friction,
    )
    right_bowl = _geom(
        "black_bowl",
        "cylinder",
        "0.042 0.012",
        f"{board_x - half - 0.060:.5f} {board_y + 0.055:.5f} {top_z:.5f}",
        "0.12 0.12 0.12 0.45",
        friction=spec.board_friction,
    )
    arm_xml, equality_xml, actuator_xml = _approx_so101_xml(arm_spec) if include_arm else ("", "", "")
    include_xml = ""
    compiler_xml = '<compiler angle="radian"/>'
    if robotstudio_include_path is not None:
        include_xml = f'<include file="{escape(str(robotstudio_include_path))}"/>'
        compiler_xml = ""

    return textwrap.dedent(
        f"""
        <mujoco model="go_board_placement_sandbox">
          {include_xml}
          {compiler_xml}
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>
            <rgba haze="0.85 0.88 0.92 1"/>
          </visual>
          <asset>
            <texture name="wood_tex" type="2d" builtin="checker"
                     rgb1="0.76 0.52 0.25" rgb2="0.68 0.43 0.18"
                     width="256" height="256" mark="edge" markrgb="0.25 0.14 0.05"/>
            <material name="board_mat" texture="wood_tex" texrepeat="2 2"
                      specular="0.18" shininess="0.35"/>
          </asset>
          <worldbody>
            <light name="overhead_light" pos="0 -0.25 0.8" diffuse="0.85 0.82 0.75"/>
            <camera name="overhead" pos="0 0 0.75" xyaxes="1 0 0 0 1 0"/>
            <camera name="side" pos="0.30 -0.45 0.26" xyaxes="0.83 0.55 0 -0.18 0.27 0.95"/>
            {_geom(
                "table",
                "box",
                f"{half + 0.11:.5f} {half + 0.04:.5f} 0.01000",
                f"{board_x - 0.035:.5f} {board_y:.5f} {-0.010:.5f}",
                "0.35 0.32 0.28 1",
                friction=spec.board_friction,
            )}
            {_geom(
                "board",
                "box",
                f"{half:.5f} {half:.5f} {spec.thickness_m / 2:.5f}",
                f"{board_x:.5f} {board_y:.5f} 0",
                "0.74 0.48 0.20 1",
                material="board_mat",
                friction=spec.board_friction,
            )}
            {" ".join(grid_geoms)}
            {" ".join(star_geoms)}
            {" ".join(target_marker)}
            {left_bowl}
            {right_bowl}
            {arm_xml}
            {" ".join(stone_bodies)}
          </worldbody>
          {equality_xml}
          {actuator_xml}
        </mujoco>
        """
    ).strip()


class GoBoardMuJoCoSim:
    def __init__(
        self,
        target_coord: str = "K10",
        spec: BoardSpec | None = None,
        arm_spec: ArmSpec | None = None,
        board_center_m: tuple[float, float] = (0.0, 0.0),
        white_stone_pos: tuple[float, float, float] | None = None,
        include_arm: bool = False,
        robotstudio_arm: bool = False,
        reset_on_init: bool = True,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo is not installed in this environment. Install it with `uv add mujoco` "
                "or try `uv run --with mujoco python examples/go_board/mujoco_board_sim.py --drop-demo`."
            ) from exc

        self.mujoco = mujoco
        self.spec = spec or BoardSpec()
        self.arm_spec = arm_spec or ArmSpec()
        self.board_center_m = np.array(board_center_m, dtype=np.float64)
        self.include_arm = include_arm or robotstudio_arm
        self.robotstudio_arm = robotstudio_arm
        self.target_row, self.target_col, self.target_coord = parse_go_coord(target_coord, self.spec.size)
        self.target_xy = (
            np.array(intersection_xy(self.target_row, self.target_col, self.spec), dtype=np.float64)
            + self.board_center_m
        )
        robotstudio_include = robotstudio_so101_xml_for_include() if robotstudio_arm else None
        self.model = mujoco.MjModel.from_xml_string(
            build_scene_xml(
                self.target_coord,
                self.spec,
                white_stone_pos=white_stone_pos,
                include_arm=include_arm,
                arm_spec=self.arm_spec,
                board_center_m=board_center_m,
                robotstudio_include_path=robotstudio_include,
            )
        )
        self.data = mujoco.MjData(self.model)
        if reset_on_init:
            self.reset()
        else:
            self.mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: int | None = None, stone: str = "white_stone") -> None:
        rng = random.Random(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        half = self.spec.board_half_extent_m
        x = self.board_center_m[0] + rng.uniform(-half * 0.25, half * 0.25)
        y = self.board_center_m[1] + rng.uniform(-half * 0.25, half * 0.25)
        z = self.spec.board_top_z_m + self.spec.stone_half_height_m * 5.0
        self.set_stone_pose(stone, np.array([x, y, z], dtype=np.float64))
        self.mujoco.mj_forward(self.model, self.data)

    def set_stone_pose(self, stone: str, xyz: np.ndarray) -> None:
        joint = self.data.joint(f"{stone}_free")
        joint.qpos[:3] = xyz
        joint.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        joint.qvel[:] = 0.0

    def step(self, n: int = 1) -> None:
        self.mujoco.mj_step(self.model, self.data, nstep=n)

    def stone_xy(self, stone: str = "white_stone") -> np.ndarray:
        return self.data.body(stone).xpos[:2].copy()

    def target_error_m(self, stone: str = "white_stone") -> float:
        return float(np.linalg.norm(self.stone_xy(stone) - self.target_xy))

    def is_success(self, tolerance_m: float = 0.0075, stone: str = "white_stone") -> bool:
        return self.target_error_m(stone) <= tolerance_m

    def set_arm_control(
        self,
        *,
        shoulder_pan: float = 0.0,
        shoulder_lift: float = 0.0,
        elbow_flex: float = 0.0,
        wrist_flex: float = 0.0,
        wrist_roll: float = 0.0,
        gripper: float = 50.0,
        degrees: bool = True,
    ) -> None:
        if not self.include_arm:
            raise RuntimeError("Arm controls are unavailable. Create the sim with include_arm=True.")
        arm_actuators = {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
            "wrist_roll": wrist_roll,
        }
        if degrees:
            arm_actuators = {name: math.radians(value) for name, value in arm_actuators.items()}
        if self.robotstudio_arm:
            values = arm_actuators
            gripper_actuator = "gripper"
        else:
            values = {f"{name}_act": value for name, value in arm_actuators.items()}
            gripper_actuator = "gripper_act"
        for name, value in values.items():
            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise KeyError(f"Actuator not found: {name}")
            self.data.ctrl[actuator_id] = value
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        if actuator_id < 0:
            raise KeyError(f"Actuator not found: {gripper_actuator}")
        if self.robotstudio_arm:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = low + np.clip(gripper, 0.0, 100.0) / 100.0 * (high - low)
        else:
            self.data.ctrl[actuator_id] = np.clip(gripper, 0.0, 100.0) / 100.0 * self.arm_spec.open_width_m

    def gripper_position(self) -> np.ndarray:
        if not self.include_arm:
            raise RuntimeError("The arm is not included in this simulation.")
        site_name = "gripperframe" if self.robotstudio_arm else "gripper_frame_link"
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"Site not found: {site_name}")
        return self.data.site_xpos[site_id].copy()


def run_drop_demo(args: argparse.Namespace) -> None:
    board_center = (args.board_center_x, args.board_center_y)
    sim = GoBoardMuJoCoSim(
        target_coord=args.target,
        include_arm=args.include_arm,
        robotstudio_arm=args.robotstudio_arm,
        board_center_m=board_center,
    )
    stone_xyz = np.array([sim.target_xy[0], sim.target_xy[1], sim.spec.board_top_z_m + args.drop_height], dtype=np.float64)
    sim.set_stone_pose("white_stone", stone_xyz)
    sim.mujoco.mj_forward(sim.model, sim.data)
    sim.step(args.steps)
    print(
        {
            "target": sim.target_coord,
            "stone_xy_m": np.round(sim.stone_xy(), 5).tolist(),
            "target_xy_m": np.round(sim.target_xy, 5).tolist(),
            "target_error_mm": round(sim.target_error_m() * 1000, 2),
            "success": sim.is_success(),
        }
    )


def run_arm_demo(args: argparse.Namespace) -> None:
    board_center = (args.board_center_x, args.board_center_y)
    sim = GoBoardMuJoCoSim(
        target_coord=args.target,
        include_arm=not args.robotstudio_arm,
        robotstudio_arm=args.robotstudio_arm,
        board_center_m=board_center,
    )
    before = sim.gripper_position()
    sim.set_arm_control(
        shoulder_pan=-40.0,
        shoulder_lift=35.0,
        elbow_flex=-80.0,
        wrist_flex=45.0,
        wrist_roll=0.0,
        gripper=70.0,
    )
    sim.step(args.steps)
    after = sim.gripper_position()
    print(
        {
            "target": sim.target_coord,
            "gripper_start_m": np.round(before, 4).tolist(),
            "gripper_after_m": np.round(after, 4).tolist(),
            "target_xy_m": np.round(sim.target_xy, 4).tolist(),
            "arm_note": (
                "RobotStudio SO-101 model loaded from so101_new_calib.xml."
                if args.robotstudio_arm
                else "Approximate SO-101 geometry, not calibrated URDF."
            ),
        }
    )


def run_viewer(args: argparse.Namespace) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise ImportError(
            "The MuJoCo viewer is unavailable. Install MuJoCo with `uv add mujoco`, "
            "or try `uv run --with mujoco python examples/go_board/mujoco_board_sim.py --viewer`."
        ) from exc

    spec = BoardSpec()
    board_center = (args.board_center_x, args.board_center_y)
    target_row, target_col, _coord = parse_go_coord(args.target, spec.size)
    target_x, target_y = intersection_xy(target_row, target_col, spec)
    target_x += board_center[0]
    target_y += board_center[1]
    white_stone_pos = None
    reset_on_init = True
    if args.viewer_drop_demo:
        white_stone_pos = (
            target_x,
            target_y,
            spec.board_top_z_m + args.drop_height,
        )
        reset_on_init = False
        print("Viewer drop demo: press Play/Space in MuJoCo if the simulation is paused.")

    include_arm = (args.include_arm or args.viewer_arm_demo) and not args.robotstudio_arm
    sim = GoBoardMuJoCoSim(
        target_coord=args.target,
        spec=spec,
        board_center_m=board_center,
        white_stone_pos=white_stone_pos,
        include_arm=include_arm,
        robotstudio_arm=args.robotstudio_arm,
        reset_on_init=reset_on_init,
    )
    if args.viewer_arm_demo:
        sim.set_arm_control(
            shoulder_pan=-40.0,
            shoulder_lift=35.0,
            elbow_flex=-80.0,
            wrist_flex=45.0,
            wrist_roll=0.0,
            gripper=70.0,
        )
        arm_name = "RobotStudio SO-101" if args.robotstudio_arm else "approximate SO-101"
        print(f"Viewer arm demo: press Play/Space to watch the {arm_name} move to its target pose.")
    mujoco.viewer.launch(sim.model, sim.data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="K10", help="Target Go coordinate, e.g. K10.")
    parser.add_argument("--export-xml", type=Path, help="Write the generated MJCF scene XML to this path.")
    parser.add_argument("--drop-demo", action="store_true", help="Drop a white stone above the target and report error.")
    parser.add_argument("--arm-demo", action="store_true", help="Step an approximate SO-101 arm toward a visible pose and print gripper position.")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer.")
    parser.add_argument("--viewer-drop-demo", action="store_true", help="Open the viewer with a stone suspended above the target.")
    parser.add_argument("--viewer-arm-demo", action="store_true", help="Open the viewer with the SO-101 moving to a target pose.")
    parser.add_argument("--include-arm", action="store_true", help="Include the approximate SO-101 arm in exported/viewed scenes.")
    parser.add_argument(
        "--robotstudio-arm",
        action="store_true",
        help="Include RobotStudio's SO-101 MuJoCo model from examples/go_board/assets/robotstudio_so101.",
    )
    parser.add_argument("--board-center-x", type=float, default=0.0, help="Board center X in MuJoCo meters.")
    parser.add_argument("--board-center-y", type=float, default=0.0, help="Board center Y in MuJoCo meters.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--drop-height", type=float, default=0.055)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.export_xml:
        args.export_xml.parent.mkdir(parents=True, exist_ok=True)
        robotstudio_include = None
        if args.robotstudio_arm:
            robotstudio_include = robotstudio_so101_xml_for_include(
                args.export_xml.with_name("robotstudio_so101.normalized.xml")
            )
        args.export_xml.write_text(
            build_scene_xml(
                args.target,
                include_arm=args.include_arm and not args.robotstudio_arm,
                board_center_m=(args.board_center_x, args.board_center_y),
                robotstudio_include_path=robotstudio_include,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.export_xml}")
    if args.drop_demo:
        run_drop_demo(args)
    if args.arm_demo:
        run_arm_demo(args)
    if args.viewer:
        run_viewer(args)
    if not args.export_xml and not args.drop_demo and not args.arm_demo and not args.viewer:
        print(
            build_scene_xml(
                args.target,
                include_arm=args.include_arm and not args.robotstudio_arm,
                board_center_m=(args.board_center_x, args.board_center_y),
                robotstudio_include_path=robotstudio_so101_xml_for_include() if args.robotstudio_arm else None,
            )
        )


if __name__ == "__main__":
    main()
