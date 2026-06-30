# Go Board CV Example

This example detects the board state from a fixed overhead camera snapshot. It reports
all detected stones as Go coordinates with color labels.

## Usage

For a clear image where the full board border is visible:

```bash
uv run python examples/go_board/detect_board_state.py snapshot.jpg \
  --json-output outputs/go_board/state.json \
  --debug-output outputs/go_board/debug.jpg
```

For reliable runs, pass the four board corners in image pixels, ordered
top-left, top-right, bottom-right, bottom-left:

```bash
uv run python examples/go_board/detect_board_state.py snapshot.jpg \
  --corners "105,82 913,76 930,884 88,891" \
  --json-output outputs/go_board/state.json \
  --debug-output outputs/go_board/debug.jpg
```

The output uses normal Go columns on a 19x19 board, skipping `I`:

```json
{
  "stones": [
    {
      "coord": "Q16",
      "row": 15,
      "col": 15,
      "color": "white",
      "confidence": 0.91
    }
  ]
}
```

Rows and columns in JSON are zero-based. The human coordinate `Q16` means
column `Q`, row `16`.

## Tuning

Use `--debug-output` first. If the overlay marks too many board lines as black
stones, lower `--sample-radius-ratio` or lower `--black-l-threshold`. If white
stones are missed, reduce `--white-l-threshold` or raise `--white-s-threshold`.

This detector is meant as a practical starting point for a fixed overhead camera.
Once you have a few real snapshots, tune the thresholds for your lighting and
stone material.

## Recording Dashboard

The dashboard is a local web UI for data-collection setup. It shows two camera
feeds, joint telemetry, and optional end-effector pose.

Default real-device mode reads `examples/go_board/dashboard_config.json`:

```bash
uv run python examples/go_board/recording_dashboard.py
```

The config stores the camera indices, follower port, and leader port so the
normal dashboard startup stays a one-command flow.

### Laptop/desktop profiles

Use the profile manager when switching between the laptop recording rig and the
desktop model-running setup:

```bash
# Laptop: local cameras + SO-101 leader/follower for teleoperation recording.
uv run python examples/go_board/manage_dashboard.py laptop

# If macOS renames the arm ports after reconnecting:
uv run python examples/go_board/manage_dashboard.py laptop \
  --follower-port /dev/tty.usbmodemFOLLOWER \
  --leader-port /dev/tty.usbmodemLEADER

# Desktop: start the dashboard on the desktop over Tailscale/SSH.
uv run python examples/go_board/manage_dashboard.py desktop

# Copy profile files to the desktop after editing them locally.
uv run python examples/go_board/manage_dashboard.py sync-desktop-configs

# Status/stop helpers.
uv run python examples/go_board/manage_dashboard.py status-laptop
uv run python examples/go_board/manage_dashboard.py stop-laptop
uv run python examples/go_board/manage_dashboard.py status-desktop
uv run python examples/go_board/manage_dashboard.py stop-desktop
```

Profile files:

- `dashboard_config.laptop.json`: macOS camera indices and USB modem ports for recording.
- `dashboard_config.desktop.json`: Linux `/dev/v4l/by-id` and `/dev/serial/by-id` paths for the desktop.
- `dashboard_config.json`: convenience default, currently matching the laptop profile.

The main dashboard includes an alignment panel. `Set Rest from Leader` snapshots
the current leader joint angles into `robot.rest_position` in
`dashboard_config.json`; `Move Follower to Rest` sends the follower to that saved
pose. The leader/follower delta list shows `follower - leader` for each shared
joint. Move the leader until those deltas are near zero before starting
teleoperation.

Open the CV annotation dashboard from the same server:

```text
http://127.0.0.1:8766/annotate
```

Use it to create a small ground-truth test set: arrange the real board, click
the matching black and white stones on the virtual board, then save. Each save
writes a paired `.jpg` and `.json` file to:

```text
examples/go_board/cv_snapshots/
```

You can override that location with `--annotation-dir`.

The annotation dashboard also includes 10 suggested patterns. Pick one, place
those stones on the real board, click `Load Pattern` to fill the virtual board,
then save. `Detect Current` runs the CV detector on the live overhead frame and
loads the detector's current guess into the virtual board so you can compare or
correct it before saving.

Evaluate the saved snapshots against the detector:

```bash
uv run python examples/go_board/evaluate_board_cv.py \
  --json-output examples/go_board/cv_snapshots/evaluation.json
```

The evaluator writes detector overlays to `examples/go_board/cv_snapshots/debug/`
and expected-vs-detected overlays to
`examples/go_board/cv_snapshots/expected_vs_detected/`. In comparison overlays,
green labels are the clicked ground truth and red labels are CV detections.

The board-state buttons use the configured `board.camera` feed. For robust CV,
set `board.corners_tl_tr_br_bl` in `dashboard_config.json` once you know the
four board corners in overhead-camera pixels. If corners are `null`, the
detector will try to find the board automatically.

If the overhead camera is rotated relative to the robot-facing board, set
`board.camera_to_robot_rotation_degrees` to `0`, `90`, `180`, or `270`. The value
is the clockwise rotation needed to turn camera-detected coordinates into the
robot-facing coordinates used by the annotation board and board-state delta.

Mock mode:

```bash
uv run python examples/go_board/recording_dashboard.py --mock
```

Two OpenCV cameras:

```bash
uv run python examples/go_board/recording_dashboard.py \
  --camera overhead=0,1280x720@30 \
  --camera wrist=1,640x480@30
```

If the feeds are around the wrong way, swap the labels rather than changing the
dashboard code:

```bash
uv run python examples/go_board/recording_dashboard.py \
  --camera overhead=1,640x480@30 \
  --camera wrist=0,1280x720@30
```

With SO-101 joint telemetry:

```bash
uv run python examples/go_board/recording_dashboard.py \
  --camera overhead=0,1280x720@30 \
  --camera wrist=1,640x480@30 \
  --so101-port /dev/tty.usbmodemXXXX \
  --robot-id go_follower
```

If you also pass `--urdf-path ./SO101/so101_new_calib.urdf`, the dashboard will
compute forward kinematics for the end effector. Without a URDF, it still shows
live joint angles.

## MuJoCo Placement Sandbox

`mujoco_board_sim.py` is a first-pass simulation sandbox for the physical
placement layer. It builds a 19x19 board, visual grid, side bowls, target
marker, and rounded Go stones. Stones are modeled as flattened ellipsoid geoms,
which is closer to the real rounded pill/lens shape than a cylinder.

Install MuJoCo if it is not already in your local environment:

```bash
uv add mujoco
```

For a no-project-change trial, you can also prefix the commands with:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py
```

Export the generated MJCF scene:

```bash
uv run python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --export-xml outputs/go_board/mujoco_scene.xml
```

Run a tiny physics smoke test that drops a white stone above the target and
prints the final target error:

```bash
uv run python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --drop-demo
```

Open an interactive viewer with an obvious stone-drop demo:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --viewer \
  --viewer-drop-demo
```

If you install MuJoCo into the project with `uv add mujoco`, you can omit
`--with mujoco`. We avoid `mjpython` here because uv's standalone Python can
miss the `libpython` dynamic library that `mjpython` expects on macOS.

If the viewer opens paused, press Play or Space. Reset should put the white
stone back above the target when using `--viewer-drop-demo`.

Add the first-pass SO-101 arm approximation:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --viewer \
  --viewer-arm-demo
```

This arm is native MJCF scaffolding with the same joint-channel names used by
the Go dataset (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`). It is not a calibrated RobotStudio URDF conversion
yet. Use it to start testing scene scale, actuator directions, and staged
pickup/place logic.

Run a non-GUI arm smoke test:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --arm-demo
```

Export a standalone scene with the board and arm:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --include-arm \
  --export-xml outputs/go_board/mujoco_so101_scene.xml
```

Use this as the canonical inspection scene. It contains the blue RobotStudio
SO-101 arm, board, bowls, and stones:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --robotstudio-arm \
  --board-center-x 0.280 \
  --board-center-y -0.030 \
  --viewer \
  --viewer-arm-demo
```

The RobotStudio assets live in
`examples/go_board/assets/robotstudio_so101/`. That folder includes the original
`so101_new_calib.urdf`, plus RobotStudio's native MuJoCo `so101_new_calib.xml`.
The simulator loads the native MuJoCo file because it preserves the same model
while avoiding a URDF-to-MJCF conversion step at runtime. The generated include
uses absolute STL mesh paths and recolors the arm blue in a normalized copy so
MuJoCo can find meshes when the Go-board scene is built from an XML string.
Normal runs put that copy under the system temp directory; `--export-xml` writes
it next to the exported scene.

For the first physical alignment pass, leave the RobotStudio base at MuJoCo
origin and move the board with:

```bash
--board-center-x 0.280 --board-center-y -0.030
```

Those values are meters. With the measured board footprint, `x=0.280` puts the
near board edge about 60 mm from the front edge of the RobotStudio SO-101 base
mesh. The `y=-0.030` offset represents the arm sitting a few centimeters left of
the board centerline.

Run a non-GUI RobotStudio arm smoke test:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --robotstudio-arm \
  --board-center-x 0.280 \
  --board-center-y -0.030 \
  --arm-demo
```

### Synthetic Nudge Data

`generate_mujoco_nudge_data.py` creates short MuJoCo nudge-only correction
episodes. A target stone starts near an intended coordinate, optional neighboring
stones are placed around it, and a candidate-search teacher chooses board-local
push deltas that reduce target error while penalizing neighbor disturbance.

Generate a small starter set:

```bash
uv run python examples/go_board/generate_mujoco_nudge_data.py \
  --episodes 20 \
  --output-dir outputs/go_board_mujoco_nudge_recordings \
  --force
```

Generate the same nudge task with RobotStudio's SO-101 arm meshes in the scene
and SO-101-shaped joint state/action telemetry:

```bash
uv run --with mujoco python examples/go_board/generate_mujoco_nudge_data.py \
  --robot-arm \
  --episodes 20 \
  --output-dir outputs/go_board_mujoco_nudge_robot_recordings \
  --force
```

Open one seeded robot-arm nudge scenario in MuJoCo's interactive viewer:

```bash
uv run --with mujoco python examples/go_board/generate_mujoco_nudge_data.py \
  --robot-arm \
  --viewer \
  --viewer-episode-index 0 \
  --target K10
```

Convert the robot-arm synthetic recordings with the normal LeRobot converter:

```bash
uv run --extra dataset python examples/go_board/convert_recordings_to_lerobot.py \
  --recordings-dir outputs/go_board_mujoco_nudge_robot_recordings \
  --dataset-root outputs/datasets/go_board_mujoco_nudge_robot_v1 \
  --repo-id callum/go_board_mujoco_nudge_robot_v1 \
  --force \
  --no-done-env-state
```

Render a watchable side-by-side MP4 preview:

```bash
uv run python examples/go_board/visualize_mujoco_nudge_data.py \
  --input-dir outputs/go_board_mujoco_nudge_recordings \
  --output outputs/go_board_mujoco_nudge_preview.mp4 \
  --max-episodes 8 \
  --fps 5
```

The output is intentionally under `outputs/`, which is git-ignored. Each episode
folder contains:

- `metadata.json`
- `telemetry.jsonl`
- `frames/overhead/*.jpg`
- `frames/wrist/*.jpg`

The synthetic telemetry stores board-local state under
`telemetry.synthetic`, including `stone_xy_m`, `stone_offset_xy_m`,
`neighbor_xy_m`, `target_error_m`, `done`, and the chosen push action:

```json
{
  "push_dx_m": 0.004,
  "push_dy_m": -0.002,
  "push_distance_m": 0.007
}
```

This first generator uses MuJoCo for scene rendering and neighbor context, but
the teacher is still an oracle-style board-local nudge controller. With
`--robot-arm`, the scene includes RobotStudio's SO-101 MuJoCo model and STL
meshes, the wrist camera is attached to the RobotStudio gripper body, and
`telemetry.joints` / `telemetry.leader_joints` are populated with simulated
SO-101 joint values. The current wrist camera pose is a simulation mount that
must be tuned against the physical camera mount before using the renders as
serious sim-to-real data. The current version moves the stone according to the
teacher while making the arm follow matching joint-space targets; the next
fidelity step is replacing that oracle stone motion with contact-only gripper
motion.
