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

Use RobotStudio's SO-101 model instead of the local approximation:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --robotstudio-arm \
  --board-center-x 0.30 \
  --viewer \
  --viewer-arm-demo
```

The RobotStudio assets live in
`examples/go_board/assets/robotstudio_so101/`. That folder includes the original
`so101_new_calib.urdf`, plus RobotStudio's native MuJoCo `so101_new_calib.xml`.
The simulator loads the native MuJoCo file because it preserves the same model
while avoiding a URDF-to-MJCF conversion step at runtime. The generated include
uses absolute STL mesh paths in a normalized copy so MuJoCo can find meshes when
the Go-board scene is built from an XML string. Normal runs put that copy under
the system temp directory; `--export-xml` writes it next to the exported scene.

For the first physical alignment pass, leave the RobotStudio base at MuJoCo
origin and move the board with:

```bash
--board-center-x 0.30 --board-center-y 0.00
```

Those values are meters. The `x=0.30` default suggestion puts the board under
the side of the RobotStudio gripper frame in the initial scene; tune it from the
overhead camera view once the physical base-to-board offset is clearer.

Run a non-GUI RobotStudio arm smoke test:

```bash
uv run --with mujoco python examples/go_board/mujoco_board_sim.py \
  --target K10 \
  --robotstudio-arm \
  --board-center-x 0.30 \
  --arm-demo
```

This sandbox intentionally starts below the robot layer. The next useful steps
are adding a simple gripper, then a scripted pick-carry-release teacher that can
export phase labels such as `stone_held`, `stone_on_board`, and
`target_error_mm`.
