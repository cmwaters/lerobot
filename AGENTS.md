This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

## Local Go Board Robot Workflow

The user is building a Go-stone placement pipeline around `examples/go_board/`.
Treat this as an active local experiment, not generic LeRobot usage.

### Dashboard and Hardware

- Main dashboard:
  ```bash
  uv run python examples/go_board/recording_dashboard.py
  ```
  Default URL is `http://127.0.0.1:8766/`.
- CV annotation page is served by the same process at
  `http://127.0.0.1:8766/annotate`.
- Persistent dashboard settings live in
  `examples/go_board/dashboard_config.json`. This includes camera mapping,
  board corners, grid curve, camera-to-robot rotation, SO-101 ports, and rest
  pose.
- Do not use mock robot data when the user says hardware is connected. If real
  arm data is missing, surface the concrete device/port/calibration error.
- SO-101 calibration files may differ between LeRobot tools and local
  experiments. Inspect the actual calibration files before inferring from
  symptoms. The common cache location is
  `~/.cache/huggingface/lerobot/calibration`.
- The dashboard uses a slower ramp when moving the follower to rest and when
  starting teleop from the leader pose. Preserve that behavior unless the user
  explicitly wants snappier motion.

### Recording Data

- A valid training recording should add exactly one Go stone: one `added`
  delta, no `removed`, no `changed`.
- Dashboard recordings live under `examples/go_board/recordings/<recording>/`
  with:
  - `metadata.json`
  - `telemetry.jsonl`
  - `frames/overhead/*.jpg`
  - `frames/wrist/*.jpg`
  - optional `overhead_processed/*.jpg`
- Recording stop flow is important: stop teleop, move follower to rest, capture
  final board state, compute the board delta, rename the recording with the move
  name, then start overhead post-processing in the background.
- `process_overhead_recording.py` overlays the target stone onto each raw
  overhead frame and writes `overhead_processed/`. Raw frames under
  `frames/overhead/` must not be modified.
  ```bash
  uv run python examples/go_board/process_overhead_recording.py \
    examples/go_board/recordings/<recording_name>

  uv run python examples/go_board/process_overhead_recording.py \
    examples/go_board/recordings --all --force
  ```
- Dashboard replay should prefer `overhead_processed` for the overhead camera
  when present, while wrist replay stays raw.

### Go Board CV

- Ground-truth CV snapshots live in `examples/go_board/cv_snapshots/` as paired
  `.jpg` and `.json` files.
- Evaluate detector snapshots with:
  ```bash
  uv run python examples/go_board/evaluate_board_cv.py \
    --json-output examples/go_board/cv_snapshots/evaluation.json
  ```
- The detector and overlay rely on:
  - `board.corners_tl_tr_br_bl`
  - `board.overlay_fisheye_k`
  - `board.camera_to_robot_rotation_degrees`
  - board size, normally `19`
- The yellow overlay grid should be the single source of truth for projected
  intersections. If the board/camera moves, update corners and grid curve in
  metadata or dashboard config, then re-evaluate.
- JSON row/col values are zero-based. Human Go coordinates skip `I`.

### LeRobot Dataset Conversion

- Convert valid recordings to a LeRobot image dataset with:
  ```bash
  uv run python examples/go_board/convert_recordings_to_lerobot.py \
    --dataset-root outputs/datasets/go_board_act_v1 \
    --repo-id callum/go_board_act_v1 \
    --force
  ```
- The converter filters recordings to exactly one added stone and uses only
  telemetry rows where `telemetry.teleop_enabled == true`.
- Dataset features are:
  - `observation.images.overhead`: processed overhead image if available,
    resized to 224x224 RGB
  - `observation.images.wrist`: raw wrist image, resized to 224x224 RGB
  - `observation.state`: follower joints
  - `action`: leader joints
- Joint order is:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper`.
- A known full conversion had 100 episodes and 20,421 frames. Verify current
  counts from `outputs/datasets/go_board_act_v1/go_conversion_summary.json`
  rather than assuming they are unchanged.

### ACT Training on Desktop GPU

- The user's GPU desktop is reachable over Tailscale:
  ```bash
  ssh cal@desktop
  ```
- The remote training workspace used so far is:
  `~/lerobot-go-train/lerobot`.
- The desktop may not include `~/.local/bin` in non-interactive SSH `PATH`.
  Call uv explicitly as `/home/cal/.local/bin/uv` when in doubt.
- Verify CUDA before training:
  ```bash
  ssh cal@desktop 'cd ~/lerobot-go-train/lerobot && \
    /home/cal/.local/bin/uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"'
  ```
- Sync training dependencies on the desktop with:
  ```bash
  ssh cal@desktop 'cd ~/lerobot-go-train/lerobot && \
    /home/cal/.local/bin/uv sync --locked --extra dataset --extra training'
  ```
- A sanity check before long ACT runs:
  ```bash
  ssh cal@desktop 'cd ~/lerobot-go-train/lerobot && \
    /home/cal/.local/bin/uv run lerobot-train \
      --dataset.repo_id=callum/go_board_act_v1 \
      --dataset.root=outputs/datasets/go_board_act_v1 \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.chunk_size=100 \
      --policy.n_action_steps=100 \
      --output_dir=outputs/train/act_go_board_v1_gpu_config_check \
      --job_name=act_go_board_v1_gpu_config_check \
      --batch_size=8 \
      --steps=2 \
      --num_workers=2 \
      --save_freq=2 \
      --log_freq=1 \
      --eval_freq=0 \
      --wandb.enable=false'
  ```
- Long-run command pattern:
  ```bash
  ssh cal@desktop 'cd ~/lerobot-go-train/lerobot && mkdir -p outputs/train/logs && \
    nohup /home/cal/.local/bin/uv run lerobot-train \
      --dataset.repo_id=callum/go_board_act_v1 \
      --dataset.root=outputs/datasets/go_board_act_v1 \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.chunk_size=100 \
      --policy.n_action_steps=100 \
      --output_dir=outputs/train/act_go_board_v1_30k \
      --job_name=act_go_board_v1_30k \
      --batch_size=8 \
      --steps=30000 \
      --num_workers=2 \
      --save_freq=5000 \
      --log_freq=100 \
      --eval_freq=0 \
      --wandb.enable=false \
      > outputs/train/logs/act_go_board_v1_30k.log 2>&1 < /dev/null & echo $!'
  ```
- Monitor with:
  ```bash
  ssh cal@desktop 'tail -f ~/lerobot-go-train/lerobot/outputs/train/logs/act_go_board_v1_30k.log'
  ssh cal@desktop 'nvidia-smi'
  ssh cal@desktop 'pgrep -af "lerobot-train|act_go_board_v1"'
  ```
- On the RTX 3060 12GB, batch size 8 and 100-step ACT chunks passed a two-step
  CUDA sanity run. Early speed for 30k steps was roughly 13-15 hours, but always
  estimate from the current log/progress bar.
