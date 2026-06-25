#!/usr/bin/env python
"""Tune Go-board CV parameters against saved annotations.

The search is deliberately small and inspectable: it performs repeated
bracket-halving passes over each global detector parameter, keeping the best
score after each probe. Snapshot-local geometry such as corners and grid curve
is read from each annotation, but detector thresholds remain global.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_board_state import board_state_from_image, transform_board_state
from evaluate_board_cv import compare_occupied, load_config_rotation


DEFAULT_SNAPSHOT_DIR = Path("examples/go_board/cv_snapshots")
DEFAULT_CONFIG_PATH = Path("examples/go_board/dashboard_config.json")
TUNING_KEYS = (
    "sample_radius_ratio",
    "black_l_threshold",
    "white_l_threshold",
    "white_s_threshold",
    "stone_min_radius_ratio",
    "stone_max_radius_ratio",
    "stone_min_circularity",
    "stone_max_snap_distance_ratio",
    "black_grid_min_edge_score",
)


@dataclass
class Case:
    path: Path
    image: np.ndarray
    corners: np.ndarray | None
    size: int
    rotation: int
    overlay_fisheye_k: float
    expected: dict[str, str]
    summary: dict[str, int]


@dataclass
class Score:
    passed: int
    total_errors: int
    missing: int
    extra: int
    wrong_color: int
    failures: list[dict[str, Any]]

    def rank(self) -> tuple[int, int, int, int, int]:
        # Prefer passing more files, then fewer total errors. Extra detections
        # are slightly more annoying for the current detector, so break ties
        # against them before missing stones and wrong colors.
        return (self.passed, -self.total_errors, -self.extra, -self.missing, -self.wrong_color)


def load_config_params(path: Path) -> dict[str, float]:
    defaults = {
        "sample_radius_ratio": 0.34,
        "black_l_threshold": 80.0,
        "white_l_threshold": 165.0,
        "white_s_threshold": 75.0,
        "stone_min_radius_ratio": 0.2,
        "stone_max_radius_ratio": 0.48,
        "stone_min_circularity": 0.45,
        "stone_max_snap_distance_ratio": 0.52,
        "black_grid_min_edge_score": 0.18,
    }
    if not path.is_file():
        return defaults
    data = json.loads(path.read_text())
    board = data.get("board", {})
    return {key: float(board.get(key, defaults[key])) for key in TUNING_KEYS}


def load_cases(snapshot_dir: Path, config_path: Path) -> list[Case]:
    fallback_rotation = load_config_rotation(config_path)
    cases = []
    for path in sorted(snapshot_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "image" not in data or "occupied" not in data:
            continue
        image = cv2.imread(str(path.parent / data["image"]))
        if image is None:
            continue
        corners = (
            np.array(data["corners_tl_tr_br_bl"], dtype=np.float32)
            if data.get("corners_tl_tr_br_bl") is not None
            else None
        )
        cases.append(
            Case(
                path=path,
                image=image,
                corners=corners,
                size=int(data.get("board_size", 19)),
                rotation=int(data.get("camera_to_robot_rotation_degrees", fallback_rotation)),
                overlay_fisheye_k=float(data.get("overlay_fisheye_k", 0.0)),
                expected={str(coord): str(color) for coord, color in data["occupied"].items()},
                summary=data.get("summary", {}),
            )
        )
    return cases


def evaluate(cases: list[Case], params: dict[str, float]) -> Score:
    failures: list[dict[str, Any]] = []
    missing_count = 0
    extra_count = 0
    wrong_count = 0
    passed = 0
    for case in cases:
        raw_state = board_state_from_image(
            image=case.image,
            corners=case.corners,
            size=case.size,
            overlay_fisheye_k=case.overlay_fisheye_k,
            **params,
        )
        state = transform_board_state(raw_state, case.rotation)
        comparison = compare_occupied(case.expected, state.occupied)
        missing_count += len(comparison["missing"])
        extra_count += len(comparison["extra"])
        wrong_count += len(comparison["wrong_color"])
        if comparison["ok"]:
            passed += 1
        else:
            failures.append(
                {
                    "file": case.path.name,
                    "expected": case.summary,
                    "detected": state.summary,
                    **comparison,
                }
            )
    total_errors = missing_count + extra_count + wrong_count
    return Score(
        passed=passed,
        total_errors=total_errors,
        missing=missing_count,
        extra=extra_count,
        wrong_color=wrong_count,
        failures=failures,
    )


def better(candidate: Score, current: Score) -> bool:
    return candidate.rank() > current.rank()


def rounded_params(params: dict[str, float]) -> dict[str, float]:
    rounded = {}
    for key, value in params.items():
        rounded[key] = round(value, 3 if key.endswith("_ratio") or "circularity" in key else 1)
    return rounded


def format_score(score: Score, total_cases: int) -> str:
    return (
        f"{score.passed}/{total_cases} passed, errors={score.total_errors} "
        f"(missing={score.missing}, extra={score.extra}, wrong={score.wrong_color})"
    )


def bracket_candidates(low: float, high: float) -> list[float]:
    mid = (low + high) / 2
    return [low, mid, high]


def tune(
    cases: list[Case],
    start: dict[str, float],
    ranges: dict[str, tuple[float, float]],
    passes: int,
) -> tuple[dict[str, float], Score]:
    cache: dict[tuple[tuple[str, float], ...], Score] = {}

    def cached_evaluate(params: dict[str, float]) -> Score:
        key = tuple((name, round(params[name], 5)) for name in TUNING_KEYS)
        if key not in cache:
            cache[key] = evaluate(cases, params)
        return cache[key]

    best_params = start.copy()
    best_score = cached_evaluate(best_params)
    print(f"baseline {format_score(best_score, len(cases))} params={rounded_params(best_params)}", flush=True)

    brackets = ranges.copy()
    for pass_index in range(1, passes + 1):
        improved_this_pass = False
        print(f"\nPASS {pass_index}", flush=True)
        for key in TUNING_KEYS:
            low, high = brackets[key]
            probes = sorted(set(bracket_candidates(low, high) + [best_params[key]]))
            local_best_value = best_params[key]
            local_best_score = best_score
            local_best_params = best_params

            for value in probes:
                params = best_params.copy()
                params[key] = float(value)
                if params["stone_min_radius_ratio"] > params["stone_max_radius_ratio"]:
                    continue
                score = cached_evaluate(params)
                if better(score, local_best_score):
                    local_best_value = float(value)
                    local_best_score = score
                    local_best_params = params

            midpoint = (low + high) / 2
            half_width = (high - low) / 4
            next_low = max(ranges[key][0], local_best_value - half_width)
            next_high = min(ranges[key][1], local_best_value + half_width)
            if next_low == next_high:
                next_low, next_high = low, high
            brackets[key] = (next_low, next_high)

            if better(local_best_score, best_score):
                improved_this_pass = True
                best_params = local_best_params
                best_score = local_best_score
                print(
                    f"  improved {key}={local_best_value:.4g}: "
                    f"{format_score(best_score, len(cases))}",
                    flush=True,
                )
            else:
                direction = "low" if local_best_value < midpoint else "high" if local_best_value > midpoint else "mid"
                print(f"  {key}: keep {best_params[key]:.4g}, shrink around {direction}", flush=True)

        if not improved_this_pass:
            print("  no score improvement this pass; continuing with narrower brackets", flush=True)

    return best_params, best_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--start-json", type=Path, help="Start from a previous tuning_sweep.json result.")
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_cases(args.snapshot_dir, args.config)
    if not cases:
        raise FileNotFoundError(f"No annotation cases found in {args.snapshot_dir}")

    start = load_config_params(args.config)
    if args.start_json is not None:
        previous = json.loads(args.start_json.read_text())
        start.update({key: float(value) for key, value in previous.get("params", {}).items() if key in TUNING_KEYS})
    ranges = {
        "sample_radius_ratio": (
            max(0.18, start["sample_radius_ratio"] - 0.08),
            min(0.44, start["sample_radius_ratio"] + 0.08),
        ),
        "black_l_threshold": (
            max(55.0, start["black_l_threshold"] - 20.0),
            min(115.0, start["black_l_threshold"] + 20.0),
        ),
        "white_l_threshold": (
            max(140.0, start["white_l_threshold"] - 20.0),
            min(205.0, start["white_l_threshold"] + 20.0),
        ),
        "white_s_threshold": (
            max(35.0, start["white_s_threshold"] - 25.0),
            min(115.0, start["white_s_threshold"] + 25.0),
        ),
        "stone_min_radius_ratio": (
            max(0.1, start["stone_min_radius_ratio"] - 0.18),
            min(0.6, start["stone_min_radius_ratio"] + 0.18),
        ),
        "stone_max_radius_ratio": (
            max(0.32, start["stone_max_radius_ratio"] - 0.12),
            min(0.75, start["stone_max_radius_ratio"] + 0.12),
        ),
        "stone_min_circularity": (
            max(0.25, start["stone_min_circularity"] - 0.2),
            min(0.95, start["stone_min_circularity"] + 0.2),
        ),
        "stone_max_snap_distance_ratio": (
            max(0.2, start["stone_max_snap_distance_ratio"] - 0.18),
            min(0.85, start["stone_max_snap_distance_ratio"] + 0.18),
        ),
        "black_grid_min_edge_score": (
            max(0.12, start["black_grid_min_edge_score"] - 0.2),
            min(0.78, start["black_grid_min_edge_score"] + 0.2),
        ),
    }
    params, score = tune(cases, start, ranges, args.passes)

    report = {
        "passed": score.passed,
        "failed": len(cases) - score.passed,
        "total": len(cases),
        "errors": score.total_errors,
        "missing": score.missing,
        "extra": score.extra,
        "wrong_color": score.wrong_color,
        "params": rounded_params(params),
        "failures": score.failures,
    }
    print("\nFINAL")
    print(json.dumps(report, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
