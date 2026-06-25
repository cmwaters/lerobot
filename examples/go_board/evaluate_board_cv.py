#!/usr/bin/env python
"""Evaluate Go-board CV against annotated snapshot JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_board_state import (
    GO_COLUMNS,
    board_state_from_image,
    draw_debug_overlay,
    transform_board_state,
    transform_coord,
)


def load_annotation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "image" not in data:
        raise ValueError(f"{path} is missing image.")
    if "occupied" not in data:
        raise ValueError(f"{path} is missing occupied ground truth.")
    return data


def load_config_rotation(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    data = json.loads(path.read_text())
    return int(data.get("board", {}).get("camera_to_robot_rotation_degrees", 0))


def load_config_tuning(path: Path | None) -> dict[str, float]:
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
    if path is None or not path.is_file():
        return defaults
    data = json.loads(path.read_text())
    board = data.get("board", {})
    return {key: float(board.get(key, value)) for key, value in defaults.items()}


def compare_occupied(expected: dict[str, str], detected: dict[str, str]) -> dict[str, Any]:
    expected_coords = set(expected)
    detected_coords = set(detected)
    missing = {coord: expected[coord] for coord in sorted(expected_coords - detected_coords)}
    extra = {coord: detected[coord] for coord in sorted(detected_coords - expected_coords)}
    wrong_color = {
        coord: {"expected": expected[coord], "detected": detected[coord]}
        for coord in sorted(expected_coords & detected_coords)
        if expected[coord] != detected[coord]
    }
    return {
        "ok": not missing and not extra and not wrong_color,
        "missing": missing,
        "extra": extra,
        "wrong_color": wrong_color,
    }


def radial_curve_point(point: np.ndarray, image_shape: tuple[int, ...], overlay_fisheye_k: float) -> np.ndarray:
    k = float(overlay_fisheye_k)
    if abs(k) < 1e-9:
        return point.astype(np.float32)

    x = float(point[0])
    y = float(point[1])
    height, width = image_shape[:2]
    cx = width / 2.0
    cy = height / 2.0
    scale = max(width, height) / 2.0
    dx = (float(x) - cx) / scale
    dy = (float(y) - cy) / scale
    factor = 1.0 + k * (dx * dx + dy * dy)
    curved_x = cx + dx * factor * scale
    curved_y = cy + dy * factor * scale
    return np.array([curved_x, curved_y], dtype=np.float32)


def bilinear(corners: np.ndarray, row_t: float, col_t: float) -> np.ndarray:
    top = corners[0] * (1.0 - col_t) + corners[1] * col_t
    bottom = corners[3] * (1.0 - col_t) + corners[2] * col_t
    return top * (1.0 - row_t) + bottom * row_t


def curved_grid_points(
    image_shape: tuple[int, ...],
    corners: np.ndarray,
    board_size: int,
    overlay_fisheye_k: float,
) -> list[list[tuple[int, int]]]:
    """Return camera-space grid points with curved interior and fixed corners."""
    corners = corners.astype(np.float32)
    corner_offsets = np.array(
        [radial_curve_point(point, image_shape, overlay_fisheye_k) - point for point in corners],
        dtype=np.float32,
    )
    points: list[list[tuple[int, int]]] = []
    for row in range(board_size):
        row_t = row / (board_size - 1)
        point_row = []
        for col in range(board_size):
            col_t = col / (board_size - 1)
            base = bilinear(corners, row_t, col_t)
            radial_offset = radial_curve_point(base, image_shape, overlay_fisheye_k) - base
            anchored_offset = radial_offset - bilinear(corner_offsets, row_t, col_t)
            point = base + anchored_offset
            point_row.append(tuple(np.round(point).astype(int)))
        points.append(point_row)
    return points


def image_point_for_coord(
    coord: str,
    image: np.ndarray,
    corners: np.ndarray,
    board_pixels: int = 1000,
    board_size: int = 19,
    camera_to_robot_rotation_degrees: int = 0,
    overlay_fisheye_k: float = 0.0,
) -> tuple[int, int]:
    _ = board_pixels
    camera_coord = transform_coord(coord, board_size, -camera_to_robot_rotation_degrees)
    col = GO_COLUMNS.index(camera_coord[0])
    row = int(camera_coord[1:]) - 1
    return curved_grid_points(image.shape, corners, board_size, overlay_fisheye_k)[row][col]


def draw_comparison_overlay(
    image: np.ndarray,
    corners: np.ndarray,
    expected: dict[str, str],
    detected: dict[str, str],
    output_path: Path,
    board_size: int = 19,
    camera_to_robot_rotation_degrees: int = 0,
    overlay_fisheye_k: float = 0.0,
) -> None:
    overlay = image.copy()
    points = curved_grid_points(image.shape, corners, board_size, overlay_fisheye_k)
    for row in range(board_size):
        cv2.polylines(overlay, [np.array(points[row], dtype=np.int32)], False, (0, 220, 255), 2)
    for col in range(board_size):
        cv2.polylines(overlay, [np.array([points[row][col] for row in range(board_size)], dtype=np.int32)], False, (0, 220, 255), 2)
    cv2.polylines(overlay, [corners.astype(np.int32)], isClosed=True, color=(0, 180, 255), thickness=2)
    for coord, color in expected.items():
        point = image_point_for_coord(
            coord,
            image,
            corners,
            board_size=board_size,
            camera_to_robot_rotation_degrees=camera_to_robot_rotation_degrees,
            overlay_fisheye_k=overlay_fisheye_k,
        )
        cv2.circle(overlay, point, 16, (0, 220, 0), 3)
        cv2.putText(
            overlay,
            f"E {coord} {color[0]}",
            (point[0] + 10, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 0),
            1,
            cv2.LINE_AA,
        )
    for coord, color in detected.items():
        point = image_point_for_coord(
            coord,
            image,
            corners,
            board_size=board_size,
            camera_to_robot_rotation_degrees=camera_to_robot_rotation_degrees,
            overlay_fisheye_k=overlay_fisheye_k,
        )
        cv2.circle(overlay, point, 10, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"D {coord} {color[0]}",
            (point[0] + 10, point[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def evaluate_one(
    annotation_path: Path,
    debug_dir: Path | None,
    comparison_dir: Path | None,
    fallback_rotation_degrees: int = 0,
    tuning: dict[str, float] | None = None,
) -> dict[str, Any]:
    annotation = load_annotation(annotation_path)
    image_path = annotation_path.parent / str(annotation["image"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image for {annotation_path}: {image_path}")

    corners = annotation.get("corners_tl_tr_br_bl")
    corners_array = np.array(corners, dtype=np.float32) if corners is not None else None
    board_size = int(annotation.get("board_size", 19))
    rotation = int(annotation.get("camera_to_robot_rotation_degrees", fallback_rotation_degrees))
    overlay_fisheye_k = float(annotation.get("overlay_fisheye_k", 0.0))
    raw_state = board_state_from_image(
        image=image,
        corners=corners_array,
        size=board_size,
        overlay_fisheye_k=overlay_fisheye_k,
        **(tuning or {}),
    )
    state = transform_board_state(raw_state, rotation)
    expected = {str(coord): str(color) for coord, color in annotation["occupied"].items()}
    comparison = compare_occupied(expected, state.occupied)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        draw_debug_overlay(
            image=image,
            stones=raw_state.stones,
            corners=np.array(raw_state.corners_tl_tr_br_bl, dtype=np.float32),
            size=raw_state.board_size,
            board_pixels=1000,
            output_path=debug_dir / f"{annotation_path.stem}_detected.jpg",
        )
    if comparison_dir is not None:
        draw_comparison_overlay(
            image=image,
            corners=np.array(state.corners_tl_tr_br_bl, dtype=np.float32),
            expected=expected,
            detected=state.occupied,
            output_path=comparison_dir / f"{annotation_path.stem}_expected_detected.jpg",
            board_size=board_size,
            camera_to_robot_rotation_degrees=rotation,
            overlay_fisheye_k=overlay_fisheye_k,
        )

    return {
        "file": annotation_path.name,
        "image": image_path.name,
        "ok": comparison["ok"],
        "expected_summary": annotation.get("summary", {}),
        "detected_summary": state.summary,
        "expected": expected,
        "detected": state.occupied,
        "camera_to_robot_rotation_degrees": rotation,
        "overlay_fisheye_k": overlay_fisheye_k,
        "missing": comparison["missing"],
        "extra": comparison["extra"],
        "wrong_color": comparison["wrong_color"],
        "stones": [asdict(stone) for stone in state.stones],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot_dir",
        type=Path,
        nargs="?",
        default=Path("examples/go_board/cv_snapshots"),
        help="Directory containing paired annotation JSON and image files.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("examples/go_board/cv_snapshots/debug"),
        help="Directory for detection overlay images. Use --no-debug to disable.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("examples/go_board/cv_snapshots/expected_vs_detected"),
        help="Directory for expected-vs-detected overlay images. Use --no-debug to disable.",
    )
    parser.add_argument("--no-debug", action="store_true", help="Do not write detection overlays.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/go_board/dashboard_config.json"),
        help="Dashboard config used for camera-to-robot rotation fallback on older annotations.",
    )
    parser.add_argument(
        "--camera-to-robot-rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
        help="Override the fallback rotation for annotations that do not store orientation metadata.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to write the full evaluation report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotation_paths = []
    for path in sorted(args.snapshot_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "image" in data and "occupied" in data:
            annotation_paths.append(path)
    if not annotation_paths:
        raise FileNotFoundError(f"No annotation JSON files found in {args.snapshot_dir}")

    debug_dir = None if args.no_debug else args.debug_dir
    comparison_dir = None if args.no_debug else args.comparison_dir
    fallback_rotation = (
        args.camera_to_robot_rotation_degrees
        if args.camera_to_robot_rotation_degrees is not None
        else load_config_rotation(args.config)
    )
    tuning = load_config_tuning(args.config)
    results = [
        evaluate_one(
            path,
            debug_dir=debug_dir,
            comparison_dir=comparison_dir,
            fallback_rotation_degrees=fallback_rotation,
            tuning=tuning,
        )
        for path in annotation_paths
    ]
    passed = sum(item["ok"] for item in results)
    failed = len(results) - passed
    report = {"passed": passed, "failed": failed, "total": len(results), "results": results}

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Go CV evaluation: {passed}/{len(results)} passed")
    for item in results:
        status = "PASS" if item["ok"] else "FAIL"
        print(
            f"{status} {item['file']}: "
            f"expected={item['expected_summary']} detected={item['detected_summary']}"
        )
        if not item["ok"]:
            if item["missing"]:
                print(f"  missing: {item['missing']}")
            if item["extra"]:
                print(f"  extra: {item['extra']}")
            if item["wrong_color"]:
                print(f"  wrong_color: {item['wrong_color']}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
