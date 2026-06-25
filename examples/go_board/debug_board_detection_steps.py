#!/usr/bin/env python
"""Generate step-by-step Go-board CV debug images for one annotation."""

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
    BoardState,
    board_state_from_image,
    contour_stone_candidates,
    coord_for,
    curved_grid_points_from_corners,
    intersection_stone_candidates,
    transform_board_state,
    warp_board_with_margin,
    warp_grid_points,
)
from evaluate_board_cv import compare_occupied, draw_comparison_overlay, load_config_rotation, load_config_tuning


def draw_grid(image: np.ndarray, offset: float, spacing: float, size: int, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image.copy()
    for idx in range(size):
        p = int(round(offset + idx * spacing))
        cv2.line(overlay, (int(round(offset)), p), (int(round(offset + (size - 1) * spacing)), p), color, 1)
        cv2.line(overlay, (p, int(round(offset))), (p, int(round(offset + (size - 1) * spacing))), color, 1)
    return overlay


def draw_detector_grid_points(
    image: np.ndarray,
    grid_points: list[list[tuple[float, float]]],
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    for point_row in grid_points:
        cv2.polylines(overlay, [np.array(point_row, dtype=np.int32)], False, color, 1)
    for col in range(len(grid_points[0])):
        cv2.polylines(overlay, [np.array([row[col] for row in grid_points], dtype=np.int32)], False, color, 1)
    return overlay


def draw_stones(
    image: np.ndarray,
    stones: list[Any],
    to_board: np.ndarray,
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    cv2.putText(overlay, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    for stone in stones:
        point = cv2.perspectiveTransform(
            np.array([[[float(stone.image_xy[0]), float(stone.image_xy[1])]]], dtype=np.float32),
            to_board,
        )[0, 0]
        x, y = tuple(np.round(point).astype(int))
        fill = (20, 20, 20) if stone.color == "black" else (245, 245, 245)
        cv2.circle(overlay, (x, y), 19, color, 3)
        cv2.circle(overlay, (x, y), 13, fill, -1)
        cv2.putText(
            overlay,
            f"{stone.coord} {stone.color[0]} {stone.confidence:.2f}",
            (x + 18, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def draw_expected_on_warped(
    image: np.ndarray,
    expected: dict[str, str],
    grid_points: list[list[tuple[float, float]]],
    size: int,
    rotation: int,
) -> np.ndarray:
    overlay = image.copy()
    columns = GO_COLUMNS if size == 19 else "".join(chr(ord("A") + i) for i in range(size))
    for coord, stone_color in expected.items():
        robot_col = columns.index(coord[0])
        robot_row = int(coord[1:]) - 1
        camera_state = BoardState(
            board_size=size,
            corners_tl_tr_br_bl=[],
            stones=[],
            occupied={coord: stone_color},
            summary={},
        )
        # Use the inverse of transform_board_state's coordinate rule without
        # constructing a synthetic Stone object.
        from detect_board_state import inverse_transform_row_col

        row, col = inverse_transform_row_col(robot_row, robot_col, size, rotation)
        x, y = tuple(np.round(grid_points[row][col]).astype(int))
        outline = (0, 220, 0)
        cv2.circle(overlay, (x, y), 20, outline, 3)
        cv2.putText(
            overlay,
            f"E {coord} {stone_color[0]}",
            (x + 18, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            outline,
            1,
            cv2.LINE_AA,
        )
        _ = camera_state
    return overlay


def write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def load_annotation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "image" not in data or "occupied" not in data:
        raise ValueError(f"{path} is not an annotation JSON with image and occupied fields.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "annotation",
        type=Path,
        nargs="?",
        default=Path("examples/go_board/cv_snapshots/20260618_184540_10_bowl_and_arm_side.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("examples/go_board/dashboard_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/go_board/cv_snapshots/debug_steps"))
    args = parser.parse_args()

    annotation = load_annotation(args.annotation)
    image_path = args.annotation.parent / annotation["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    tuning = load_config_tuning(args.config)
    size = int(annotation.get("board_size", 19))
    board_pixels = 1000
    rotation = int(annotation.get("camera_to_robot_rotation_degrees", load_config_rotation(args.config)))
    corners = np.array(annotation["corners_tl_tr_br_bl"], dtype=np.float32)
    expected = {str(coord): str(color) for coord, color in annotation["occupied"].items()}

    spacing = (board_pixels - 1) / (size - 1)
    grid_margin = int(round(spacing * max(tuning["stone_max_radius_ratio"], 0.55)))
    warped, to_board, to_image = warp_board_with_margin(image, corners, board_pixels, grid_margin)
    grid_offset = float(grid_margin)
    sample_radius = spacing * tuning["sample_radius_ratio"]
    overlay_fisheye_k = float(annotation.get("overlay_fisheye_k", 0.0))
    camera_grid_points = curved_grid_points_from_corners(image.shape, corners, size, overlay_fisheye_k)
    detector_grid_points = warp_grid_points(camera_grid_points, to_board)

    stem = args.annotation.stem
    output_dir = args.output_dir / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    original = image.copy()
    cv2.polylines(original, [corners.astype(np.int32)], isClosed=True, color=(0, 180, 255), thickness=3)
    cv2.putText(original, "1 original frame + saved grid corners", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)
    write(output_dir / "01_original_with_corners.jpg", original)

    draw_comparison_overlay(
        image=image,
        corners=corners,
        expected=expected,
        detected={},
        output_path=output_dir / "02_camera_curved_grid_expected.jpg",
        board_size=size,
        camera_to_robot_rotation_degrees=rotation,
        overlay_fisheye_k=overlay_fisheye_k,
    )

    warped_grid = draw_detector_grid_points(warped, detector_grid_points, (0, 220, 255))
    warped_grid = draw_expected_on_warped(warped_grid, expected, detector_grid_points, size, rotation)
    cv2.putText(warped_grid, "3 detector-space warped board + grid + expected", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    write(output_dir / "03_detector_warped_grid_expected.jpg", warped_grid)

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    l_channel = lab[:, :, 0]
    s_channel = hsv[:, :, 1]
    h_channel = hsv[:, :, 0]
    write(output_dir / "04_lightness_channel.jpg", l_channel)
    write(output_dir / "05_saturation_channel.jpg", s_channel)

    black_mask_raw = (l_channel < tuning["black_l_threshold"] + 18).astype(np.uint8) * 255
    white_mask_raw = ((l_channel > tuning["white_l_threshold"] - 8) & (s_channel < tuning["white_s_threshold"] + 18)).astype(np.uint8) * 255
    blue_mask = ((h_channel > 90) & (h_channel < 135) & (s_channel > 80)).astype(np.uint8) * 255
    black_mask_raw = cv2.bitwise_and(black_mask_raw, cv2.bitwise_not(blue_mask))
    write(output_dir / "06_black_mask_raw.jpg", black_mask_raw)
    write(output_dir / "07_white_mask_raw.jpg", white_mask_raw)
    write(output_dir / "08_blue_rejection_mask.jpg", blue_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    black_mask_clean = cv2.morphologyEx(black_mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    black_mask_clean = cv2.morphologyEx(black_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    white_mask_clean = cv2.morphologyEx(white_mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    white_mask_clean = cv2.morphologyEx(white_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    write(output_dir / "09_black_mask_cleaned.jpg", black_mask_clean)
    write(output_dir / "10_white_mask_cleaned.jpg", white_mask_clean)

    contour_stones = contour_stone_candidates(
        warped=warped,
        spacing=spacing,
        radius=sample_radius,
        grid_offset=grid_offset,
        black_l_threshold=tuning["black_l_threshold"],
        white_l_threshold=tuning["white_l_threshold"],
        white_s_threshold=tuning["white_s_threshold"],
        skip_i=True,
        size=size,
        to_image=to_image,
        min_radius_ratio=tuning["stone_min_radius_ratio"],
        max_radius_ratio=tuning["stone_max_radius_ratio"],
        min_circularity=tuning["stone_min_circularity"],
        max_snap_distance_ratio=tuning["stone_max_snap_distance_ratio"],
        grid_points=detector_grid_points,
    )
    contour_overlay = draw_stones(warped_grid, contour_stones, to_board, "11 accepted contour/blob candidates", (255, 0, 180))
    write(output_dir / "11_contour_candidates.jpg", contour_overlay)

    intersection_stones = intersection_stone_candidates(
        warped=warped,
        spacing=spacing,
        radius=sample_radius,
        grid_offset=grid_offset,
        black_l_threshold=tuning["black_l_threshold"],
        white_l_threshold=tuning["white_l_threshold"],
        white_s_threshold=tuning["white_s_threshold"],
        skip_i=True,
        size=size,
        to_image=to_image,
        min_radius_ratio=tuning["stone_min_radius_ratio"],
        max_radius_ratio=tuning["stone_max_radius_ratio"],
        min_circularity=tuning["stone_min_circularity"],
        black_grid_min_edge_score=tuning.get("black_grid_min_edge_score", 0.18),
        grid_points=detector_grid_points,
    )
    intersection_overlay = draw_stones(warped_grid, intersection_stones, to_board, "12 accepted intersection-local candidates", (255, 120, 0))
    write(output_dir / "12_intersection_candidates.jpg", intersection_overlay)

    raw_state = board_state_from_image(
        image=image,
        corners=corners,
        size=size,
        board_pixels=board_pixels,
        overlay_fisheye_k=overlay_fisheye_k,
        **tuning,
    )
    state = transform_board_state(raw_state, rotation)
    comparison = compare_occupied(expected, state.occupied)

    final_raw_overlay = draw_stones(warped_grid, raw_state.stones, to_board, "13 final raw camera-coordinate detections", (0, 0, 255))
    write(output_dir / "13_final_raw_detections.jpg", final_raw_overlay)

    draw_comparison_overlay(
        image=image,
        corners=np.array(state.corners_tl_tr_br_bl, dtype=np.float32),
        expected=expected,
        detected=state.occupied,
        output_path=output_dir / "14_expected_vs_detected_robot_coords.jpg",
        board_size=size,
        camera_to_robot_rotation_degrees=rotation,
        overlay_fisheye_k=overlay_fisheye_k,
    )

    report = {
        "annotation": args.annotation.name,
        "image": image_path.name,
        "tuning": tuning,
        "passed": comparison["ok"],
        "expected": expected,
        "detected": state.occupied,
        "missing": comparison["missing"],
        "extra": comparison["extra"],
        "wrong_color": comparison["wrong_color"],
        "contour_candidates": [asdict(stone) for stone in transform_board_state(
            BoardState(size, raw_state.corners_tl_tr_br_bl, contour_stones, {stone.coord: stone.color for stone in contour_stones}, {}),
            rotation,
        ).stones],
        "intersection_candidates": [asdict(stone) for stone in transform_board_state(
            BoardState(size, raw_state.corners_tl_tr_br_bl, intersection_stones, {stone.coord: stone.color for stone in intersection_stones}, {}),
            rotation,
        ).stones],
        "final_stones": [asdict(stone) for stone in state.stones],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote debug steps to {output_dir}")
    print(json.dumps({key: report[key] for key in ("passed", "missing", "extra", "wrong_color")}, indent=2))


if __name__ == "__main__":
    main()
