#!/usr/bin/env python
"""Detect black and white Go stones from an overhead board snapshot.

The script is intentionally calibration-friendly: pass the four board corners
for best results, or let it try a simple automatic board-corner detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"


@dataclass
class Stone:
    coord: str
    row: int
    col: int
    color: str
    confidence: float
    board_xy: tuple[float, float]
    image_xy: tuple[float, float]


@dataclass
class BoardState:
    board_size: int
    corners_tl_tr_br_bl: list[list[float]]
    stones: list[Stone]
    occupied: dict[str, str]
    summary: dict[str, int]


@dataclass
class BoardDelta:
    added: list[Stone]
    removed: list[Stone]
    changed: list[dict[str, object]]
    before_summary: dict[str, int]
    after_summary: dict[str, int]


def parse_corners(value: str) -> np.ndarray:
    """Parse corners as 'x1,y1 x2,y2 x3,y3 x4,y4' in TL,TR,BR,BL order."""
    points: list[list[float]] = []
    for pair in value.replace(";", " ").split():
        parts = pair.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "Corners must look like 'x1,y1 x2,y2 x3,y3 x4,y4' in TL,TR,BR,BL order."
            )
        points.append([float(parts[0]), float(parts[1])])

    if len(points) != 4:
        raise argparse.ArgumentTypeError("Exactly four board corners are required.")

    return np.array(points, dtype=np.float32)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return points ordered top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def auto_detect_board_corners(image: np.ndarray) -> np.ndarray:
    """Find a likely board quadrilateral.

    This works best when the full board border is visible and the background
    outside the board is not too similar to the board surface.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * 0.12
    candidates: list[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect board corners. Pass --corners 'x1,y1 x2,y2 x3,y3 x4,y4'."
        )

    best = max(candidates, key=cv2.contourArea)
    return order_corners(best)


def warp_board(
    image: np.ndarray,
    corners: np.ndarray,
    board_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst = np.array(
        [[0, 0], [board_pixels - 1, 0], [board_pixels - 1, board_pixels - 1], [0, board_pixels - 1]],
        dtype=np.float32,
    )
    to_board = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    to_image = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    warped = cv2.warpPerspective(image, to_board, (board_pixels, board_pixels))
    return warped, to_board, to_image


def warp_board_with_margin(
    image: np.ndarray,
    corners: np.ndarray,
    board_pixels: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst = np.array(
        [
            [margin, margin],
            [margin + board_pixels - 1, margin],
            [margin + board_pixels - 1, margin + board_pixels - 1],
            [margin, margin + board_pixels - 1],
        ],
        dtype=np.float32,
    )
    to_board = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    to_image = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    output_size = board_pixels + 2 * margin
    warped = cv2.warpPerspective(image, to_board, (output_size, output_size))
    return warped, to_board, to_image


def coord_for(row: int, col: int, size: int, skip_i: bool) -> str:
    columns = GO_COLUMNS if skip_i and size == 19 else "".join(chr(ord("A") + i) for i in range(size))
    return f"{columns[col]}{row + 1}"


def transform_row_col(row: int, col: int, size: int, rotation_degrees: int) -> tuple[int, int]:
    """Rotate a board coordinate clockwise in 90-degree steps."""
    rotation = rotation_degrees % 360
    if rotation == 0:
        return row, col
    if rotation == 90:
        return col, size - 1 - row
    if rotation == 180:
        return size - 1 - row, size - 1 - col
    if rotation == 270:
        return size - 1 - col, row
    raise ValueError("Board rotation must be one of 0, 90, 180, or 270 degrees.")


def inverse_transform_row_col(row: int, col: int, size: int, rotation_degrees: int) -> tuple[int, int]:
    return transform_row_col(row, col, size, -rotation_degrees)


def transform_coord(coord: str, size: int, rotation_degrees: int, skip_i: bool = True) -> str:
    columns = GO_COLUMNS if skip_i and size == 19 else "".join(chr(ord("A") + i) for i in range(size))
    col = columns.index(coord[0])
    row = int(coord[1:]) - 1
    transformed_row, transformed_col = transform_row_col(row, col, size, rotation_degrees)
    return coord_for(transformed_row, transformed_col, size, skip_i)


def transform_board_state(state: BoardState, rotation_degrees: int, skip_i: bool = True) -> BoardState:
    rotation = rotation_degrees % 360
    if rotation == 0:
        return state

    transformed_stones = []
    for stone in state.stones:
        row, col = transform_row_col(stone.row, stone.col, state.board_size, rotation)
        transformed_stones.append(
            Stone(
                coord=coord_for(row, col, state.board_size, skip_i),
                row=row,
                col=col,
                color=stone.color,
                confidence=stone.confidence,
                board_xy=stone.board_xy,
                image_xy=stone.image_xy,
            )
        )

    transformed_stones.sort(key=lambda stone: (stone.row, stone.col))
    occupied = {stone.coord: stone.color for stone in transformed_stones}
    return BoardState(
        board_size=state.board_size,
        corners_tl_tr_br_bl=state.corners_tl_tr_br_bl,
        stones=transformed_stones,
        occupied=occupied,
        summary=state.summary,
    )


def circle_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cx, cy = center
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def radial_curve_point(point: np.ndarray, image_shape: tuple[int, ...], curve_k: float) -> np.ndarray:
    if abs(curve_k) < 1e-9:
        return point.astype(np.float32)

    height, width = image_shape[:2]
    cx = width / 2.0
    cy = height / 2.0
    scale = max(width, height) / 2.0
    dx = (float(point[0]) - cx) / scale
    dy = (float(point[1]) - cy) / scale
    factor = 1.0 + curve_k * (dx * dx + dy * dy)
    return np.array([cx + dx * factor * scale, cy + dy * factor * scale], dtype=np.float32)


def bilinear(corners: np.ndarray, row_t: float, col_t: float) -> np.ndarray:
    top = corners[0] * (1.0 - col_t) + corners[1] * col_t
    bottom = corners[3] * (1.0 - col_t) + corners[2] * col_t
    return top * (1.0 - row_t) + bottom * row_t


def curved_grid_points_from_corners(
    image_shape: tuple[int, ...],
    corners: np.ndarray,
    size: int,
    curve_k: float,
) -> list[list[tuple[float, float]]]:
    """Return camera-space grid points with curved interior and fixed corners."""
    corners = corners.astype(np.float32)
    corner_offsets = np.array([radial_curve_point(point, image_shape, curve_k) - point for point in corners], dtype=np.float32)
    points: list[list[tuple[float, float]]] = []
    for row in range(size):
        row_t = row / (size - 1)
        point_row = []
        for col in range(size):
            col_t = col / (size - 1)
            base = bilinear(corners, row_t, col_t)
            radial_offset = radial_curve_point(base, image_shape, curve_k) - base
            anchored_offset = radial_offset - bilinear(corner_offsets, row_t, col_t)
            point = base + anchored_offset
            point_row.append((float(point[0]), float(point[1])))
        points.append(point_row)
    return points


def warp_grid_points(
    grid_points: list[list[tuple[float, float]]],
    transform: np.ndarray,
) -> list[list[tuple[float, float]]]:
    flat = np.array([[[x, y]] for row in grid_points for x, y in row], dtype=np.float32)
    transformed = cv2.perspectiveTransform(flat, transform).reshape(-1, 2)
    warped: list[list[tuple[float, float]]] = []
    index = 0
    for row in grid_points:
        warped_row = []
        for _point in row:
            x, y = transformed[index]
            warped_row.append((float(x), float(y)))
            index += 1
        warped.append(warped_row)
    return warped


def nearest_grid_intersection(
    x: float,
    y: float,
    grid_points: list[list[tuple[float, float]]],
) -> tuple[int, int, float, float, float]:
    best_row = 0
    best_col = 0
    best_x = 0.0
    best_y = 0.0
    best_distance = float("inf")
    for row, point_row in enumerate(grid_points):
        for col, (grid_x, grid_y) in enumerate(point_row):
            distance = float(np.hypot(x - grid_x, y - grid_y))
            if distance < best_distance:
                best_row = row
                best_col = col
                best_x = grid_x
                best_y = grid_y
                best_distance = distance
    return best_row, best_col, best_x, best_y, best_distance


def classify_patch(
    warped: np.ndarray,
    center: tuple[float, float],
    radius: float,
    black_l_threshold: float,
    white_l_threshold: float,
    white_s_threshold: float,
) -> tuple[str | None, float]:
    """Classify a local intersection patch as black, white, or empty."""
    x, y = center
    pad = int(radius * 1.35)
    x0 = max(int(x) - pad, 0)
    y0 = max(int(y) - pad, 0)
    x1 = min(int(x) + pad + 1, warped.shape[1])
    y1 = min(int(y) + pad + 1, warped.shape[0])
    patch = warped[y0:y1, x0:x1]

    if patch.size == 0:
        return None, 0.0

    local_center = (x - x0, y - y0)
    mask = circle_mask(patch.shape[:2], local_center, radius)
    core_mask = circle_mask(patch.shape[:2], local_center, radius * 0.55)
    outer_mask = circle_mask(patch.shape[:2], local_center, radius * 1.65)
    annulus_mask = outer_mask & ~mask
    if mask.sum() < 16:
        return None, 0.0

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    l_values = lab[:, :, 0][mask].astype(np.float32)
    h_values = hsv[:, :, 0][mask].astype(np.float32)
    s_values = hsv[:, :, 1][mask].astype(np.float32)
    core_l_values = lab[:, :, 0][core_mask].astype(np.float32)
    annulus_l_values = lab[:, :, 0][annulus_mask].astype(np.float32)
    if core_l_values.size == 0 or annulus_l_values.size == 0:
        return None, 0.0

    median_l = float(np.median(l_values))
    median_s = float(np.median(s_values))
    core_median_l = float(np.median(core_l_values))
    annulus_median_l = float(np.median(annulus_l_values))
    blue_fraction = float(np.mean((h_values > 90) & (h_values < 135) & (s_values > 80)))
    dark_fraction = float(np.mean(l_values < black_l_threshold))
    bright_low_sat_fraction = float(np.mean((l_values > white_l_threshold) & (s_values < white_s_threshold)))

    if blue_fraction > 0.08 and dark_fraction < 0.55:
        return None, 0.0

    black_contrast = (annulus_median_l - core_median_l) / 45.0
    white_contrast = (core_median_l - annulus_median_l) / 45.0
    black_score = max(dark_fraction, black_contrast)
    white_score = max(
        bright_low_sat_fraction,
        white_contrast,
    )

    if (
        dark_fraction > 0.55
        and median_l < black_l_threshold + 20
        and (black_score > white_score or dark_fraction > 0.85)
    ):
        return "black", float(np.clip(black_score, 0.0, 1.0))
    if (
        (bright_low_sat_fraction > 0.36 or white_contrast > 0.65)
        and white_contrast > 0.15
        and median_s < white_s_threshold + 15
        and white_score > black_score
    ):
        return "white", float(np.clip(white_score, 0.0, 1.0))
    return None, 0.0


def circular_edge_score(
    warped: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> float:
    """Score whether edges around a fixed grid point look like a stone rim."""
    x, y = center
    pad = int(round(radius * 1.7))
    x0 = max(int(round(x)) - pad, 0)
    y0 = max(int(round(y)) - pad, 0)
    x1 = min(int(round(x)) + pad + 1, warped.shape[1])
    y1 = min(int(round(y)) + pad + 1, warped.shape[0])
    patch = warped[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    local_center = (x - x0, y - y0)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 130)

    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    distance = np.sqrt((xx - local_center[0]) ** 2 + (yy - local_center[1]) ** 2)
    ring = (distance >= radius * 0.7) & (distance <= radius * 1.25)
    if int(ring.sum()) < 20:
        return 0.0

    edge_pixels = ring & (edges > 0)
    edge_density = float(edge_pixels.sum() / ring.sum())
    if not edge_pixels.any():
        return 0.0

    ys, xs = np.nonzero(edge_pixels)
    angles = np.arctan2(ys.astype(np.float32) - local_center[1], xs.astype(np.float32) - local_center[0])
    bins = np.floor((angles + np.pi) / (2 * np.pi) * 24).astype(int)
    coverage = len(set(np.clip(bins, 0, 23).tolist())) / 24.0
    density_score = min(edge_density / 0.08, 1.0)
    return float(np.clip(0.6 * coverage + 0.4 * density_score, 0.0, 1.0))


def intersection_stone_candidates(
    warped: np.ndarray,
    spacing: float,
    radius: float,
    grid_offset: float,
    black_l_threshold: float,
    white_l_threshold: float,
    white_s_threshold: float,
    skip_i: bool,
    size: int,
    to_image: np.ndarray,
    min_radius_ratio: float,
    max_radius_ratio: float,
    grid_points: list[list[tuple[float, float]]],
) -> list[Stone]:
    """Classify fixed grid intersections instead of running a global circle detector."""
    stones_by_coord: dict[str, Stone] = {}
    sample_radius = float(np.clip(radius, spacing * min_radius_ratio, spacing * max_radius_ratio))

    for row in range(size):
        for col in range(size):
            x, y = grid_points[row][col]
            color, color_confidence = classify_patch(
                warped,
                (x, y),
                sample_radius,
                black_l_threshold,
                white_l_threshold,
                white_s_threshold,
            )
            if color is None:
                continue

            edge_score = circular_edge_score(warped, (x, y), sample_radius)
            edge_band = row in {0, 1, size - 2, size - 1} or col in {0, 1, size - 2, size - 1}
            min_color_confidence = 0.35 if edge_band else 0.45
            min_edge_score = 0.12 if edge_band else 0.18
            if color_confidence < min_color_confidence:
                continue
            if edge_score < min_edge_score and color_confidence < 0.72:
                continue

            center_image_point = cv2.perspectiveTransform(
                np.array([[[x, y]]], dtype=np.float32),
                to_image,
            )[0, 0]
            coord = coord_for(row, col, size, skip_i)
            confidence = float(np.clip(0.72 * color_confidence + 0.28 * edge_score, 0.0, 1.0))
            stones_by_coord[coord] = Stone(
                coord=coord,
                row=row,
                col=col,
                color=color,
                confidence=round(confidence, 3),
                board_xy=(round(float(col * spacing), 2), round(float(row * spacing), 2)),
                image_xy=(round(float(center_image_point[0]), 2), round(float(center_image_point[1]), 2)),
            )

    return list(stones_by_coord.values())


def contour_stone_candidates(
    warped: np.ndarray,
    spacing: float,
    radius: float,
    grid_offset: float,
    black_l_threshold: float,
    white_l_threshold: float,
    white_s_threshold: float,
    skip_i: bool,
    size: int,
    to_image: np.ndarray,
    min_radius_ratio: float,
    max_radius_ratio: float,
    min_circularity: float,
    max_snap_distance_ratio: float,
    grid_points: list[list[tuple[float, float]]],
) -> list[Stone]:
    """Find round color blobs first, then snap their centers to the nearest grid intersection."""
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    l_channel = lab[:, :, 0]
    s_channel = hsv[:, :, 1]
    h_channel = hsv[:, :, 0]

    black_mask = (l_channel < black_l_threshold + 18).astype(np.uint8) * 255
    white_mask = ((l_channel > white_l_threshold - 8) & (s_channel < white_s_threshold + 18)).astype(np.uint8) * 255
    blue_mask = ((h_channel > 90) & (h_channel < 135) & (s_channel > 80)).astype(np.uint8) * 255
    black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(blue_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = {
        "black": cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1),
        "white": cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1),
    }
    masks = {
        color: cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        for color, mask in masks.items()
    }

    min_radius = spacing * min_radius_ratio
    max_radius = spacing * max_radius_ratio
    min_area = np.pi * min_radius * min_radius * 0.45
    max_area = np.pi * max_radius * max_radius * 1.35
    stones_by_coord: dict[str, Stone] = {}

    for expected_color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < min_circularity:
                continue

            (x, y), detected_radius = cv2.minEnclosingCircle(contour)
            if detected_radius < min_radius or detected_radius > max_radius:
                continue

            fill_ratio = area / (np.pi * detected_radius * detected_radius)
            if fill_ratio < 0.42:
                continue

            row, col, snapped_x, snapped_y, distance_to_intersection = nearest_grid_intersection(x, y, grid_points)
            if distance_to_intersection > spacing * max_snap_distance_ratio:
                continue

            sample_radius = float(np.clip(detected_radius * 0.85, radius * 0.65, radius * 1.4))
            color, color_confidence = classify_patch(
                warped,
                (float(x), float(y)),
                sample_radius,
                black_l_threshold,
                white_l_threshold,
                white_s_threshold,
            )
            if color is None or color != expected_color:
                continue

            board_x = col * spacing
            board_y = row * spacing
            center_image_point = cv2.perspectiveTransform(
                np.array([[[float(x), float(y)]]], dtype=np.float32),
                to_image,
            )[0, 0]
            coord = coord_for(row, col, size, skip_i)
            shape_confidence = np.clip((circularity + fill_ratio) / 2.0, 0.0, 1.0)
            snap_confidence = np.clip(1.0 - distance_to_intersection / (spacing * max_snap_distance_ratio), 0.0, 1.0)
            confidence = float(np.clip(0.55 * color_confidence + 0.3 * shape_confidence + 0.15 * snap_confidence, 0.0, 1.0))
            candidate = Stone(
                coord=coord,
                row=row,
                col=col,
                color=color,
                confidence=round(confidence, 3),
                board_xy=(round(float(board_x), 2), round(float(board_y), 2)),
                image_xy=(round(float(center_image_point[0]), 2), round(float(center_image_point[1]), 2)),
            )
            existing = stones_by_coord.get(coord)
            if existing is None or candidate.confidence > existing.confidence:
                stones_by_coord[coord] = candidate

    return list(stones_by_coord.values())


def detect_stones(
    image: np.ndarray,
    corners: np.ndarray,
    size: int,
    board_pixels: int,
    sample_radius_ratio: float,
    black_l_threshold: float,
    white_l_threshold: float,
    white_s_threshold: float,
    stone_min_radius_ratio: float,
    stone_max_radius_ratio: float,
    stone_min_circularity: float,
    stone_max_snap_distance_ratio: float,
    overlay_fisheye_k: float,
    skip_i: bool,
) -> tuple[list[Stone], np.ndarray, np.ndarray]:
    spacing = (board_pixels - 1) / (size - 1)
    grid_margin = int(round(spacing * max(stone_max_radius_ratio, 0.55)))
    warped, to_board, to_image = warp_board_with_margin(image, corners, board_pixels, grid_margin)
    radius = spacing * sample_radius_ratio
    camera_grid_points = curved_grid_points_from_corners(image.shape, corners, size, overlay_fisheye_k)
    detector_grid_points = warp_grid_points(camera_grid_points, to_board)
    stones_by_coord: dict[str, Stone] = {}

    for stone in contour_stone_candidates(
        warped=warped,
        spacing=spacing,
        radius=radius,
        grid_offset=float(grid_margin),
        black_l_threshold=black_l_threshold,
        white_l_threshold=white_l_threshold,
        white_s_threshold=white_s_threshold,
        skip_i=skip_i,
        size=size,
        to_image=to_image,
        min_radius_ratio=stone_min_radius_ratio,
        max_radius_ratio=stone_max_radius_ratio,
        min_circularity=stone_min_circularity,
        max_snap_distance_ratio=stone_max_snap_distance_ratio,
        grid_points=detector_grid_points,
    ):
        stones_by_coord[stone.coord] = stone

    for stone in intersection_stone_candidates(
        warped=warped,
        spacing=spacing,
        radius=radius,
        grid_offset=float(grid_margin),
        black_l_threshold=black_l_threshold,
        white_l_threshold=white_l_threshold,
        white_s_threshold=white_s_threshold,
        skip_i=skip_i,
        size=size,
        to_image=to_image,
        min_radius_ratio=stone_min_radius_ratio,
        max_radius_ratio=stone_max_radius_ratio,
        grid_points=detector_grid_points,
    ):
        existing = stones_by_coord.get(stone.coord)
        if existing is None or stone.confidence > existing.confidence:
            stones_by_coord[stone.coord] = stone

    stones = sorted(stones_by_coord.values(), key=lambda stone: (stone.row, stone.col))
    return stones, warped, to_image


def board_state_from_image(
    image: np.ndarray,
    corners: np.ndarray | None = None,
    size: int = 19,
    board_pixels: int = 1000,
    sample_radius_ratio: float = 0.34,
    black_l_threshold: float = 80.0,
    white_l_threshold: float = 165.0,
    white_s_threshold: float = 75.0,
    stone_min_radius_ratio: float = 0.2,
    stone_max_radius_ratio: float = 0.48,
    stone_min_circularity: float = 0.45,
    stone_max_snap_distance_ratio: float = 0.52,
    overlay_fisheye_k: float = 0.0,
    skip_i: bool = True,
) -> BoardState:
    """Detect the complete Go board state from an image array."""
    detected_corners = corners if corners is not None else auto_detect_board_corners(image)
    stones, _warped, _to_image = detect_stones(
        image=image,
        corners=detected_corners,
        size=size,
        board_pixels=board_pixels,
        sample_radius_ratio=sample_radius_ratio,
        black_l_threshold=black_l_threshold,
        white_l_threshold=white_l_threshold,
        white_s_threshold=white_s_threshold,
        stone_min_radius_ratio=stone_min_radius_ratio,
        stone_max_radius_ratio=stone_max_radius_ratio,
        stone_min_circularity=stone_min_circularity,
        stone_max_snap_distance_ratio=stone_max_snap_distance_ratio,
        overlay_fisheye_k=overlay_fisheye_k,
        skip_i=skip_i,
    )
    occupied = {stone.coord: stone.color for stone in stones}
    return BoardState(
        board_size=size,
        corners_tl_tr_br_bl=detected_corners.round(2).tolist(),
        stones=stones,
        occupied=occupied,
        summary={
            "black": sum(stone.color == "black" for stone in stones),
            "white": sum(stone.color == "white" for stone in stones),
            "total": len(stones),
        },
    )


def board_state_to_jsonable(state: BoardState) -> dict:
    return {
        "board_size": state.board_size,
        "corners_tl_tr_br_bl": state.corners_tl_tr_br_bl,
        "stones": [asdict(stone) for stone in state.stones],
        "occupied": state.occupied,
        "summary": state.summary,
    }


def board_delta_to_jsonable(delta: BoardDelta) -> dict:
    return {
        "added": [asdict(stone) for stone in delta.added],
        "removed": [asdict(stone) for stone in delta.removed],
        "changed": delta.changed,
        "before_summary": delta.before_summary,
        "after_summary": delta.after_summary,
    }


def delta_between_board_states(before: BoardState, after: BoardState) -> BoardDelta:
    """Return stones added, removed, or color-changed between two detected states."""
    before_by_coord = {stone.coord: stone for stone in before.stones}
    after_by_coord = {stone.coord: stone for stone in after.stones}

    added = [after_by_coord[coord] for coord in sorted(after_by_coord.keys() - before_by_coord.keys())]
    removed = [before_by_coord[coord] for coord in sorted(before_by_coord.keys() - after_by_coord.keys())]
    changed = []
    for coord in sorted(before_by_coord.keys() & after_by_coord.keys()):
        before_stone = before_by_coord[coord]
        after_stone = after_by_coord[coord]
        if before_stone.color != after_stone.color:
            changed.append(
                {
                    "coord": coord,
                    "before": asdict(before_stone),
                    "after": asdict(after_stone),
                }
            )

    return BoardDelta(
        added=added,
        removed=removed,
        changed=changed,
        before_summary=before.summary,
        after_summary=after.summary,
    )


def delta_between_snapshots(
    before_image: np.ndarray,
    after_image: np.ndarray,
    corners: np.ndarray | None = None,
    size: int = 19,
    board_pixels: int = 1000,
    sample_radius_ratio: float = 0.34,
    black_l_threshold: float = 80.0,
    white_l_threshold: float = 165.0,
    white_s_threshold: float = 75.0,
    stone_min_radius_ratio: float = 0.2,
    stone_max_radius_ratio: float = 0.48,
    stone_min_circularity: float = 0.45,
    stone_max_snap_distance_ratio: float = 0.52,
    overlay_fisheye_k: float = 0.0,
    skip_i: bool = True,
) -> tuple[BoardState, BoardState, BoardDelta]:
    """Detect two board states and return their delta."""
    before_state = board_state_from_image(
        before_image,
        corners=corners,
        size=size,
        board_pixels=board_pixels,
        sample_radius_ratio=sample_radius_ratio,
        black_l_threshold=black_l_threshold,
        white_l_threshold=white_l_threshold,
        white_s_threshold=white_s_threshold,
        stone_min_radius_ratio=stone_min_radius_ratio,
        stone_max_radius_ratio=stone_max_radius_ratio,
        stone_min_circularity=stone_min_circularity,
        stone_max_snap_distance_ratio=stone_max_snap_distance_ratio,
        overlay_fisheye_k=overlay_fisheye_k,
        skip_i=skip_i,
    )
    after_state = board_state_from_image(
        after_image,
        corners=corners,
        size=size,
        board_pixels=board_pixels,
        sample_radius_ratio=sample_radius_ratio,
        black_l_threshold=black_l_threshold,
        white_l_threshold=white_l_threshold,
        white_s_threshold=white_s_threshold,
        stone_min_radius_ratio=stone_min_radius_ratio,
        stone_max_radius_ratio=stone_max_radius_ratio,
        stone_min_circularity=stone_min_circularity,
        stone_max_snap_distance_ratio=stone_max_snap_distance_ratio,
        overlay_fisheye_k=overlay_fisheye_k,
        skip_i=skip_i,
    )
    return before_state, after_state, delta_between_board_states(before_state, after_state)


def draw_debug_overlay(
    image: np.ndarray,
    stones: list[Stone],
    corners: np.ndarray,
    size: int,
    board_pixels: int,
    output_path: Path,
) -> None:
    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(np.int32)], isClosed=True, color=(0, 180, 255), thickness=2)

    _warped, _to_board, to_image = warp_board(image, corners, board_pixels)
    spacing = (board_pixels - 1) / (size - 1)

    for row in range(size):
        for col in range(size):
            board_point = np.array([[[col * spacing, row * spacing]]], dtype=np.float32)
            image_point = cv2.perspectiveTransform(board_point, to_image)[0, 0]
            cv2.circle(overlay, tuple(np.round(image_point).astype(int)), 2, (120, 120, 120), -1)

    for stone in stones:
        point = tuple(np.round(stone.image_xy).astype(int))
        color = (30, 30, 30) if stone.color == "black" else (245, 245, 245)
        outline = (255, 255, 255) if stone.color == "black" else (0, 0, 0)
        cv2.circle(overlay, point, 12, outline, 2)
        cv2.circle(overlay, point, 9, color, -1)
        cv2.putText(
            overlay,
            stone.coord,
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an overhead board snapshot.")
    parser.add_argument("--corners", type=parse_corners, help="Board corners in TL,TR,BR,BL order.")
    parser.add_argument("--size", type=int, default=19, help="Board size. Defaults to 19.")
    parser.add_argument("--board-pixels", type=int, default=1000, help="Internal rectified board size.")
    parser.add_argument(
        "--sample-radius-ratio",
        type=float,
        default=0.34,
        help="Sample radius as a fraction of grid spacing.",
    )
    parser.add_argument("--black-l-threshold", type=float, default=80.0, help="LAB L threshold for black stones.")
    parser.add_argument("--white-l-threshold", type=float, default=165.0, help="LAB L threshold for white stones.")
    parser.add_argument("--white-s-threshold", type=float, default=75.0, help="HSV S max for white stones.")
    parser.add_argument("--stone-min-radius-ratio", type=float, default=0.2, help="Minimum stone radius as grid-spacing ratio.")
    parser.add_argument("--stone-max-radius-ratio", type=float, default=0.48, help="Maximum stone radius as grid-spacing ratio.")
    parser.add_argument("--stone-min-circularity", type=float, default=0.45, help="Minimum contour circularity for stone candidates.")
    parser.add_argument(
        "--stone-max-snap-distance-ratio",
        type=float,
        default=0.52,
        help="Maximum distance from candidate center to nearest intersection as grid-spacing ratio.",
    )
    parser.add_argument("--overlay-fisheye-k", type=float, default=0.0, help="Curvature value for detector grid points.")
    parser.add_argument("--include-i", action="store_true", help="Do not skip column I in coordinates.")
    parser.add_argument("--json-output", type=Path, help="Optional path to write JSON board state.")
    parser.add_argument("--debug-output", type=Path, help="Optional path to write a visual detection overlay.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    corners = args.corners if args.corners is not None else None
    state = board_state_from_image(
        image=image,
        corners=corners,
        size=args.size,
        board_pixels=args.board_pixels,
        sample_radius_ratio=args.sample_radius_ratio,
        black_l_threshold=args.black_l_threshold,
        white_l_threshold=args.white_l_threshold,
        white_s_threshold=args.white_s_threshold,
        stone_min_radius_ratio=args.stone_min_radius_ratio,
        stone_max_radius_ratio=args.stone_max_radius_ratio,
        stone_min_circularity=args.stone_min_circularity,
        stone_max_snap_distance_ratio=args.stone_max_snap_distance_ratio,
        overlay_fisheye_k=args.overlay_fisheye_k,
        skip_i=not args.include_i,
    )
    result = {"image": str(args.image), **board_state_to_jsonable(state)}

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")

    if args.debug_output:
        draw_debug_overlay(
            image=image,
            stones=state.stones,
            corners=np.array(state.corners_tl_tr_br_bl, dtype=np.float32),
            size=args.size,
            board_pixels=args.board_pixels,
            output_path=args.debug_output,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
