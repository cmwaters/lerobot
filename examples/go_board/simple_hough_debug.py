#!/usr/bin/env python
"""Run a plain whole-image Hough circle detector for one Go-board snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(annotation_path: Path) -> tuple[np.ndarray, Path]:
    data = json.loads(annotation_path.read_text())
    image_path = annotation_path.parent / data["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    return image, image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "annotation",
        type=Path,
        nargs="?",
        default=Path("examples/go_board/cv_snapshots/20260618_200742_10_bowl_and_arm_side.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/go_board/cv_snapshots/simple_hough"))
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-dist", type=float, default=42.0)
    parser.add_argument("--param1", type=float, default=80.0)
    parser.add_argument("--param2", type=float, default=18.0)
    parser.add_argument("--min-radius", type=int, default=12)
    parser.add_argument("--max-radius", type=int, default=34)
    args = parser.parse_args()

    image, image_path = load_image(args.annotation)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 120)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=args.dp,
        minDist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.min_radius,
        maxRadius=args.max_radius,
    )

    overlay = image.copy()
    count = 0
    if circles is not None:
        for idx, (x, y, radius) in enumerate(np.round(circles[0]).astype(int), start=1):
            count += 1
            cv2.circle(overlay, (x, y), radius, (0, 0, 255), 3)
            cv2.circle(overlay, (x, y), 2, (255, 255, 255), -1)
            cv2.putText(
                overlay,
                str(idx),
                (x + radius + 4, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    label = (
        f"plain full-frame Hough: {count} circles "
        f"r={args.min_radius}-{args.max_radius} p2={args.param2:g}"
    )
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    output_dir = args.output_dir / args.annotation.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "plain_hough_circles.jpg"
    edges_path = output_dir / "canny_edges.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(edges_path), edges)
    print(f"image={image_path}")
    print(f"circles={count}")
    print(f"overlay={overlay_path}")
    print(f"edges={edges_path}")


if __name__ == "__main__":
    main()
