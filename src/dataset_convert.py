"""Convert raw JSON annotations to YOLO detection format."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def load_annotation(path: Path) -> dict[str, Any]:
    """Load a JSON annotation file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygon_to_bbox(points: list[list[float]]) -> list[float] | None:
    """Convert polygon points to bounding box [x1, y1, x2, y2]."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox(boxes: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert [x1, y1, x2, y2] to YOLO normalized [x_center, y_center, w, h]."""
    x1, y1, x2, y2 = boxes
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return x_center, y_center, w, h


def extract_valid_boxes(annotation: dict[str, Any], width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Extract valid YOLO boxes from a raw annotation."""
    boxes: list[tuple[float, float, float, float]] = []
    for shape in annotation.get("shape", []):
        label = shape.get("label", "")
        if label != "pig":
            continue

        raw_boxes = shape.get("boxes")
        if raw_boxes is None or len(raw_boxes) != 4:
            points = shape.get("points")
            if points:
                raw_boxes = polygon_to_bbox(points)
            else:
                continue

        if raw_boxes is None:
            continue

        normalized = normalize_bbox(raw_boxes, width, height)
        if normalized is None:
            continue
        boxes.append(normalized)
    return boxes


def convert_single_annotation(
    annotation_path: Path,
    image_dir: Path,
    output_label_path: Path,
) -> tuple[bool, int]:
    """Convert one annotation file to YOLO label file. Return (success, box_count)."""
    annotation = load_annotation(annotation_path)
    image_path = image_dir / (annotation_path.stem + ".jpg")
    if not image_path.exists():
        image_path = image_dir / (annotation_path.stem + ".png")
    if not image_path.exists():
        return False, 0

    with Image.open(image_path) as img:
        width, height = img.size

    boxes = extract_valid_boxes(annotation, width, height)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with output_label_path.open("w", encoding="utf-8") as f:
        for x_center, y_center, w, h in boxes:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    return True, len(boxes)
