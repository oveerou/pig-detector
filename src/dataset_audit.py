"""Dataset audit utilities for pig-detector."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def load_annotation(path: Path) -> dict[str, Any]:
    """Load a JSON annotation file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_bbox(boxes: list[float] | None) -> bool:
    """Check whether a bounding box is valid."""
    if boxes is None or len(boxes) != 4:
        return False
    x1, y1, x2, y2 = boxes
    return x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0


def get_image_size(image_path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image, or None if unreadable."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def audit_dataset(
    image_dir: Path,
    annotation_dir: Path,
    test_image_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit the raw pig dataset and return summary statistics."""
    image_paths = sorted(image_dir.iterdir())
    annotation_paths = sorted(annotation_dir.iterdir())
    image_files = [p for p in image_paths if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    annotation_files = [p for p in annotation_paths if p.is_file() and p.suffix.lower() == ".json"]

    image_stems = {p.stem for p in image_files}
    annotation_stems = {p.stem for p in annotation_files}
    matched = image_stems & annotation_stems
    unmatched_images = image_stems - annotation_stems
    unmatched_annotations = annotation_stems - image_stems

    per_image_counts: list[int] = []
    empty_annotations = 0
    non_pig_labels: Counter[str] = Counter()
    invalid_boxes = 0
    valid_pig_boxes = 0
    total_shapes = 0
    missing_image_sizes = 0

    for ann_path in annotation_files:
        ann = load_annotation(ann_path)
        shapes = ann.get("shape", [])
        if not shapes:
            empty_annotations += 1
            per_image_counts.append(0)
            continue

        pig_count = 0
        for shape in shapes:
            total_shapes += 1
            label = shape.get("label", "")
            if label != "pig":
                non_pig_labels[label] += 1
                continue
            boxes = shape.get("boxes")
            if not is_valid_bbox(boxes):
                invalid_boxes += 1
                continue
            pig_count += 1
            valid_pig_boxes += 1
        per_image_counts.append(pig_count)

    # Verify image sizes for a sample
    sample_sizes: list[tuple[int, int]] = []
    for img_path in image_files[:10]:
        size = get_image_size(img_path)
        if size is None:
            missing_image_sizes += 1
        else:
            sample_sizes.append(size)

    summary = {
        "train_image_count": len(image_files),
        "train_annotation_count": len(annotation_files),
        "test_image_count": len(list(test_image_dir.iterdir())) if test_image_dir and test_image_dir.exists() else 0,
        "matched_image_annotation": len(matched),
        "unmatched_images": len(unmatched_images),
        "unmatched_annotations": len(unmatched_annotations),
        "max_pigs_per_image": max(per_image_counts) if per_image_counts else 0,
        "min_pigs_per_image": min(per_image_counts) if per_image_counts else 0,
        "mean_pigs_per_image": round(sum(per_image_counts) / len(per_image_counts), 2) if per_image_counts else 0.0,
        "empty_annotations": empty_annotations,
        "non_pig_labels": dict(non_pig_labels),
        "invalid_boxes": invalid_boxes,
        "valid_pig_boxes": valid_pig_boxes,
        "total_shapes": total_shapes,
        "sample_image_sizes": sample_sizes,
        "missing_sample_image_sizes": missing_image_sizes,
    }
    return summary
