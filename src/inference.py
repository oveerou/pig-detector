"""Inference utilities for pig detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.analytics import calculate_bbox_area_ratio, classify_risk
from src.utils import load_config


def load_yolo_model(weights_path: str | Path):
    """Load a YOLO model from weights path."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("未安装 ultralytics，请先运行：pip install -r requirements.txt") from e

    path = Path(weights_path)
    if not path.exists():
        return None, f"模型权重不存在：{path}"
    return YOLO(str(path)), ""


def run_inference(
    model,
    image: Image.Image | np.ndarray | str | Path,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Run YOLO inference on a single image and return structured results."""
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image.convert("RGB")

    width, height = pil_image.size
    results = model(pil_image, conf=conf, verbose=False)
    result = results[0]

    detections: list[dict[str, Any]] = []
    confidences: list[float] = []
    annotated_image = result.plot()
    annotated_pil = Image.fromarray(annotated_image)

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "pig_count": 0,
            "mean_confidence": 0.0,
            "bbox_area_ratio": 0.0,
            "risk_level": "normal",
            "risk_reason": "未检测到猪只",
            "detections": [],
            "annotated_image": annotated_pil,
        }

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
        if int(cls_id) != 0:
            continue
        confidences.append(float(score))
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
        detections.append({
            "class_name": "pig",
            "confidence": float(score),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "area_ratio": float(area_ratio),
        })

    pig_count = len(detections)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    bbox_area_ratio = calculate_bbox_area_ratio(detections, width, height)

    cfg = load_config()
    risk_level, risk_reason = classify_risk(pig_count, bbox_area_ratio, mean_confidence, cfg.get("risk", {}))

    return {
        "pig_count": pig_count,
        "mean_confidence": mean_confidence,
        "bbox_area_ratio": bbox_area_ratio,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "detections": detections,
        "annotated_image": annotated_pil,
    }
