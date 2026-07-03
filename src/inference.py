"""Inference utilities for pig detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    annotated_pil = pil_image.copy()

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

    annotated_pil = _draw_annotations(pil_image, detections, risk_level, risk_reason)

    return {
        "pig_count": pig_count,
        "mean_confidence": mean_confidence,
        "bbox_area_ratio": bbox_area_ratio,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "detections": detections,
        "annotated_image": annotated_pil,
    }


def _draw_annotations(
    pil_image: Image.Image,
    detections: list[dict[str, Any]],
    risk_level: str,
    risk_reason: str,
) -> Image.Image:
    draw = ImageDraw.Draw(pil_image)
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 20)
        font_small = ImageFont.truetype(str(font_path), 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = f"pig {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.rectangle([x1, y1 - 26, x1 + 120, y1], fill=(0, 255, 0))
        draw.text((x1 + 4, y1 - 24), label, fill=(0, 0, 0), font=font_small)

    title_text = "猪只检测结果 (Pig Detection)"
    draw.rectangle([10, 10, 10 + 260, 10 + 30], fill=(0, 0, 0, 128))
    draw.text((16, 14), title_text, fill=(255, 255, 255), font=font_small)

    risk_color = {"normal": (0, 255, 0), "medium": (255, 165, 0), "high": (255, 0, 0)}.get(risk_level, (255, 255, 255))
    risk_text = f"风险等级: {risk_level} | {risk_reason}"
    draw.rectangle([10, 45, 10 + 360, 45 + 26], fill=(0, 0, 0, 128))
    draw.text((16, 49), risk_text, fill=risk_color, font=font_small)

    return pil_image
