"""Analytics helpers for pig detection results."""
from __future__ import annotations

from typing import Any


def calculate_bbox_area_ratio(detections: list[dict[str, Any]], image_width: int, image_height: int) -> float:
    """Calculate the ratio of total detection area to image area."""
    if not detections or image_width <= 0 or image_height <= 0:
        return 0.0
    total_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        total_area += max(0.0, (x2 - x1) * (y2 - y1))
    image_area = image_width * image_height
    return total_area / image_area if image_area > 0 else 0.0


def classify_risk(
    pig_count: int,
    bbox_area_ratio: float,
    mean_confidence: float,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Classify risk level based on count, area ratio, and confidence.

    Returns (risk_level, risk_reason).
    """
    if thresholds is None:
        thresholds = {
            "medium_count": 8,
            "high_count": 15,
            "medium_area": 0.25,
            "high_area": 0.45,
            "low_confidence": 0.5,
        }

    reasons: list[str] = []
    risk_level = "normal"

    if pig_count >= thresholds["high_count"]:
        risk_level = "high"
        reasons.append(f"猪只数量较多（{pig_count} 只）")
    elif pig_count >= thresholds["medium_count"]:
        risk_level = "medium"
        reasons.append(f"猪只数量中等（{pig_count} 只）")

    if bbox_area_ratio >= thresholds["high_area"]:
        risk_level = "high"
        reasons.append(f"检测框占比较高（{bbox_area_ratio:.1%}）")
    elif bbox_area_ratio >= thresholds["medium_area"] and risk_level != "high":
        risk_level = "medium"
        reasons.append(f"检测框占比中等（{bbox_area_ratio:.1%}）")

    if mean_confidence < thresholds["low_confidence"]:
        reasons.append("平均置信度较低，可能存在遮挡、光照或图像质量问题")

    if not reasons:
        risk_reason = "无异常"
    else:
        risk_reason = "；".join(reasons)

    return risk_level, risk_reason
