from src.analytics import calculate_bbox_area_ratio, classify_risk


def test_calculate_bbox_area_ratio():
    detections = [
        {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
    ]
    ratio = calculate_bbox_area_ratio(detections, 200, 200)
    assert ratio == pytest.approx(0.25, abs=1e-6)


def test_calculate_bbox_area_ratio_empty():
    ratio = calculate_bbox_area_ratio([], 100, 100)
    assert ratio == 0.0


def test_classify_risk_normal():
    level, reason = classify_risk(5, 0.1, 0.8)
    assert level == "normal"
    assert reason == "无异常"


def test_classify_risk_medium_by_count():
    level, reason = classify_risk(10, 0.1, 0.8)
    assert level == "medium"
    assert "猪只数量中等" in reason


def test_classify_risk_high_by_count():
    level, reason = classify_risk(20, 0.1, 0.8)
    assert level == "high"
    assert "猪只数量较多" in reason


def test_classify_risk_low_confidence():
    level, reason = classify_risk(5, 0.1, 0.3)
    assert "平均置信度较低" in reason


import pytest
