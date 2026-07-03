import pytest

from src.dataset_convert import normalize_bbox, polygon_to_bbox


def test_normalize_bbox_standard_case():
    boxes = [100, 50, 300, 250]
    width, height = 1000, 500
    result = normalize_bbox(boxes, width, height)
    assert result is not None
    x_center, y_center, w, h = result
    assert x_center == pytest.approx(0.2, abs=1e-6)
    assert y_center == pytest.approx(0.3, abs=1e-6)
    assert w == pytest.approx(0.2, abs=1e-6)
    assert h == pytest.approx(0.4, abs=1e-6)


def test_normalize_bbox_invalid_width():
    result = normalize_bbox([0, 0, 10, 10], 0, 100)
    assert result is None


def test_normalize_bbox_invalid_coords():
    result = normalize_bbox([100, 100, 50, 50], 1000, 1000)
    assert result is None


def test_polygon_to_bbox():
    points = [[100, 50], [300, 50], [300, 250], [100, 250]]
    result = polygon_to_bbox(points)
    assert result == [100, 50, 300, 250]


def test_polygon_to_bbox_empty():
    result = polygon_to_bbox([])
    assert result is None
