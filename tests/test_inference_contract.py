from PIL import Image

from app import predict_image
from src.inference import load_yolo_model


def test_load_yolo_model_missing_weights_returns_message(tmp_path):
    missing = tmp_path / "missing.pt"
    model, message = load_yolo_model(missing)
    assert model is None
    assert "模型权重不存在" in message


def test_app_predict_image_none_returns_prompt():
    result = predict_image(None)
    _, _, _, _, message, _ = result
    assert "请上传猪舍图片" in message


def test_app_predict_image_missing_weights_returns_message(tmp_path):
    image = Image.new("RGB", (640, 480), color="gray")
    missing = tmp_path / "missing.pt"
    result = predict_image(image, weights_path=str(missing))
    _, _, _, _, message, _ = result
    assert "模型权重未找到" in message
