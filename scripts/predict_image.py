"""Run inference on a single pig image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference import load_yolo_model, run_inference
from src.utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pig detection on a single image")
    parser.add_argument("--weights", required=True, help="Path to model weights .pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    pred_dir = ensure_dir(output_dir / "predictions")

    model, message = load_yolo_model(args.weights)
    if model is None:
        print(message)
        return

    image_path = Path(args.image)
    result = run_inference(model, image_path, conf=args.conf)

    stem = image_path.stem
    annotated_path = pred_dir / f"{stem}_detected.jpg"
    result.pop("annotated_image").save(annotated_path)

    json_path = pred_dir / f"{stem}_result.json"
    serializable = {k: v for k, v in result.items() if k != "annotated_image"}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("单图推理结果")
    print("=" * 50)
    print(f"猪只数量：{result['pig_count']}")
    print(f"平均置信度：{result['mean_confidence']:.4f}")
    print(f"检测框面积占比：{result['bbox_area_ratio']:.4f}")
    print(f"风险等级：{result['risk_level']}")
    print(f"风险原因：{result['risk_reason']}")
    print("=" * 50)
    print(f"检测图已保存：{annotated_path}")
    print(f"结果 JSON：{json_path}")


if __name__ == "__main__":
    main()
