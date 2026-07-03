"""Evaluate trained YOLOv8 model and save metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 pig detection model")
    parser.add_argument("--weights", required=True, help="Path to model weights .pt")
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml = args.data or str(Path(cfg["paths"]["yolo_root"]) / "data.yaml")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics，请先运行：pip install -r requirements.txt")
        return

    model = YOLO(args.weights)
    metrics = model.val(data=data_yaml)

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"Precision：{result['precision']:.4f}")
    print(f"Recall：{result['recall']:.4f}")
    print(f"mAP@50：{result['map50']:.4f}")
    print(f"mAP@50:95：{result['map50_95']:.4f}")
    print("=" * 50)
    print(f"指标已保存：{metrics_path}")


if __name__ == "__main__":
    main()
