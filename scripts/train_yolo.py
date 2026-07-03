"""Train YOLOv8n on the pig detection dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n for pig detection")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    data_yaml = Path(cfg["paths"]["yolo_root"]) / "data.yaml"

    if not data_yaml.exists():
        print(f"data.yaml not found: {data_yaml}")
        print("Please run `python scripts/convert_json_to_yolo.py --config configs/default.yaml` first.")
        return

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]

    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "无"

    print("=" * 50)
    print("开始训练 YOLOv8n 猪只检测模型")
    print("=" * 50)
    print(f"模型 (Model)：{train_cfg['model']}")
    print(f"轮数 (Epochs)：{epochs}")
    print(f"图片尺寸 (Image size)：{train_cfg['imgsz']}")
    print(f"批次大小 (Batch)：{train_cfg['batch']}")
    print(f"设备 (Device)：{train_cfg['device']}")
    print(f"GPU 可用：{gpu_available} ({gpu_name})")
    print("=" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics，请先运行：pip install -r requirements.txt")
        return

    output_dir = Path(cfg["paths"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n训练中，每轮会显示一个进度条，请稍候...\n")

    model = YOLO(train_cfg["model"])
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        project=str(output_dir),
        name="pig_yolov8n",
        verbose=False,
    )

    print("\n" + "=" * 50)
    print("训练完成")
    print("=" * 50)
    print(f"模型权重保存位置：{output_dir / 'pig_yolov8n' / 'weights'}")
    print(f"  - best.pt：验证集上表现最好的模型")
    print(f"  - last.pt：最后一轮的模型")
    print("=" * 50)


if __name__ == "__main__":
    main()
