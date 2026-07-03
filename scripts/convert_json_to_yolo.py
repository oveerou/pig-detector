"""Convert raw JSON annotations to YOLO format and split train/val."""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_convert import convert_single_annotation
from src.utils import ensure_dir, load_config


def collect_image_files(image_dir: Path, extensions: list[str]) -> list[Path]:
    """Collect image files with given extensions."""
    files = []
    for ext in extensions:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON annotations to YOLO format")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    raw_image_dir = Path(paths["raw_train_images"])
    raw_annotation_dir = Path(paths["raw_train_annotations"])
    raw_test_dir = Path(paths["raw_test_images"])
    yolo_root = ensure_dir(paths["yolo_root"])

    image_files = collect_image_files(raw_image_dir, data_cfg["image_extensions"])
    random.seed(data_cfg["seed"])
    random.shuffle(image_files)

    train_ratio = data_cfg["train_ratio"]
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    splits = {
        "train": train_images,
        "val": val_images,
    }

    total_boxes = 0
    for split_name, images in splits.items():
        image_output_dir = ensure_dir(yolo_root / "images" / split_name)
        label_output_dir = ensure_dir(yolo_root / "labels" / split_name)
        for img_path in images:
            ann_path = raw_annotation_dir / (img_path.stem + ".json")
            if not ann_path.exists():
                continue
            label_path = label_output_dir / (img_path.stem + ".txt")
            success, box_count = convert_single_annotation(ann_path, raw_image_dir, label_path)
            if success:
                shutil.copy2(img_path, image_output_dir / img_path.name)
                total_boxes += box_count

    # Copy test images
    test_output_dir = ensure_dir(yolo_root / "images" / "test")
    if raw_test_dir.exists():
        for img_path in collect_image_files(raw_test_dir, data_cfg["image_extensions"]):
            shutil.copy2(img_path, test_output_dir / img_path.name)

    data_yaml = yolo_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {yolo_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("names:\n")
        for idx, name in enumerate(cfg["project"]["class_names"]):
            f.write(f"  {idx}: {name}\n")

    print("=" * 50)
    print("YOLO 数据集转换完成")
    print("=" * 50)
    print(f"训练集图片：{len(train_images)}")
    print(f"验证集图片：{len(val_images)}")
    print(f"测试集图片：{len(list(test_output_dir.iterdir()))}")
    print(f"总标注框数：{total_boxes}")
    print(f"data.yaml：{data_yaml}")
    print("=" * 50)


if __name__ == "__main__":
    main()
