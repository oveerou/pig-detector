"""Audit the raw pig dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_audit import audit_dataset
from src.utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw pig dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    image_dir = Path(paths["raw_train_images"])
    annotation_dir = Path(paths["raw_train_annotations"])
    test_image_dir = Path(paths["raw_test_images"])
    output_dir = ensure_dir(paths["output_dir"])

    print("正在审计数据集...")
    summary = audit_dataset(image_dir, annotation_dir, test_image_dir)

    summary_path = output_dir / "dataset_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("数据审计结果")
    print("=" * 50)
    print(f"训练图片数量：{summary['train_image_count']}")
    print(f"训练标注数量：{summary['train_annotation_count']}")
    print(f"测试图片数量：{summary['test_image_count']}")
    print(f"图片与标注匹配数量：{summary['matched_image_annotation']}")
    print(f"未匹配图片：{summary['unmatched_images']}")
    print(f"未匹配标注：{summary['unmatched_annotations']}")
    print(f"每张图最多猪只数：{summary['max_pigs_per_image']}")
    print(f"每张图最少猪只数：{summary['min_pigs_per_image']}")
    print(f"每张图平均猪只数：{summary['mean_pigs_per_image']}")
    print(f"空标注数量：{summary['empty_annotations']}")
    print(f"非 pig 标签：{summary['non_pig_labels']}")
    print(f"无效 bbox 数量：{summary['invalid_boxes']}")
    print(f"有效 pig bbox 数量：{summary['valid_pig_boxes']}")
    print(f"样例图片尺寸：{summary['sample_image_sizes']}")
    print("=" * 50)
    print(f"详细摘要已保存：{summary_path}")


if __name__ == "__main__":
    main()
