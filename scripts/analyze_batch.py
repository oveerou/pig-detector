"""Run batch inference on test images and generate analysis report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.inference import load_yolo_model, run_inference
from src.utils import ensure_dir, load_config
from src.visualization import (
    plot_confidence_distribution,
    plot_count_distribution,
    plot_risk_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch analysis on pig test images")
    parser.add_argument("--weights", required=True, help="Path to model weights .pt")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    batch_dir = ensure_dir(output_dir / "batch_analysis")

    model, message = load_yolo_model(args.weights)
    if model is None:
        print(message)
        return

    image_dir = Path(args.image_dir)
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    records = []
    for img_path in image_paths:
        result = run_inference(model, img_path, conf=args.conf)
        records.append({
            "image_name": img_path.name,
            "pig_count": result["pig_count"],
            "mean_confidence": result["mean_confidence"],
            "bbox_area_ratio": result["bbox_area_ratio"],
            "risk_level": result["risk_level"],
            "risk_reason": result["risk_reason"],
        })

    df = pd.DataFrame(records)
    csv_path = batch_dir / "predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_count_distribution(df, batch_dir / "count_distribution.png")
    plot_confidence_distribution(df, batch_dir / "confidence_distribution.png")
    plot_risk_distribution(df, batch_dir / "risk_distribution.png")

    summary_lines = [
        "# 批量分析汇总",
        "",
        "## 数据集",
        f"- 图片总数：{len(df)}",
        f"- 平均猪只数：{df['pig_count'].mean():.2f}",
        f"- 最大猪只数：{df['pig_count'].max()}",
        f"- 最小猪只数：{df['pig_count'].min()}",
        "",
        "## 风险分布",
        f"- 正常：{len(df[df['risk_level'] == 'normal'])}",
        f"- 中等：{len(df[df['risk_level'] == 'medium'])}",
        f"- 高：{len(df[df['risk_level'] == 'high'])}",
        "",
        "## 说明",
        "本分析基于模型对本地测试图片文件夹的预测结果。",
    ]
    summary_path = batch_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 50)
    print("批量分析完成")
    print("=" * 50)
    print(f"图片总数：{len(df)}")
    print(f"平均猪只数：{df['pig_count'].mean():.2f}")
    print(f"风险分布：\n{df['risk_level'].value_counts().to_string()}")
    print("=" * 50)
    print(f"CSV 报告：{csv_path}")
    print(f"汇总文档：{summary_path}")


if __name__ == "__main__":
    main()
