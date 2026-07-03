"""Visualization helpers for batch analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_count_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot pig count distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["pig_count"], bins=20, kde=True)
    plt.xlabel("猪只数量 (Pig Count)")
    plt.ylabel("图片数量 (Image Count)")
    plt.title("猪只数量分布 (Pig Count Distribution)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean confidence distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["mean_confidence"], bins=20, kde=True)
    plt.xlabel("平均置信度 (Mean Confidence)")
    plt.ylabel("图片数量 (Image Count)")
    plt.title("平均置信度分布 (Confidence Distribution)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_risk_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot risk level distribution."""
    plt.figure(figsize=(8, 6))
    order = ["normal", "medium", "high"]
    counts = df["risk_level"].value_counts().reindex(order, fill_value=0)
    colors = {"normal": "green", "medium": "orange", "high": "red"}
    counts.plot(kind="bar", color=[colors.get(x, "gray") for x in counts.index])
    plt.xlabel("风险等级 (Risk Level)")
    plt.ylabel("图片数量 (Image Count)")
    plt.title("风险等级分布 (Risk Distribution)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
