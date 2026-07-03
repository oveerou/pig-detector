"""Visualization helpers for batch analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_count_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot pig count distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["pig_count"], bins=20, kde=True)
    plt.xlabel("猪只数量")
    plt.ylabel("图片数量")
    plt.title("猪只数量分布")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean confidence distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["mean_confidence"], bins=20, kde=True)
    plt.xlabel("平均置信度")
    plt.ylabel("图片数量")
    plt.title("平均置信度分布")
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
    plt.xlabel("风险等级")
    plt.ylabel("图片数量")
    plt.title("风险等级分布")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
