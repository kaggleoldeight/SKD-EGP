# -*- coding: utf-8 -*-
"""
读取消融实验训练日志，绘制符合顶刊规范的 F1 学习曲线对比图。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from config import OUTPUT_DIR


def plot_ablation_f1_curves(data_dir):
    """
    Reads multiple training logs from ablation experiments,
    extracts the F1 score per epoch, and plots a publication-ready line chart.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    sns.set_theme(style="ticks", context="paper", font="Times New Roman")

    file_label_mapping = {
        "train_log_完整模型.csv": "Full Model",
        "train_log_去除LLM数据增强.csv": "w/o LLM Augmentation",
        "train_log_去除动态 Token 掩码.csv": "w/o Dynamic Token Masking",
        "train_log_去除序列拼接与动态位置平移.csv": "w/o Seq. Concatenation & Shift",
        "train_log_去除efficient.csv": "w/o Efficient GlobalPointer",
        "train_log_baseline.csv": "Baseline",
    }

    styles = {
        "Full Model": {"color": "#D62728", "marker": "o", "linestyle": "-"},
        "w/o LLM Augmentation": {"color": "#1F77B4", "marker": "s", "linestyle": "--"},
        "w/o Dynamic Token Masking": {"color": "#FF7F0E", "marker": "^", "linestyle": "-."},
        "w/o Seq. Concatenation & Shift": {"color": "#2CA02C", "marker": "v", "linestyle": ":"},
        "w/o Efficient GlobalPointer": {"color": "#9467BD", "marker": "D", "linestyle": "--"},
        "Baseline": {"color": "#7F7F7F", "marker": "x", "linestyle": "-"},
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for filename, label in file_label_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            continue

        df = pd.read_csv(filepath)

        if "epoch" not in df.columns or "f1" not in df.columns:
            print(f"Error: Required columns missing in {filename}")
            continue

        style = styles[label]
        ax.plot(
            df["epoch"],
            df["f1"],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    ax.set_xlabel("Training Epochs", fontweight="bold", labelpad=10)
    ax.set_ylabel("F1 Score", fontweight="bold", labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linestyle="--", alpha=0.6)

    sns.despine()

    ax.legend(frameon=True, edgecolor="black", loc="lower right")

    plt.tight_layout()

    pdf_path = os.path.join(data_dir, "ablation_f1_learning_curves.pdf")
    png_path = os.path.join(data_dir, "ablation_f1_learning_curves.png")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")

    print("\nPlot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    plot_ablation_f1_curves(OUTPUT_DIR)
