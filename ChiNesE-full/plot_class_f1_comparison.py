# -*- coding: utf-8 -*-
"""
对比 Baseline 与 Full Model 各类别峰值 F1，生成顶刊风格并排柱状图。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, TRAIN_CLASS_LOG_CSV


def generate_class_f1_comparison(data_dir):
    """
    Reads class-level training logs for Baseline and Full Model,
    extracts the peak F1 score per category, and generates a publication-ready
    grouped bar chart with an external legend to prevent overlapping.
    """
    baseline_path = os.path.join(data_dir, "train_class_log_baseline.csv")
    full_model_path = TRAIN_CLASS_LOG_CSV

    if not os.path.exists(baseline_path) or not os.path.exists(full_model_path):
        raise FileNotFoundError("Error: Log files missing in the specified directory.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 16,
        "font.size": 11,
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    sns.set_theme(style="ticks", context="paper", font="Times New Roman")

    df_base = pd.read_csv(baseline_path)
    df_full = pd.read_csv(full_model_path)

    base_f1 = df_base.groupby("class_name")["f1"].max().reset_index()
    base_f1.rename(columns={"f1": "Baseline"}, inplace=True)

    full_f1 = df_full.groupby("class_name")["f1"].max().reset_index()
    full_f1.rename(columns={"f1": "Full Model"}, inplace=True)

    df_merged = pd.merge(base_f1, full_f1, on="class_name")
    df_merged = df_merged.sort_values(by="Baseline", ascending=False).reset_index(drop=True)

    categories = df_merged["class_name"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        x - width / 2,
        df_merged["Baseline"],
        width,
        label="Baseline",
        color="#4C72B0",
        edgecolor="black",
        linewidth=1,
    )

    bars2 = ax.bar(
        x + width / 2,
        df_merged["Full Model"],
        width,
        label="Full Model",
        color="#C44E52",
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xlabel("Entity Category", fontweight="bold", labelpad=12, fontsize=16)
    ax.set_ylabel("Peak F1 Score", fontweight="bold", labelpad=12, fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)

    ax.set_ylim(0, 1.15)

    sns.despine()

    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=90,
            )

    annotate_bars(bars1)
    annotate_bars(bars2)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=14,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")

    plt.tight_layout()

    pdf_path = os.path.join(data_dir, "class_f1_comparison.pdf")
    png_path = os.path.join(data_dir, "class_f1_comparison.png")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")

    print("\nPlot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    generate_class_f1_comparison(data_dir=OUTPUT_DIR)
