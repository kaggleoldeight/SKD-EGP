# -*- coding: utf-8 -*-
"""
将 CMeEE-V2 与 CLUENER 各类别峰值 F1（Baseline vs Full Model）并排柱状图
绘制在同一张图中，便于论文对比展示。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR, PROJECT_DIR

ABLATION_ROOT = os.path.dirname(PROJECT_DIR)
CLUENER_PROJECT = os.path.join(ABLATION_ROOT, "完整模型 - CLUENER - 完整")
CLUENER_OUTPUT = os.path.join(CLUENER_PROJECT, "outputs")

DEFAULT_CMEEE_BASELINE = os.path.join(OUTPUT_DIR, "train_class_log_baseline.csv")
DEFAULT_CMEEE_FULL = os.path.join(OUTPUT_DIR, "train_class_log.csv")
DEFAULT_CLUENER_BASELINE = os.path.join(
    CLUENER_OUTPUT, "CLUENER_train_class_log_baseline.csv"
)
DEFAULT_CLUENER_FULL = os.path.join(CLUENER_OUTPUT, "CLUENER_train_class_log.csv")

FONT_AXIS_LABEL = 21
FONT_TICK_X = 20
FONT_TICK_Y = 19
FONT_LEGEND = 19
FONT_ANNOT = 17
FONT_PANEL = 20


def _load_peak_f1_df(baseline_path, full_path):
    for path in (baseline_path, full_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing class log file: {path}")

    df_base = pd.read_csv(baseline_path)
    df_full = pd.read_csv(full_path)

    base_f1 = df_base.groupby("class_name")["f1"].max().reset_index()
    base_f1.rename(columns={"f1": "Baseline"}, inplace=True)

    full_f1 = df_full.groupby("class_name")["f1"].max().reset_index()
    full_f1.rename(columns={"f1": "Full Model"}, inplace=True)

    df_merged = pd.merge(base_f1, full_f1, on="class_name")
    return df_merged.sort_values(by="Baseline", ascending=False).reset_index(drop=True)


def _plot_f1_panel(ax, df, panel_title):
    categories = df["class_name"]
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["Baseline"],
        width,
        label="Baseline",
        color="#4C72B0",
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        df["Full Model"],
        width,
        label="Full Model",
        color="#C44E52",
        edgecolor="black",
        linewidth=1,
    )

    ax.set_title(panel_title, fontsize=FONT_PANEL, fontweight="bold", pad=12)
    ax.set_xlabel(
        "Entity Category", fontweight="bold", labelpad=12, fontsize=FONT_AXIS_LABEL
    )
    ax.set_ylabel(
        "Peak F1 Score", fontweight="bold", labelpad=12, fontsize=FONT_AXIS_LABEL
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
    ax.set_ylim(0, 1.25)
    sns.despine(ax=ax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")

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
                fontsize=FONT_ANNOT,
                rotation=90,
            )

    annotate_bars(bars1)
    annotate_bars(bars2)


def generate_dual_class_f1_comparison(
    cmeee_baseline_path,
    cmeee_full_path,
    cluener_baseline_path,
    cluener_full_path,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": FONT_AXIS_LABEL,
            "font.size": FONT_TICK_Y,
            "legend.fontsize": FONT_LEGEND,
            "xtick.labelsize": FONT_TICK_X,
            "ytick.labelsize": FONT_TICK_Y,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    sns.set_theme(style="ticks", context="paper", font="Times New Roman")

    print(f"[CMeEE-V2] baseline: {cmeee_baseline_path}")
    print(f"[CMeEE-V2] full model: {cmeee_full_path}")
    cmeee_df = _load_peak_f1_df(cmeee_baseline_path, cmeee_full_path)

    print(f"[CLUENER] baseline: {cluener_baseline_path}")
    print(f"[CLUENER] full model: {cluener_full_path}")
    cluener_df = _load_peak_f1_df(cluener_baseline_path, cluener_full_path)

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))

    _plot_f1_panel(axes[0], cmeee_df, "CMeEE-V2")
    _plot_f1_panel(axes[1], cluener_df, "CLUENER")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=FONT_LEGEND,
        handlelength=2.8,
        handleheight=1.6,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, wspace=0.26)

    pdf_path = os.path.join(save_dir, "dual_class_f1_comparison.pdf")
    png_path = os.path.join(save_dir, "dual_class_f1_comparison.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    plt.close()

    print("\nPlot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CMeEE-V2 and CLUENER per-class peak F1 comparison on one figure."
    )
    parser.add_argument(
        "--cmeee_baseline",
        type=str,
        default=DEFAULT_CMEEE_BASELINE,
        help="CMeEE-V2 baseline class log CSV",
    )
    parser.add_argument(
        "--cmeee_full",
        type=str,
        default=DEFAULT_CMEEE_FULL,
        help="CMeEE-V2 full model class log CSV",
    )
    parser.add_argument(
        "--cluener_baseline",
        type=str,
        default=DEFAULT_CLUENER_BASELINE,
        help="CLUENER baseline class log CSV",
    )
    parser.add_argument(
        "--cluener_full",
        type=str,
        default=DEFAULT_CLUENER_FULL,
        help="CLUENER full model class log CSV",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save PDF/PNG outputs",
    )
    args = parser.parse_args()

    generate_dual_class_f1_comparison(
        cmeee_baseline_path=args.cmeee_baseline,
        cmeee_full_path=args.cmeee_full,
        cluener_baseline_path=args.cluener_baseline,
        cluener_full_path=args.cluener_full,
        save_dir=args.save_dir,
    )
