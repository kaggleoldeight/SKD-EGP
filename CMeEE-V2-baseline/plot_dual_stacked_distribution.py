# -*- coding: utf-8 -*-
"""
将 CMeEE-V2 与 CLUENER 增强后混合训练集的实体类别分布（原始 + 增强堆叠）
绘制在同一张图中，便于论文对比展示。
"""
import argparse
import importlib
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR, PROJECT_DIR

# 默认路径（相对「实验2-消融实验」目录）
ABLATION_ROOT = os.path.dirname(PROJECT_DIR)
CLUENER_PROJECT = os.path.join(ABLATION_ROOT, "完整模型 - CLUENER - 完整")

DEFAULT_CMEEE_MIXED = os.path.join(
    PROJECT_DIR, "datasets", "CMeEE-V2_train_mixed_1500.json"
)
DEFAULT_CLUENER_MIXED = os.path.join(
    CLUENER_PROJECT, "datasets", "train_mixed_1000.json"
)
DEFAULT_CLUENER_TRAIN = os.path.join(CLUENER_PROJECT, "datasets", "train.json")

# 较大字号，保证图中文字清晰可见
FONT_AXIS_LABEL = 21
FONT_TICK_X = 20
FONT_TICK_Y = 19
FONT_LEGEND = 19
FONT_ANNOT = 17
FONT_PANEL = 20


def _load_project_data(project_dir, data_path):
    """按项目目录隔离导入 data_loader，避免 CMeEE / CLUENER 标签映射冲突。"""
    project_dir = os.path.abspath(project_dir)
    saved_path = sys.path.copy()
    saved_modules = {}
    for mod_name in ("config", "data_loader"):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    sys.path.insert(0, project_dir)
    try:
        data_loader = importlib.import_module("data_loader")
        return data_loader.load_data(data_path), data_loader.id2ent
    finally:
        sys.path = saved_path
        for mod_name in ("config", "data_loader"):
            sys.modules.pop(mod_name, None)
        for mod_name, mod in saved_modules.items():
            sys.modules[mod_name] = mod


def _infer_num_augmented(mixed_path, train_path, project_dir):
    mixed = _load_project_data(project_dir, mixed_path)[0]
    train = _load_project_data(project_dir, train_path)[0]
    return len(mixed) - len(train)


def _build_stacked_dataframe(data, num_augmented, id2ent):
    total_len = len(data)
    if num_augmented >= total_len:
        original_data, augmented_data = [], data
    elif num_augmented <= 0:
        original_data, augmented_data = data, []
    else:
        original_data = data[:-num_augmented]
        augmented_data = data[-num_augmented:]

    orig_counts, aug_counts = Counter(), Counter()
    for item in original_data:
        for ent in item[1:]:
            orig_counts[id2ent[ent[2]]] += 1
    for item in augmented_data:
        for ent in item[1:]:
            aug_counts[id2ent[ent[2]]] += 1

    all_categories = sorted(set(id2ent.values()))
    records = []
    for cat in all_categories:
        orig_val = orig_counts.get(cat, 0)
        aug_val = aug_counts.get(cat, 0)
        records.append(
            {
                "Category": cat,
                "Original": orig_val,
                "Augmented": aug_val,
                "Total": orig_val + aug_val,
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values(by="Total", ascending=False).reset_index(drop=True), len(
        original_data
    ), len(augmented_data)


def _plot_stacked_panel(ax, df, panel_title):
    categories = df["Category"]
    orig_vals = df["Original"].values
    aug_vals = df["Augmented"].values
    x = np.arange(len(categories))

    ax.bar(
        x,
        orig_vals,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.2,
        label="Original Data",
    )
    ax.bar(
        x,
        aug_vals,
        bottom=orig_vals,
        color="#55A868",
        edgecolor="black",
        linewidth=1.2,
        label="Augmented Data",
    )

    ax.set_title(panel_title, fontsize=FONT_PANEL, fontweight="bold", pad=12)
    ax.set_xlabel("Entity Category", fontweight="bold", labelpad=10, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Number of Instances", fontweight="bold", labelpad=10, fontsize=FONT_AXIS_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)

    ymax = max(df["Total"].max() * 1.18, 1)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")
    sns.despine(ax=ax)

    for i, total in enumerate(df["Total"]):
        ax.annotate(
            f"{int(total)}",
            (i, total),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=FONT_ANNOT,
        )


def generate_dual_stacked_plot(
    cmeee_mixed_path,
    cmeee_num_aug,
    cluener_mixed_path,
    cluener_num_aug,
    cluener_project_dir,
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

    print(f"[CMeEE-V2] Loading: {cmeee_mixed_path}")
    cmeee_data, cmeee_id2ent = _load_project_data(PROJECT_DIR, cmeee_mixed_path)
    cmeee_df, cmeee_orig_n, cmeee_aug_n = _build_stacked_dataframe(
        cmeee_data, cmeee_num_aug, cmeee_id2ent
    )
    print(
        f"  total={len(cmeee_data)}, original={cmeee_orig_n}, augmented={cmeee_aug_n}"
    )

    print(f"[CLUENER] Loading: {cluener_mixed_path}")
    cluener_data, cluener_id2ent = _load_project_data(
        cluener_project_dir, cluener_mixed_path
    )
    cluener_df, cluener_orig_n, cluener_aug_n = _build_stacked_dataframe(
        cluener_data, cluener_num_aug, cluener_id2ent
    )
    print(
        f"  total={len(cluener_data)}, original={cluener_orig_n}, augmented={cluener_aug_n}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    _plot_stacked_panel(axes[0], cmeee_df, "CMeEE-V2")
    _plot_stacked_panel(axes[1], cluener_df, "CLUENER")

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

    pdf_path = os.path.join(save_dir, "dual_stacked_category_distribution.pdf")
    png_path = os.path.join(save_dir, "dual_stacked_category_distribution.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    plt.close()

    print("\nPlot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CMeEE-V2 and CLUENER stacked category distributions on one figure."
    )
    parser.add_argument(
        "--cmeee_mixed",
        type=str,
        default=DEFAULT_CMEEE_MIXED,
        help="CMeEE-V2 mixed training JSON path",
    )
    parser.add_argument(
        "--cmeee_num_aug",
        type=int,
        default=1500,
        help="Number of augmented samples appended in CMeEE-V2 mixed file (default: 1500)",
    )
    parser.add_argument(
        "--cluener_mixed",
        type=str,
        default=DEFAULT_CLUENER_MIXED,
        help="CLUENER mixed training JSON path",
    )
    parser.add_argument(
        "--cluener_train",
        type=str,
        default=DEFAULT_CLUENER_TRAIN,
        help="CLUENER original train.json (for inferring num_aug when --cluener_num_aug=0)",
    )
    parser.add_argument(
        "--cluener_num_aug",
        type=int,
        default=0,
        help="Augmented sample count in CLUENER mixed file; 0 = auto from mixed - train",
    )
    parser.add_argument(
        "--cluener_project",
        type=str,
        default=CLUENER_PROJECT,
        help="CLUENER project root (for data_loader import)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save PDF/PNG outputs",
    )
    args = parser.parse_args()

    cluener_num_aug = args.cluener_num_aug
    if cluener_num_aug <= 0:
        cluener_num_aug = _infer_num_augmented(
            args.cluener_mixed, args.cluener_train, args.cluener_project
        )
        print(f"[CLUENER] Auto-inferred num_aug = {cluener_num_aug}")

    generate_dual_stacked_plot(
        cmeee_mixed_path=args.cmeee_mixed,
        cmeee_num_aug=args.cmeee_num_aug,
        cluener_mixed_path=args.cluener_mixed,
        cluener_num_aug=cluener_num_aug,
        cluener_project_dir=args.cluener_project,
        save_dir=args.save_dir,
    )
