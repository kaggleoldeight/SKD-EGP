# -*- coding: utf-8 -*-
"""
统计混合训练集中「末尾增强段」的实体类别分布，生成符合期刊出版要求的柱状图。
"""
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

from config import DATA_DIR, TRAIN_FILE, OUTPUT_DIR
from data_loader import load_data, id2ent


def generate_augmented_data_plot(data_path, save_dir, num_augmented):
    """
    Reads the combined dataset, extracts the augmented portion (last N items),
    counts the occurrences of each entity category, and generates a publication-ready bar chart.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Dataset file not found at '{data_path}'")

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 13,
        "font.size": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    sns.set_theme(style="ticks", context="paper", font="Times New Roman")

    print(f"Loading combined data from: {data_path}")
    data = load_data(data_path)
    total_len = len(data)

    if num_augmented > total_len:
        print(
            f"Warning: Requested {num_augmented} augmented items, but total dataset size "
            f"is only {total_len}. Using all data."
        )
        augmented_data = data
    else:
        augmented_data = data[-num_augmented:]

    print(f"Total dataset size: {total_len}")
    print(f"Extracted augmented data size: {len(augmented_data)}")

    category_counts = Counter()
    for item in augmented_data:
        entities = item[1:]
        for ent in entities:
            label_id = ent[2]
            category_name = id2ent[label_id]
            category_counts[category_name] += 1

    df = pd.DataFrame(category_counts.items(), columns=["Category", "Count"])
    df = df.sort_values(by="Count", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x="Category",
        y="Count",
        data=df,
        color="#55A868",
        edgecolor="black",
        linewidth=1.2,
        ax=ax,
    )

    ax.set_xlabel("Entity Category (Augmented Data)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Instances", fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right")
    sns.despine()

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=10,
        )

    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "augmented_category_distribution.pdf")
    png_path = os.path.join(save_dir, "augmented_category_distribution.png")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")

    print("\nPlot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    default_data_path = os.path.join(DATA_DIR, TRAIN_FILE)

    parser = argparse.ArgumentParser(
        description="Generate dataset statistics plot for augmented data ONLY."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help=f"Path to the combined training dataset JSON (default: {default_data_path})",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save the generated plots (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=1500,
        help="Number of augmented items appended to the end of the dataset (default: 1500)",
    )

    args = parser.parse_args()

    generate_augmented_data_plot(
        data_path=args.data_path,
        save_dir=args.save_dir,
        num_augmented=args.num_aug,
    )
