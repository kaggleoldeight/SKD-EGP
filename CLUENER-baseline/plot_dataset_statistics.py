# -*- coding: utf-8 -*-
"""
统计训练集各实体类别出现频次，并生成符合期刊出版要求的柱状图。
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

from config import DATA_DIR, TRAIN_FILE
from data_loader import load_data, id2ent


def generate_publication_ready_plot():
    """
    读取训练数据集，统计各个实体类别的出现次数，
    并生成符合出版要求的柱状图。
    """
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

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    print(f"Loading data from: {train_path}")

    data = load_data(train_path)

    category_counts = Counter()
    for item in data:
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
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.2,
        ax=ax,
    )

    ax.set_xlabel("Entity Category", fontweight="bold", labelpad=10)
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

    pdf_path = os.path.join(DATA_DIR, "train_category_distribution.pdf")
    png_path = os.path.join(DATA_DIR, "train_category_distribution.png")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")

    print("Plot generation complete.")
    print(f"Saved Vector Format (PDF): {pdf_path}")
    print(f"Saved Image Format (PNG): {png_path}")


if __name__ == "__main__":
    generate_publication_ready_plot()
