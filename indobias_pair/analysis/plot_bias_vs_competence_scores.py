from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_PATH = Path("indobias_pair/analysis/bias_vs_competence_scores.csv")
OUTPUT_PATH = Path("indobias_pair/analysis/bias_vs_competence_scores_heatmap.png")
LANGUAGE_ORDER = ["ID", "JV", "MKS", "SU"]


def short_label(label: str) -> str:
    return label.split(":", 1)[1]


def build_matrix(df: pd.DataFrame, value_col: str, model_order: list[str]) -> pd.DataFrame:
    matrix = (
        df.assign(model_short=df["label"].map(short_label))
        .pivot(index="model_short", columns="language", values=value_col)
        .reindex(index=model_order, columns=LANGUAGE_ORDER)
    )
    return matrix


def draw_heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str) -> None:
    image = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            text = "--" if pd.isna(value) else f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    model_order = (
        df[df["language"] != "ID"]
        .groupby("label")["suspicion_score"]
        .max()
        .sort_values(ascending=False)
        .index.map(lambda label: label.split(":", 1)[1])
        .tolist()
    )

    suspicion = build_matrix(df, "suspicion_score", model_order)
    robust = build_matrix(df, "robust_low_bias_score", model_order)
    bias = build_matrix(df, "bias_strength", model_order)
    competence = build_matrix(df, "competence_ratio_vs_id", model_order)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    draw_heatmap(axes[0, 0], suspicion, "Suspicion Score", "Reds")
    draw_heatmap(axes[0, 1], robust, "Robust Low-Bias Score", "Greens")
    draw_heatmap(axes[1, 0], bias, "|Protrope Win Rate - 50|", "Blues")
    draw_heatmap(axes[1, 1], competence, "Mean Pair Perplexity / ID", "Purples")

    fig.suptitle("Bias vs Competence Score Summary", fontsize=14)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
