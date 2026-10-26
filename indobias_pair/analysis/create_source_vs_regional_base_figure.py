from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from source_vs_regional_base_common import (
    BASE_PAIRS,
    LANGUAGE_COLORS,
    LANGUAGE_LABELS,
    LANGUAGE_ORDER,
    PROJECT_DIR,
    RAW_RESULT_DIR,
)

OUTPUT_PATH = PROJECT_DIR / "analysis/source_vs_regional_base_models.png"


def protrope_win_rate(file_path: Path) -> float:
    df = pd.read_excel(file_path)
    return (df["protrope_perplexity"] < df["antitrope_perplexity"]).mean() * 100.0


def collect_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pair in BASE_PAIRS:
        for language_dir_name, language_code in LANGUAGE_LABELS.items():
            source_path = RAW_RESULT_DIR / language_dir_name / pair.source_file
            regional_path = RAW_RESULT_DIR / language_dir_name / pair.regional_file
            if not source_path.exists() or not regional_path.exists():
                continue
            records.append(
                {
                    "family": pair.family,
                    "language": language_code,
                    "source_label": pair.source_label,
                    "regional_label": pair.regional_label,
                    "source_win_rate": protrope_win_rate(source_path),
                    "regional_win_rate": protrope_win_rate(regional_path),
                }
            )
    scores = pd.DataFrame.from_records(records)
    scores["language"] = pd.Categorical(
        scores["language"], categories=LANGUAGE_ORDER, ordered=True
    )
    return scores.sort_values(["family", "source_label", "language"]).reset_index(
        drop=True
    )


def plot_family(ax: plt.Axes, family_scores: pd.DataFrame, title: str) -> None:
    pairs = (
        family_scores[["source_label", "regional_label"]]
        .drop_duplicates()
        .values.tolist()
    )
    x_positions = []
    x_labels = []

    for idx, (source_label, regional_label) in enumerate(pairs):
        pair_df = family_scores[
            (family_scores["source_label"] == source_label)
            & (family_scores["regional_label"] == regional_label)
        ]
        x0 = idx * 3
        x1 = x0 + 1
        x_positions.extend([x0, x1])
        x_labels.extend([source_label, regional_label])

        for row in pair_df.itertuples():
            color = LANGUAGE_COLORS[str(row.language)]
            ax.plot(
                [x0, x1],
                [row.source_win_rate, row.regional_win_rate],
                color=color,
                linewidth=1.8,
                alpha=0.45,
                zorder=2,
            )
            ax.scatter(
                [x0, x1],
                [row.source_win_rate, row.regional_win_rate],
                color=color,
                s=38,
                alpha=0.8,
                zorder=3,
            )

        mean_source = pair_df["source_win_rate"].mean()
        mean_regional = pair_df["regional_win_rate"].mean()
        ax.plot(
            [x0, x1],
            [mean_source, mean_regional],
            color="#111827",
            linewidth=3.2,
            marker="o",
            markersize=7,
            zorder=4,
        )
        ax.text(
            (x0 + x1) / 2,
            max(mean_source, mean_regional) + 1.5,
            f"{mean_regional - mean_source:+.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111827",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylim(45, 75)
    ax.set_ylabel("Protrope Win Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.2)


def main() -> None:
    scores = collect_scores()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    plot_family(
        axes[0], scores[scores["family"] == "Indonesian"], "Indonesian Base Models"
    )
    plot_family(axes[1], scores[scores["family"] == "SEA"], "SEA Base Models")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LANGUAGE_COLORS[lang],
            linewidth=2.2,
            marker="o",
            markersize=6,
            label=lang,
        )
        for lang in LANGUAGE_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=3.2,
            marker="o",
            markersize=7,
            label="Mean",
        )
    )

    fig.suptitle(
        "Source Base Models vs Regional Base Models", fontsize=15, fontweight="bold"
    )
    fig.text(
        0.5,
        0.02,
        "Only explicit base-to-base adaptations are included; Cendol is excluded because it is chat-tuned.",
        ha="center",
        fontsize=10,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
