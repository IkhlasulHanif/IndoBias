from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RAW_RESULT_DIR = Path("result/pairs/raw")
OUTPUT_PATH = Path("analysis/before_after_finetuning_models.png")

LANGUAGE_ORDER = ["ID", "JV", "SU", "MKS"]
LANGUAGE_LABELS = {
    "IndoBias_ID": "ID",
    "IndoBias_Javanese": "JV",
    "IndoBias_Sundanese": "SU",
    "IndoBias_Makassar": "MKS",
}

LANGUAGE_COLORS = {
    "ID": "#2563eb",
    "JV": "#dc2626",
    "SU": "#16a34a",
    "MKS": "#d97706",
}


@dataclass(frozen=True)
class ModelPair:
    family: str
    title: str
    before_label: str
    before_file: str
    after_label: str
    after_file: str


MODEL_PAIRS = [
    ModelPair(
        family="Indonesian Models",
        title="Llama-2 Chat -> Cendol",
        before_label="Llama-2-7B-Chat",
        before_file="decoder_meta-llama_Llama-2-7b-chat-hf.xlsx",
        after_label="Cendol-Llama2-7B-Chat",
        after_file="decoder_indonlp_cendol-llama2-7b-chat.xlsx",
    ),
    ModelPair(
        family="Indonesian Models",
        title="Llama-2 Base -> Komodo",
        before_label="Llama-2-7B",
        before_file="decoder_meta-llama_Llama-2-7b-hf.xlsx",
        after_label="Komodo-7B",
        after_file="decoder_Yellow-AI-NLP_komodo-7b-base.xlsx",
    ),
    ModelPair(
        family="SEA Models",
        title="Qwen2 -> SeaLLM-v3",
        before_label="Qwen2-7B",
        before_file="decoder_Qwen_Qwen2-7B.xlsx",
        after_label="SeaLLM-v3-7B",
        after_file="decoder_SeaLLMs_SeaLLMs-v3-7B.xlsx",
    ),
    ModelPair(
        family="SEA Models",
        title="Qwen2.5 -> Sailor2",
        before_label="Qwen2.5-7B",
        before_file="decoder_Qwen_Qwen2_5-7B.xlsx",
        after_label="Sailor2-8B",
        after_file="decoder_sail_Sailor2-8B.xlsx",
    ),
]


def protrope_win_rate(file_path: Path) -> float:
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError(f"Empty result file: {file_path}")
    return ((df["protrope_perplexity"] < df["antitrope_perplexity"]).mean() * 100.0)


def collect_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for pair in MODEL_PAIRS:
        for language_dir_name, language_code in LANGUAGE_LABELS.items():
            language_dir = RAW_RESULT_DIR / language_dir_name
            before_path = language_dir / pair.before_file
            after_path = language_dir / pair.after_file

            if not before_path.exists() or not after_path.exists():
                continue

            records.append(
                {
                    "family": pair.family,
                    "pair_title": pair.title,
                    "language": language_code,
                    "before_label": pair.before_label,
                    "after_label": pair.after_label,
                    "before_win_rate": protrope_win_rate(before_path),
                    "after_win_rate": protrope_win_rate(after_path),
                }
            )

    scores = pd.DataFrame.from_records(records)
    if scores.empty:
        raise ValueError(f"No before/after result pairs found under {RAW_RESULT_DIR}")

    scores["language"] = pd.Categorical(
        scores["language"], categories=LANGUAGE_ORDER, ordered=True
    )
    return scores.sort_values(["family", "pair_title", "language"]).reset_index(drop=True)


def draw_family_panel(ax: plt.Axes, family_scores: pd.DataFrame) -> None:
    ordered_pairs = family_scores["pair_title"].drop_duplicates().tolist()
    x_positions = []
    tick_labels = []

    for idx, pair_title in enumerate(ordered_pairs):
        pair_df = family_scores[family_scores["pair_title"] == pair_title]
        before_x = idx * 3
        after_x = before_x + 1

        x_positions.extend([before_x, after_x])
        tick_labels.extend(
            [
                pair_df["before_label"].iloc[0],
                pair_df["after_label"].iloc[0],
            ]
        )

        for row in pair_df.itertuples():
            color = LANGUAGE_COLORS[str(row.language)]
            ax.plot(
                [before_x, after_x],
                [row.before_win_rate, row.after_win_rate],
                color=color,
                linewidth=1.8,
                alpha=0.45,
                zorder=2,
            )
            ax.scatter(
                [before_x, after_x],
                [row.before_win_rate, row.after_win_rate],
                color=color,
                s=38,
                alpha=0.8,
                zorder=3,
            )

        mean_before = pair_df["before_win_rate"].mean()
        mean_after = pair_df["after_win_rate"].mean()
        ax.plot(
            [before_x, after_x],
            [mean_before, mean_after],
            color="#111827",
            linewidth=3.2,
            marker="o",
            markersize=7,
            zorder=4,
        )
        ax.text(
            (before_x + after_x) / 2,
            max(mean_before, mean_after) + 1.6,
            f"{mean_after - mean_before:+.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=14, ha="right")
    ax.set_ylim(45, 75)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax.set_ylabel("Protrope Win Rate (%)")


def plot_scores(scores: pd.DataFrame) -> None:
    family_order = ["Indonesian Models", "SEA Models"]
    families = [family for family in family_order if family in set(scores["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(15, 6), sharey=True)

    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        family_scores = scores[scores["family"] == family]
        draw_family_panel(ax, family_scores)
        ax.set_title(family, fontsize=13, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=LANGUAGE_COLORS[lang], linewidth=2.2, marker="o", markersize=6, label=lang)
        for lang in LANGUAGE_ORDER
        if lang in set(scores["language"].astype(str))
    ]
    legend_handles.append(
        Line2D([0], [0], color="#111827", linewidth=3.2, marker="o", markersize=7, label="Mean")
    )

    fig.suptitle("Before vs After Fine-Tuning on IndoBias Pair Win Rate", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Thin colored lines show each language; black lines show the mean across ID, JV, SU, and MKS.",
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


def main() -> None:
    scores = collect_scores()
    plot_scores(scores)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
