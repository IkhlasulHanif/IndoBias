from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RAW_RESULT_DIR = Path("indobias_pair/result/pairs/raw")
OUTPUT_PATH = Path("indobias_pair/analysis/winrate_vs_perplexity_scatter.png")

# Pick the decoder result files you want to include in the plot.
# Set to None to include every decoder file found under RAW_RESULT_DIR.
SELECTED_MODELS: list[str] | None = [
    "decoder_meta-llama_Llama-3_1-8B.xlsx",
    "decoder_meta-llama_Llama-3_1-8B-Instruct.xlsx",
    "decoder_Qwen_Qwen3-8B-Base.xlsx",
    "decoder_Qwen_Qwen3-8B.xlsx",
    "decoder_allenai_Olmo-3-1025-7B.xlsx",
    "decoder_allenai_Olmo-3-7B-Instruct.xlsx",
    "decoder_google_gemma-3-4b-pt.xlsx",
    "decoder_google_gemma-3-4b-it.xlsx",
    "decoder_aisingapore_Gemma-SEA-LION-v3-9B.xlsx",
    "decoder_aisingapore_Gemma-SEA-LION-v3-9B-IT.xlsx",
    "decoder_SeaLLMs_SeaLLMs-v3-7B.xlsx",
    "decoder_SeaLLMs_SeaLLMs-v3-7B-Chat.xlsx",
    "decoder_sail_Sailor2-8B.xlsx",
    "decoder_sail_Sailor2-8B-Chat.xlsx",
    "decoder_Yellow-AI-NLP_komodo-7b-base.xlsx",
    "decoder_Sahabat-AI_gemma2-9b-cpt-sahabatai-v1-base.xlsx",
    "decoder_Sahabat-AI_gemma2-9b-cpt-sahabatai-v1-instruct.xlsx",
    "decoder_Ichsan2895_Merak-7B-v4.xlsx",
    "decoder_indonlp_cendol-llama2-7b-chat.xlsx",
]

LANGUAGE_LABELS = {
    "IndoBias_ID": "ID",
    "IndoBias_Javanese": "JV",
    "IndoBias_Makassar": "MKS",
    "IndoBias_Sundanese": "SU",
}

def display_name_from_path(file_path: Path) -> str:
    stem = file_path.stem
    model_type, rest = stem.split("_", 1)
    model_name = rest.replace("_", "/", 1).replace("_", ".")
    return f"{model_type}:{model_name}"


def display_name_from_filename(filename: str) -> str:
    return display_name_from_path(Path(filename))


def collect_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for language_dir in sorted(RAW_RESULT_DIR.iterdir()):
        if not language_dir.is_dir():
            continue

        language_code = LANGUAGE_LABELS.get(language_dir.name, language_dir.name)

        for file_path in sorted(language_dir.glob("*.xlsx")):
            if not file_path.name.startswith("decoder_"):
                continue
            if SELECTED_MODELS is not None and file_path.name not in SELECTED_MODELS:
                continue

            df = pd.read_excel(file_path)
            if df.empty:
                continue
            if (
                (df["protrope_perplexity"] <= 0).any()
                or (df["antitrope_perplexity"] <= 0).any()
            ):
                continue

            win_rate = (
                (df["protrope_perplexity"] < df["antitrope_perplexity"])
                .mean()
                * 100
            )
            bias_strength = abs(win_rate - 50)
            avg_pair_perplexity = (
                (df["protrope_perplexity"] + df["antitrope_perplexity"]) / 2
            ).mean()

            label = display_name_from_path(file_path)

            records.append(
                {
                    "language": language_code,
                    "file_path": str(file_path),
                    "label": label,
                    "win_rate": win_rate,
                    "bias_strength": bias_strength,
                    "avg_pair_perplexity": avg_pair_perplexity,
                }
            )

    scores = pd.DataFrame.from_records(records)
    if scores.empty:
        raise ValueError(f"No result files found under {RAW_RESULT_DIR}")

    id_anchors = (
        scores[scores["language"] == "ID"][["label", "avg_pair_perplexity"]]
        .drop_duplicates(subset=["label"])
        .rename(columns={"avg_pair_perplexity": "id_anchor"})
    )
    scores = scores.merge(id_anchors, on="label", how="left")
    scores["competence_ratio_vs_id"] = (
        scores["avg_pair_perplexity"] / scores["id_anchor"]
    )

    return scores


def plot_scores(scores: pd.DataFrame) -> None:
    language_order = ["ID", "JV", "MKS", "SU"]
    languages = [lang for lang in language_order if lang in set(scores["language"])]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    default_color = "#9ca3af"
    best_color = "#16a34a"
    worst_color = "#dc2626"
    ordered_labels = (
        [display_name_from_filename(name) for name in SELECTED_MODELS]
        if SELECTED_MODELS is not None
        else sorted(scores["label"].unique())
    )

    for ax, language in zip(axes, languages):
        lang_df = scores[scores["language"] == language].copy()
        lang_df["point_color"] = default_color
        if not lang_df.empty:
            lang_df.loc[lang_df["bias_strength"].idxmax(), "point_color"] = worst_color
            lang_df.loc[lang_df["bias_strength"].idxmin(), "point_color"] = best_color

        for label in ordered_labels:
            model_df = lang_df[lang_df["label"] == label]
            if model_df.empty:
                continue

            scatter = ax.scatter(
                model_df["competence_ratio_vs_id"],
                model_df["bias_strength"],
                s=90,
                alpha=0.85,
                color=model_df["point_color"],
            )

            for _, row in model_df.iterrows():
                ax.annotate(
                    row["label"].split(":", 1)[1],
                    (
                        row["competence_ratio_vs_id"],
                        row["bias_strength"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.9,
                )

        ax.set_title(language)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlabel("Mean Pair Perplexity / ID Anchor")
        ax.set_ylabel("|Protrope Win Rate - 50|")

    for ax in axes[len(languages):]:
        ax.axis("off")

    fig.suptitle("Bias Strength vs ID-Anchored Language Competence by Language", fontsize=14)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=worst_color, markersize=8, label="Worst"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=best_color, markersize=8, label="Best"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=default_color, markersize=8, label="Other"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        title="Per Language",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scores = collect_scores()
    plot_scores(scores)
    print(f"Plotted {len(scores)} result files across {scores['language'].nunique()} languages.")
    print(f"Saved scatter plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
