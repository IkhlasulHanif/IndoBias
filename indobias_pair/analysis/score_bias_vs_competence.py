from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_RESULT_DIR = Path("indobias_pair/result/pairs/raw")
OUTPUT_PATH = Path("indobias_pair/analysis/bias_vs_competence_scores.csv")

# Keep this list aligned with plot_winrate_vs_perplexity.py when you want a curated subset.
SELECTED_MODELS: list[str] | None = [
    "decoder_meta-llama_Llama-3_1-8B.xlsx",
    "decoder_meta-llama_Llama-3_1-8B-Instruct.xlsx",
    "decoder_Qwen_Qwen3-8B-Base.xlsx",
    "decoder_Qwen_Qwen3-8B.xlsx",
    "decoder_allenai_Olmo-3-1025-7B.xlsx",
    "decoder_allenai_Olmo-3-7B-Instruct.xlsx",
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

            win_rate = (
                (df["protrope_perplexity"] < df["antitrope_perplexity"])
                .mean()
                * 100
            )
            bias_strength = abs(win_rate - 50)
            avg_pair_perplexity = (
                (df["protrope_perplexity"] + df["antitrope_perplexity"]) / 2
            ).mean()

            records.append(
                {
                    "language": language_code,
                    "file_path": str(file_path),
                    "label": display_name_from_path(file_path),
                    "win_rate": win_rate,
                    "bias_strength": bias_strength,
                    "avg_pair_perplexity": avg_pair_perplexity,
                }
            )

    scores = pd.DataFrame.from_records(records)
    if scores.empty:
        raise ValueError(f"No result files found under {RAW_RESULT_DIR}")

    id_anchors = (
        scores[scores["language"] == "ID"][["label", "avg_pair_perplexity", "bias_strength"]]
        .drop_duplicates(subset=["label"])
        .rename(
            columns={
                "avg_pair_perplexity": "id_anchor_ppl",
                "bias_strength": "id_bias_strength",
            }
        )
    )
    scores = scores.merge(id_anchors, on="label", how="left")
    scores["competence_ratio_vs_id"] = scores["avg_pair_perplexity"] / scores["id_anchor_ppl"]
    scores["competence_drop_vs_id"] = scores["competence_ratio_vs_id"] - 1.0
    scores["bias_change_vs_id"] = scores["bias_strength"] - scores["id_bias_strength"]

    # Main suspicion score:
    # high when bias gets smaller than ID while perplexity gets worse than ID.
    scores["suspicion_score"] = (
        scores["competence_drop_vs_id"].clip(lower=0) *
        (-scores["bias_change_vs_id"]).clip(lower=0)
    )

    # Counter-score:
    # high when bias is low without worse-than-ID perplexity.
    scores["robust_low_bias_score"] = (
        (50 - scores["bias_strength"]).clip(lower=0) /
        scores["competence_ratio_vs_id"].clip(lower=1.0)
    )

    return scores.sort_values(
        ["language", "suspicion_score", "robust_low_bias_score"],
        ascending=[True, False, False],
    )


def main() -> None:
    scores = collect_scores()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_PATH, index=False)

    columns = [
        "language",
        "label",
        "bias_strength",
        "competence_ratio_vs_id",
        "bias_change_vs_id",
        "suspicion_score",
        "robust_low_bias_score",
    ]
    print(scores[columns].round(4).to_string(index=False))
    print(f"\nSaved scores to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
