from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW_RESULT_DIR = Path("indobias_pair/result/pairs/raw")
OUTPUT_CSV_PATH = Path("indobias_pair/analysis/protrope_winrate_gain_table.csv")
OUTPUT_MD_PATH = Path("indobias_pair/analysis/protrope_winrate_gain_table.md")
OUTPUT_TEX_PATH = Path("indobias_pair/analysis/protrope_winrate_gain_table.tex")

LANGUAGE_ORDER = ["ID", "JV", "SU", "MKS"]
LANGUAGE_LABELS = {
    "IndoBias_ID": "ID",
    "IndoBias_Javanese": "JV",
    "IndoBias_Sundanese": "SU",
    "IndoBias_Makassar": "MKS",
}


@dataclass(frozen=True)
class ModelPair:
    title: str
    before_file: str
    after_file: str


MODEL_PAIRS = [
    ModelPair(
        title="Llama-2 Chat -> Cendol",
        before_file="decoder_meta-llama_Llama-2-7b-chat-hf.xlsx",
        after_file="decoder_indonlp_cendol-llama2-7b-chat.xlsx",
    ),
    ModelPair(
        title="Llama-2 Base -> Komodo",
        before_file="decoder_meta-llama_Llama-2-7b-hf.xlsx",
        after_file="decoder_Yellow-AI-NLP_komodo-7b-base.xlsx",
    ),
    ModelPair(
        title="Qwen2 -> SeaLLM-v3",
        before_file="decoder_Qwen_Qwen2-7B.xlsx",
        after_file="decoder_SeaLLMs_SeaLLMs-v3-7B.xlsx",
    ),
    ModelPair(
        title="Qwen2.5 -> Sailor2",
        before_file="decoder_Qwen_Qwen2_5-7B.xlsx",
        after_file="decoder_sail_Sailor2-8B.xlsx",
    ),
]


def sanitize_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_result(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError(f"Empty result file: {file_path}")
    return df


def protrope_win_rate(df: pd.DataFrame) -> float:
    return ((df["protrope_perplexity"] < df["antitrope_perplexity"]).mean() * 100.0)


def collect_pair_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for pair in MODEL_PAIRS:
        for language_dir_name, language_code in LANGUAGE_LABELS.items():
            language_dir = RAW_RESULT_DIR / language_dir_name
            before_path = language_dir / pair.before_file
            after_path = language_dir / pair.after_file

            if not before_path.exists() or not after_path.exists():
                continue

            before_df = load_result(before_path)
            after_df = load_result(after_path)
            before_win_rate = protrope_win_rate(before_df)
            after_win_rate = protrope_win_rate(after_df)

            records.append(
                {
                    "pair_title": pair.title,
                    "language": language_code,
                    "before_win_rate": before_win_rate,
                    "after_win_rate": after_win_rate,
                    "winrate_gain": after_win_rate - before_win_rate,
                }
            )

    scores = pd.DataFrame.from_records(records)
    if scores.empty:
        raise ValueError(f"No before/after model pairs found under {RAW_RESULT_DIR}")

    scores["language"] = pd.Categorical(
        scores["language"], categories=LANGUAGE_ORDER, ordered=True
    )
    scores = scores.sort_values(["pair_title", "language"])
    return scores


def build_wide_table(scores: pd.DataFrame) -> pd.DataFrame:
    wide = (
        scores.pivot(index="pair_title", columns="language", values="winrate_gain")
        .reindex(columns=[c for c in LANGUAGE_ORDER if c in set(scores["language"].astype(str))])
        .reset_index()
    )
    language_cols = [col for col in wide.columns if col != "pair_title"]
    wide["Average"] = wide[language_cols].mean(axis=1)
    return wide


def render_markdown_table(wide: pd.DataFrame) -> str:
    formatted = wide.copy()
    for col in formatted.columns:
        if col != "pair_title":
            formatted[col] = formatted[col].map(lambda x: f"{x:+.2f}")

    header = "| Pair | " + " | ".join(col for col in formatted.columns if col != "pair_title") + " |"
    separator = "| --- | " + " | ".join("---" for _ in formatted.columns if _ != "pair_title") + " |"
    rows = [
        "| "
        + row["pair_title"]
        + " | "
        + " | ".join(str(row[col]) for col in formatted.columns if col != "pair_title")
        + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def render_latex_table(wide: pd.DataFrame) -> str:
    language_cols = [col for col in wide.columns if col != "pair_title"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(language_cols) + "}",
        r"\hline",
        r"\textbf{Pair} & " + " & ".join(rf"\textbf{{{col}}}" for col in language_cols) + r" \\",
        r"\hline",
    ]

    for _, row in wide.iterrows():
        values = " & ".join(f"{row[col]:+.2f}" for col in language_cols)
        lines.append(rf"{sanitize_latex(row['pair_title'])} & {values} \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Protrope win-rate gain after adaptation, computed as after minus before in percentage points.}",
            r"\label{tab:protrope-winrate-gain}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    scores = collect_pair_scores()
    wide = build_wide_table(scores)
    OUTPUT_CSV_PATH.write_text(wide.to_csv(index=False), encoding="utf-8")
    OUTPUT_MD_PATH.write_text(render_markdown_table(wide), encoding="utf-8")
    OUTPUT_TEX_PATH.write_text(render_latex_table(wide), encoding="utf-8")
    print(f"Saved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown table to: {OUTPUT_MD_PATH}")
    print(f"Saved LaTeX table to: {OUTPUT_TEX_PATH}")


if __name__ == "__main__":
    main()
