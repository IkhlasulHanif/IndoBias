from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_RESULT_DIR = Path("indobias_pair/result/pairs/raw")
OUTPUT_PATH = Path("indobias_pair/analysis/protrope_win_rate_tables.tex")
LANGUAGE_ORDER = ["ID", "JV", "SU", "MKS"]

DOMAIN_COLUMNS = [
    "Ideologi dan Agama",
    "Identitas dan Demografi",
    "Status Ekonomi",
    "Kultural dan Geografis",
    "Peran Sosial dan Keluarga",
]
TABLE_GROUP_COLUMNS = DOMAIN_COLUMNS

LANGUAGE_LABELS = {
    "IndoBias_ID": "Indonesia",
    "IndoBias_Javanese": "Javanese",
    "IndoBias_Makassar": "Makassar",
    "IndoBias_Sundanese": "Sundanese",
}

LANGUAGE_CODES = {
    "IndoBias_ID": "ID",
    "IndoBias_Javanese": "JV",
    "IndoBias_Sundanese": "SU",
    "IndoBias_Makassar": "MKS",
}

MODEL_GROUPS = {
    "decoder": [
        (
            "General Models",
            [
                ("decoder_meta-llama_Llama-3_1-8B.xlsx", "Llama-3.1-8B"),
                ("decoder_meta-llama_Llama-3_1-8B-Instruct.xlsx", "Llama-3.1-8B-Instruct"),
                ("decoder_Qwen_Qwen3-8B-Base.xlsx", "Qwen3-8B-Base"),
                ("decoder_Qwen_Qwen3-8B.xlsx", "Qwen3-8B"),
            ],
        ),
        (
            "SEA Models",
            [
                ("decoder_aisingapore_Gemma-SEA-LION-v3-9B.xlsx", "SEA-LION-v3-Base"),
                ("decoder_aisingapore_Gemma-SEA-LION-v3-9B-IT.xlsx", "SEA-LION-v3-IT"),
                ("decoder_SeaLLMs_SeaLLMs-v3-7B.xlsx", "SeaLLM-v3-Base"),
                ("decoder_SeaLLMs_SeaLLMs-v3-7B-Chat.xlsx", "SeaLLM-v3-Chat"),
                ("decoder_sail_Sailor2-8B.xlsx", "Sailor2-8B"),
                ("decoder_sail_Sailor2-8B-Chat.xlsx", "Sailor2-8B-Chat"),
            ],
        ),
        (
            "Indonesian Models",
            [
                ("decoder_Yellow-AI-NLP_komodo-7b-base.xlsx", "Komodo-7B"),
                ("decoder_Sahabat-AI_gemma2-9b-cpt-sahabatai-v1-base.xlsx", "SahabatAI-Base"),
                ("decoder_Sahabat-AI_gemma2-9b-cpt-sahabatai-v1-instruct.xlsx", "SahabatAI-Instruct"),
                ("decoder_Ichsan2895_Merak-7B-v4.xlsx", "Merak-7B-v4"),
                ("decoder_allenai_Olmo-3-1025-7B.xlsx", "Olmo-3-1025-7B"),
                ("decoder_allenai_Olmo-3-7B-Instruct.xlsx", "Olmo-3-7B-Instruct"),
                ("decoder_google_gemma-3-4b-it.xlsx", "Gemma-3-4B-IT"),
                ("decoder_google_gemma-3-4b-pt.xlsx", "Gemma-3-4B-PT"),
                ("decoder_indonlp_cendol-llama2-7b-chat.xlsx", "Cendol-Llama2-7B-Chat"),
            ],
        ),
    ],
    "encoder": [
        (
            "Multilingual Models",
            [
                ("encoder_google-bert_bert-base-multilingual-cased.xlsx", "mBERT"),
                ("encoder_FacebookAI_xlm-roberta-base.xlsx", "XLM-R-Base"),
            ],
        ),
        (
            "Indonesian Models",
            [
                ("encoder_indolem_indobert-base-uncased.xlsx", "IndoBERT-Base"),
                ("encoder_indolem_indobertweet-base-uncased.xlsx", "IndoBERTweet-Base"),
                (
                    (
                        "encoder_intfloat_multilingual-e5-base.xlsx",
                        "encoder_intfloat_multilingual-e5-large.xlsx",
                    ),
                    "Multilingual-E5",
                ),
            ],
        ),
    ],
}


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


def resolve_model_file(language_key: str, filename_spec) -> Path | None:
    language_dir = RAW_RESULT_DIR / language_key
    if isinstance(filename_spec, tuple):
        for filename in filename_spec:
            candidate = language_dir / filename
            if candidate.exists():
                return candidate
        return None
    candidate = language_dir / filename_spec
    return candidate if candidate.exists() else None


def compute_win_rates(file_path: Path) -> dict[str, float]:
    df = pd.read_excel(file_path)
    df["protrope_win"] = df["protrope_perplexity"] < df["antitrope_perplexity"]

    domain_scores = (
        df.groupby("domain")["protrope_win"]
        .mean()
        .mul(100)
        .to_dict()
    )

    return {
        domain: domain_scores.get(domain, float("nan"))
        for domain in DOMAIN_COLUMNS
    } | {"Average": df["protrope_win"].mean() * 100}


def empty_score_map() -> dict[str, float]:
    return {
        f"{domain}|{language_code}": float("nan")
        for domain in TABLE_GROUP_COLUMNS
        for language_code in LANGUAGE_ORDER
    }


def collect_rows(model_type: str) -> tuple[list[tuple[str, list[tuple[str, dict[str, float]]]]], list[dict[str, float]]]:
    grouped_rows: list[tuple[str, list[tuple[str, dict[str, float]]]]] = []
    flat_rows: list[dict[str, float]] = []

    for group_name, model_specs in MODEL_GROUPS[model_type]:
        rows: list[tuple[str, dict[str, float]]] = []
        for filename, display_name in model_specs:
            score_map = empty_score_map()
            found = False
            for language_key, language_label in LANGUAGE_LABELS.items():
                file_path = resolve_model_file(language_key, filename)
                if file_path is None:
                    continue
                scores = compute_win_rates(file_path)
                language_code = LANGUAGE_CODES[language_key]
                for column_name in TABLE_GROUP_COLUMNS:
                    value = scores[column_name]
                    score_map[f"{column_name}|{language_code}"] = value
                found = True
            if found:
                rows.append((display_name, score_map))
                flat_rows.append(score_map)
        if rows:
            grouped_rows.append((group_name, rows))

    return grouped_rows, flat_rows


def bold_value(value: float, should_bold: bool) -> str:
    formatted = f"{value:.1f}"
    if pd.isna(value):
        return "--"
    if should_bold:
        return rf"\textbf{{{formatted}}}"
    return formatted


def highest_value_keys(scores: dict[str, float], group_name: str) -> set[str]:
    keys = [f"{group_name}|{language_code}" for language_code in LANGUAGE_ORDER]
    valid = [(key, scores[key]) for key in keys if not pd.isna(scores[key])]
    if not valid:
        return set()

    best_value = max(value for _, value in valid)
    return {
        key
        for key, value in valid
        if abs(value - best_value) < 1e-9
    }


def render_table(model_type: str) -> str:
    grouped_rows, flat_rows = collect_rows(model_type)
    if not grouped_rows:
        return ""

    model_label = "decoder" if model_type == "decoder" else "encoder"
    caption = f"Protrope Win Rate across domains for {model_label} models across languages."
    table_label = f"tab:{model_type}-all-languages"

    group_columns = TABLE_GROUP_COLUMNS
    column_spec = "l|" + "|".join("c" * len(LANGUAGE_ORDER) for _ in group_columns)

    lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + column_spec + "}",
        r"\hline",
        (
            r"\textbf{Model} & "
            + " & ".join(
                rf"\multicolumn{{4}}{{c}}{{\textbf{{{sanitize_latex(column)}}}}}"
                for column in group_columns
            )
            + r" \\"
        ),
        (
            " "
            + " & ".join(
                [""]
                + LANGUAGE_ORDER * len(group_columns)
            )
            + r" \\"
        ),
        r"\hline",
    ]

    for group_name, rows in grouped_rows:
        lines.append(
            rf"\multicolumn{{{1 + len(LANGUAGE_ORDER) * len(group_columns)}}}{{c}}{{\textit{{{sanitize_latex(group_name)}}}}} \\"
        )
        lines.append(r"\hline")
        for display_name, scores in rows:
            bold_keys = set()
            for column_group in group_columns:
                bold_keys.update(highest_value_keys(scores, column_group))

            rendered_scores = []
            for domain in group_columns:
                for language_code in LANGUAGE_ORDER:
                    key = f"{domain}|{language_code}"
                    rendered_scores.append(bold_value(scores[key], key in bold_keys))
            row = " & ".join(
                [sanitize_latex(display_name), *rendered_scores]
            ) + r" \\"
            lines.append(row)
        lines.append(r"\hline")

    lines.extend(
        [
            r"\end{tabular}",
            r"}%",
            rf"\caption{{{sanitize_latex(caption)}}}",
            rf"\label{{{table_label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    tables: list[str] = []
    for model_type in ("decoder", "encoder"):
        table = render_table(model_type)
        if table:
            tables.append(table)

    OUTPUT_PATH.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Wrote {len(tables)} tables to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
