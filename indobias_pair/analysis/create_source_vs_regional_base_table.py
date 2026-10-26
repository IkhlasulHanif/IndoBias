from __future__ import annotations

from math import comb
from pathlib import Path

import pandas as pd

from source_vs_regional_base_common import (
    BASE_PAIRS,
    LANGUAGE_LABELS,
    LANGUAGE_ORDER,
    PROJECT_DIR,
    RAW_RESULT_DIR,
)


OUTPUT_CSV_PATH = PROJECT_DIR / "analysis/source_vs_regional_base_delta_table.csv"
OUTPUT_MD_PATH = PROJECT_DIR / "analysis/source_vs_regional_base_delta_table.md"
OUTPUT_MCNEMAR_CSV_PATH = (
    PROJECT_DIR / "analysis/source_vs_regional_base_mcnemar_table.csv"
)
OUTPUT_MCNEMAR_MD_PATH = (
    PROJECT_DIR / "analysis/source_vs_regional_base_mcnemar_table.md"
)
OUTPUT_MCNEMAR_TEX_PATH = (
    PROJECT_DIR / "analysis/source_vs_regional_base_mcnemar_table.tex"
)

ALIGNMENT_COLUMNS = ["protrope", "antitrope", "dimension", "domain"]


def exact_mcnemar_p(non_pro_to_pro: int, pro_to_non_pro: int) -> float:
    discordant = non_pro_to_pro + pro_to_non_pro
    if discordant == 0:
        return 1.0

    lower_tail = sum(
        comb(discordant, k) for k in range(min(non_pro_to_pro, pro_to_non_pro) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def load_aligned_pair(
    source_path: Path, regional_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.read_excel(source_path)
    regional_df = pd.read_excel(regional_path)
    if len(source_df) != len(regional_df):
        raise ValueError(
            f"Row-count mismatch for {source_path.name} and {regional_path.name}: "
            f"{len(source_df)} != {len(regional_df)}"
        )
    if not source_df[ALIGNMENT_COLUMNS].equals(regional_df[ALIGNMENT_COLUMNS]):
        raise ValueError(
            f"Paired rows are not aligned for {source_path} and {regional_path}"
        )
    return source_df, regional_df


def collect_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pair in BASE_PAIRS:
        pair_name = f"{pair.source_label} -> {pair.regional_label}"
        for language_dir_name, language_code in LANGUAGE_LABELS.items():
            source_path = RAW_RESULT_DIR / language_dir_name / pair.source_file
            regional_path = RAW_RESULT_DIR / language_dir_name / pair.regional_file
            if not source_path.exists() or not regional_path.exists():
                continue

            source_df, regional_df = load_aligned_pair(source_path, regional_path)
            source_win = (
                source_df["protrope_perplexity"] < source_df["antitrope_perplexity"]
            )
            regional_win = (
                regional_df["protrope_perplexity"] < regional_df["antitrope_perplexity"]
            )
            source_win_rate = source_win.mean() * 100.0
            regional_win_rate = regional_win.mean() * 100.0
            non_pro_to_pro = int((~source_win & regional_win).sum())
            pro_to_non_pro = int((source_win & ~regional_win).sum())
            records.append(
                {
                    "family": pair.family,
                    "pair": pair_name,
                    "language": language_code,
                    "n": len(source_df),
                    "source_win_rate": source_win_rate,
                    "regional_win_rate": regional_win_rate,
                    "delta": regional_win_rate - source_win_rate,
                    "non_pro_to_pro": non_pro_to_pro,
                    "pro_to_non_pro": pro_to_non_pro,
                    "discordant": non_pro_to_pro + pro_to_non_pro,
                    "mcnemar_p": exact_mcnemar_p(non_pro_to_pro, pro_to_non_pro),
                }
            )

    scores = pd.DataFrame.from_records(records)
    if scores.empty:
        raise ValueError(f"No source/regional model pairs found under {RAW_RESULT_DIR}")

    scores["significant_p_0_05"] = scores["mcnemar_p"] < 0.05
    scores["language"] = pd.Categorical(
        scores["language"], categories=LANGUAGE_ORDER, ordered=True
    )
    return scores.sort_values(["family", "pair", "language"]).reset_index(drop=True)


def build_wide_delta_table(scores: pd.DataFrame) -> pd.DataFrame:
    delta_wide = (
        scores.pivot_table(
            index=["family", "pair"],
            columns="language",
            values="delta",
            aggfunc="first",
        )
        .reindex(columns=LANGUAGE_ORDER)
        .reset_index()
    )
    delta_wide["Average"] = delta_wide[LANGUAGE_ORDER].mean(axis=1)
    return delta_wide


def build_mcnemar_summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    delta_wide = build_wide_delta_table(scores)
    significant_cells = (
        scores[scores["significant_p_0_05"]]
        .groupby(["family", "pair"], observed=True)["language"]
        .apply(lambda values: ", ".join(str(value) for value in values))
        .rename("mcnemar_diagnostics")
        .reset_index()
    )
    summary = delta_wide.merge(significant_cells, on=["family", "pair"], how="left")
    summary["mcnemar_diagnostics"] = summary["mcnemar_diagnostics"].fillna("none")
    return summary


def render_markdown_table(wide: pd.DataFrame) -> str:
    formatted = wide.copy()
    for col in LANGUAGE_ORDER + ["Average"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:+.2f}")

    header = "| Family | Pair | " + " | ".join(LANGUAGE_ORDER + ["Average"]) + " |"
    separator = (
        "| --- | --- | "
        + " | ".join("---" for _ in LANGUAGE_ORDER + ["Average"])
        + " |"
    )
    rows = [
        "| "
        + row["family"]
        + " | "
        + row["pair"]
        + " | "
        + " | ".join(str(row[col]) for col in LANGUAGE_ORDER + ["Average"])
        + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def render_mcnemar_markdown_table(scores: pd.DataFrame) -> str:
    header = (
        "| Family | Pair | ID delta | JV delta | SU delta | MKS delta | Avg delta | "
        "McNemar diagnostics |"
    )
    separator = (
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "--- |"
    )
    rows = []
    for row in scores.itertuples():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.family),
                    str(row.pair),
                    f"{row.ID:+.2f}",
                    f"{row.JV:+.2f}",
                    f"{row.SU:+.2f}",
                    f"{row.MKS:+.2f}",
                    f"{row.Average:+.2f}",
                    str(row.mcnemar_diagnostics),
                ]
            )
            + " |"
        )
    return "\n".join([header, separator, *rows]) + "\n"


def render_mcnemar_latex_table(scores: pd.DataFrame) -> str:
    def latex_pair(pair: str) -> str:
        return pair.replace(" -> ", r" $\rightarrow$ ")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrl@{}}",
        r"\toprule",
        (
            r"\textbf{Adaptation pair} & \textbf{ID $\Delta$} & "
            r"\textbf{JV $\Delta$} & \textbf{SU $\Delta$} & "
            r"\textbf{MKS $\Delta$} & \textbf{Avg. $\Delta$} & "
            r"\textbf{McNemar diagnostics} \\"
        ),
        r"\midrule",
    ]

    for row in scores.itertuples():
        lines.append(
            f"{latex_pair(str(row.pair))} & {row.ID:+.2f} & {row.JV:+.2f} & "
            f"{row.SU:+.2f} & {row.MKS:+.2f} & {row.Average:+.2f} & "
            f"{row.mcnemar_diagnostics} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            (
                r"\caption{Source-base versus regional-base protrope win-rate "
                r"deltas. The final column lists language cells where the McNemar "
                r"diagnostic reaches $p<0.05$ over the same 544 paired items.}"
            ),
            r"\label{tab:source-regional-mcnemar}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    scores = collect_scores()
    delta_wide = build_wide_delta_table(scores)
    mcnemar_summary = build_mcnemar_summary_table(scores)
    OUTPUT_CSV_PATH.write_text(delta_wide.to_csv(index=False), encoding="utf-8")
    OUTPUT_MD_PATH.write_text(render_markdown_table(delta_wide), encoding="utf-8")
    mcnemar_summary.to_csv(OUTPUT_MCNEMAR_CSV_PATH, index=False)
    OUTPUT_MCNEMAR_MD_PATH.write_text(
        render_mcnemar_markdown_table(mcnemar_summary), encoding="utf-8"
    )
    OUTPUT_MCNEMAR_TEX_PATH.write_text(
        render_mcnemar_latex_table(mcnemar_summary), encoding="utf-8"
    )
    print(f"Saved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to: {OUTPUT_MD_PATH}")
    print(f"Saved McNemar CSV to: {OUTPUT_MCNEMAR_CSV_PATH}")
    print(f"Saved McNemar Markdown to: {OUTPUT_MCNEMAR_MD_PATH}")
    print(f"Saved McNemar LaTeX to: {OUTPUT_MCNEMAR_TEX_PATH}")


if __name__ == "__main__":
    main()
