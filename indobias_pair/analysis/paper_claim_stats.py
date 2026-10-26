from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import pstdev

import pandas as pd

from source_vs_regional_base_common import BASE_PAIRS, LANGUAGE_LABELS, RAW_RESULT_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINING_RAW_DIR = REPO_ROOT / "indobias_pair/result/pretraining/raw"
QA_SUMMARY_PATH = REPO_ROOT / "indobias_qa/results/analysis/sp_summary.json"
OUTPUT_PATH = REPO_ROOT / "indobias_pair/analysis/paper_claim_stats.md"

DOMAIN_COLUMNS = [
    "Ideologi dan Agama",
    "Identitas dan Demografi",
    "Status Ekonomi",
    "Kultural dan Geografis",
    "Peran Sosial dan Keluarga",
]
LANGUAGE_DIRS = {
    "ID": "IndoBias_ID",
    "JV": "IndoBias_Javanese",
    "SU": "IndoBias_Sundanese",
    "MKS": "IndoBias_Makassar",
}
PRETRAINING_RUNS = {
    "CC100-ID": "indobert-base-uncased-pt-cc100-cc100_ind_source",
    "Wiki-ID": "indobert-base-uncased-pt-wikipedia-20231101.id",
    "Liputan6": "indobert-base-uncased-pt-liputan6-default",
    "Wiki-mix": "indobert-base-uncased-pt-wikipedia_wikipedia_wikipedia-20231101.id_20231101.jv_20231101.su",
    "CC100-mix": "indobert-base-uncased-pt-cc100_cc100_cc100-cc100_ind_source_cc100_jav_source_cc100_sun_source",
    "CC100+Wiki-mix": "indobert-base-uncased-pt-6_1_1_1_1_mix",
}
PRETRAINING_REPORT_FILES = {
    "ID": "bias_report_id.csv",
    "JV": "bias_report_javanese.csv",
    "SU": "bias_report_sundanese.csv",
    "MKS": "bias_report_makassar.csv",
}
QA_MODEL_PAIRS = [
    ("Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen"),
    ("gpt-4.1-mini", "gpt-5-mini", "GPT"),
]
QA_DEMOGRAPHY_ORDER = [
    "ethnicity",
    "lembaga",
    "name",
    "political_party",
    "religion",
    "university",
]
QA_DEMOGRAPHY_LABELS = {
    "ethnicity": "Ethnicity",
    "lembaga": "Institutions",
    "name": "Names",
    "political_party": "Political Parties",
    "religion": "Religion",
    "university": "Universities",
}


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    successes = int(successes)
    failures = int(failures)
    n = successes + failures
    if n == 0:
        return 1.0
    lower_tail = sum(comb(n, k) for k in range(min(successes, failures) + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
    pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    tie_counts: list[int] = []
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and abs(pairs[end][0] - pairs[start][0]) <= 1e-12:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for _value, index in pairs[start:end]:
            ranks[index] = average_rank
        if end - start > 1:
            tie_counts.append(end - start)
        start = end
    return ranks, tie_counts


def exact_wilcoxon_signed_rank(differences: list[float]) -> dict[str, float | int]:
    nonzero_differences = [diff for diff in differences if abs(diff) > 1e-12]
    if not nonzero_differences:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0}

    ranks, _ties = average_ranks([abs(diff) for diff in nonzero_differences])
    integer_ranks = [int(round(rank * 2.0)) for rank in ranks]
    total_rank = sum(integer_ranks)
    w_plus = sum(
        rank
        for rank, diff in zip(integer_ranks, nonzero_differences, strict=True)
        if diff > 0
    )
    w_minus = total_rank - w_plus
    observed = min(w_plus, w_minus)

    distribution = {0: 1}
    for rank in integer_ranks:
        updated: dict[int, int] = {}
        for rank_sum, count in distribution.items():
            updated[rank_sum] = updated.get(rank_sum, 0) + count
            updated[rank_sum + rank] = updated.get(rank_sum + rank, 0) + count
        distribution = updated

    lower_tail = sum(
        count for rank_sum, count in distribution.items() if rank_sum <= observed
    )
    p_value = min(1.0, 2.0 * lower_tail / (2 ** len(integer_ranks)))
    return {
        "n": len(nonzero_differences),
        "w_plus": w_plus / 2.0,
        "w_minus": w_minus / 2.0,
        "p": p_value,
    }


def exact_wilcoxon_rank_sum(
    group_a: list[float], group_b: list[float]
) -> dict[str, float | int]:
    values = group_a + group_b
    ranks, _ties = average_ranks(values)
    n_a = len(group_a)
    n_b = len(group_b)
    rank_sum_a = sum(ranks[:n_a])
    rank_sum_b = sum(ranks[n_a:])

    if n_b <= n_a:
        observed = rank_sum_b
        selected_size = n_b
    else:
        observed = rank_sum_a
        selected_size = n_a
    expected = selected_size * (len(values) + 1) / 2.0
    observed_distance = abs(observed - expected)

    total = 0
    as_or_more_extreme = 0
    for combo in combinations(range(len(values)), selected_size):
        rank_sum = sum(ranks[index] for index in combo)
        total += 1
        if abs(rank_sum - expected) >= observed_distance - 1e-12:
            as_or_more_extreme += 1

    return {
        "n_a": n_a,
        "n_b": n_b,
        "rank_sum_a": rank_sum_a,
        "rank_sum_b": rank_sum_b,
        "u_a": rank_sum_a - n_a * (n_a + 1) / 2.0,
        "u_b": rank_sum_b - n_b * (n_b + 1) / 2.0,
        "p": as_or_more_extreme / total,
    }


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.2f}"


def summarize_mcnemar_diagnostics(
    rows: list[dict[str, object]], comparison_key: str
) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        comparison = str(row[comparison_key])
        record = summary.setdefault(
            comparison,
            {
                "comparison": comparison,
                "language_deltas": {},
                "delta_sum": 0.0,
                "language_count": 0,
                "significant_languages": [],
            },
        )
        record["language_deltas"][str(row["language"])] = float(row["delta"])
        record["delta_sum"] = float(record["delta_sum"]) + float(row["delta"])
        record["language_count"] = int(record["language_count"]) + 1
        if float(row["p"]) < 0.05:
            record["significant_languages"].append(str(row["language"]))

    summary_rows = []
    for record in summary.values():
        record["average_delta"] = float(record["delta_sum"]) / int(
            record["language_count"]
        )
        record["diagnostics"] = (
            ", ".join(record["significant_languages"])
            if record["significant_languages"]
            else "none"
        )
        summary_rows.append(record)
    return summary_rows


def protrope_win(df: pd.DataFrame) -> pd.Series:
    if "protrope_win" in df.columns:
        if df["protrope_win"].dtype == object:
            return df["protrope_win"].astype(str).str.lower().isin({"true", "1", "yes"})
        return df["protrope_win"].astype(bool)
    return df["protrope_perplexity"] < df["antitrope_perplexity"]


def collect_model_family_bias_strength() -> tuple[
    list[dict[str, object]], dict[str, float]
]:
    files = sorted(
        {
            path.name
            for language_dir in LANGUAGE_LABELS
            for path in (RAW_RESULT_DIR / language_dir).glob("*.xlsx")
        }
    )
    rows: list[dict[str, object]] = []
    for filename in files:
        if filename.startswith("decoder_"):
            model_type = "decoder"
        elif filename.startswith("encoder_"):
            model_type = "encoder"
        else:
            continue

        overall_abs_devs = []
        domain_abs_devs = []
        for language_dir in LANGUAGE_LABELS:
            path = RAW_RESULT_DIR / language_dir / filename
            if not path.exists():
                continue
            df = pd.read_excel(path)
            wins = protrope_win(df)
            overall_abs_devs.append(abs(wins.mean() * 100.0 - 50.0))
            for domain in DOMAIN_COLUMNS:
                domain_wins = wins[df["domain"].eq(domain)]
                domain_abs_devs.append(abs(domain_wins.mean() * 100.0 - 50.0))

        if overall_abs_devs:
            rows.append(
                {
                    "filename": filename,
                    "model_type": model_type,
                    "overall_mean_abs_dev": sum(overall_abs_devs)
                    / len(overall_abs_devs),
                    "domain_mean_abs_dev": sum(domain_abs_devs) / len(domain_abs_devs),
                }
            )

    decoder = [
        float(row["overall_mean_abs_dev"])
        for row in rows
        if row["model_type"] == "decoder"
    ]
    encoder = [
        float(row["overall_mean_abs_dev"])
        for row in rows
        if row["model_type"] == "encoder"
    ]
    rank_sum_result = exact_wilcoxon_rank_sum(decoder, encoder)
    summary = {
        "decoder_mean": sum(decoder) / len(decoder),
        "encoder_mean": sum(encoder) / len(encoder),
        "diff": sum(decoder) / len(decoder) - sum(encoder) / len(encoder),
        "p": rank_sum_result["p"],
        "u": rank_sum_result["u_a"],
        "decoder_n": len(decoder),
        "encoder_n": len(encoder),
    }
    return rows, summary


def collect_decoder_language_effects() -> dict[str, list[dict[str, object]]]:
    files = sorted(
        path.name
        for path in (RAW_RESULT_DIR / "IndoBias_ID").glob("decoder_*.xlsx")
        if all(
            (RAW_RESULT_DIR / language_dir / path.name).exists()
            for language_dir in LANGUAGE_LABELS
        )
    )
    overall_scores = {}
    ideology_scores_by_model = {}
    for filename in files:
        for language_dir, language_code in LANGUAGE_LABELS.items():
            df = pd.read_excel(RAW_RESULT_DIR / language_dir / filename)
            wins = protrope_win(df)
            overall_scores[(filename, language_code)] = wins.mean() * 100.0
            ideology_wins = wins[df["domain"].eq("Ideologi dan Agama")]
            ideology_scores_by_model[(filename, language_code)] = (
                ideology_wins.mean() * 100.0
            )

    regional_average_diffs = [
        overall_scores[(filename, "ID")]
        - sum(overall_scores[(filename, language)] for language in ["JV", "SU", "MKS"])
        / 3.0
        for filename in files
    ]
    regional_average_wilcoxon = exact_wilcoxon_signed_rank(regional_average_diffs)
    overall_result = {
        "comparison": "ID > regional average",
        "mean_delta": sum(regional_average_diffs) / len(regional_average_diffs),
        "positive": sum(diff > 1e-12 for diff in regional_average_diffs),
        "negative": sum(diff < -1e-12 for diff in regional_average_diffs),
        "ties": sum(abs(diff) <= 1e-12 for diff in regional_average_diffs),
        "w_plus": regional_average_wilcoxon["w_plus"],
        "w_minus": regional_average_wilcoxon["w_minus"],
        "p": regional_average_wilcoxon["p"],
    }
    pairwise_results = []
    for local_language in ["JV", "SU", "MKS"]:
        diffs = [
            overall_scores[(filename, "ID")] - overall_scores[(filename, local_language)]
            for filename in files
        ]
        positive = sum(diff > 1e-12 for diff in diffs)
        negative = sum(diff < -1e-12 for diff in diffs)
        ties = sum(abs(diff) <= 1e-12 for diff in diffs)
        pairwise_results.append(
            {
                "comparison": f"ID > {local_language}",
                "mean_delta": sum(diffs) / len(diffs),
                "positive": positive,
                "negative": negative,
                "ties": ties,
            }
        )
    ideology_scores = [
        {
            "language": language_code,
            "mean": sum(
                ideology_scores_by_model[(filename, language_code)]
                for filename in files
            )
            / len(files),
        }
        for language_code in ["ID", "JV", "SU", "MKS"]
    ]
    return {
        "overall": overall_result,
        "pairwise": pairwise_results,
        "ideology": ideology_scores,
    }


def collect_source_regional_mcnemar() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in BASE_PAIRS:
        pair_name = f"{pair.source_label} -> {pair.regional_label}"
        for language_dir, language_code in LANGUAGE_LABELS.items():
            source_df = pd.read_excel(RAW_RESULT_DIR / language_dir / pair.source_file)
            regional_df = pd.read_excel(
                RAW_RESULT_DIR / language_dir / pair.regional_file
            )
            source_wins = protrope_win(source_df)
            regional_wins = protrope_win(regional_df)
            non_pro_to_pro = int((~source_wins & regional_wins).sum())
            pro_to_non_pro = int((source_wins & ~regional_wins).sum())
            rows.append(
                {
                    "pair": pair_name,
                    "language": language_code,
                    "n": len(source_wins),
                    "delta": (regional_wins.mean() - source_wins.mean()) * 100.0,
                    "non_pro_to_pro": non_pro_to_pro,
                    "pro_to_non_pro": pro_to_non_pro,
                    "p": exact_two_sided_binomial_p(non_pro_to_pro, pro_to_non_pro),
                }
            )
    return rows


def load_pretraining(run: str, language: str) -> tuple[pd.DataFrame, pd.Series]:
    path = (
        PRETRAINING_RAW_DIR
        / PRETRAINING_RUNS[run]
        / "checkpoint-500000"
        / PRETRAINING_REPORT_FILES[language]
    )
    df = pd.read_csv(path)
    return df, protrope_win(df)


def collect_pretraining_claims() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    final_rates = []
    for run in PRETRAINING_RUNS:
        record: dict[str, object] = {"run": run}
        for language in PRETRAINING_REPORT_FILES:
            _df, wins = load_pretraining(run, language)
            record[language] = wins.mean() * 100.0
        final_rates.append(record)

    comparisons = [
        ("CC100-ID", "Wiki-ID", "ID"),
        ("CC100-ID", "Liputan6", "ID"),
        ("Wiki-ID", "Wiki-mix", "ID"),
        ("Wiki-ID", "Wiki-mix", "JV"),
        ("Wiki-ID", "Wiki-mix", "SU"),
        ("Wiki-ID", "Wiki-mix", "MKS"),
        ("CC100-ID", "CC100-mix", "ID"),
        ("CC100-ID", "CC100-mix", "JV"),
        ("CC100-ID", "CC100-mix", "SU"),
        ("CC100-ID", "CC100-mix", "MKS"),
        ("CC100-ID", "CC100+Wiki-mix", "ID"),
        ("CC100-ID", "CC100+Wiki-mix", "JV"),
        ("CC100-ID", "CC100+Wiki-mix", "SU"),
        ("CC100-ID", "CC100+Wiki-mix", "MKS"),
    ]
    test_rows = []
    for before, after, language in comparisons:
        before_df, before_wins = load_pretraining(before, language)
        after_df, after_wins = load_pretraining(after, language)
        if not before_df[["protrope", "antitrope", "dimension", "domain"]].equals(
            after_df[["protrope", "antitrope", "dimension", "domain"]]
        ):
            raise ValueError(
                f"Unaligned pretraining comparison: {before} -> {after} ({language})"
            )
        non_pro_to_pro = int((~before_wins & after_wins).sum())
        pro_to_non_pro = int((before_wins & ~after_wins).sum())
        test_rows.append(
            {
                "comparison": f"{before} -> {after}",
                "language": language,
                "n": len(before_wins),
                "delta": (after_wins.mean() - before_wins.mean()) * 100.0,
                "non_pro_to_pro": non_pro_to_pro,
                "pro_to_non_pro": pro_to_non_pro,
                "p": exact_two_sided_binomial_p(non_pro_to_pro, pro_to_non_pro),
            }
        )
    return final_rates, test_rows


def collect_qa_model_comparison() -> list[dict[str, object]]:
    summary = json.loads(QA_SUMMARY_PATH.read_text(encoding="utf-8"))
    category_entity_scores: dict[tuple[str, str, str], list[float]] = {}
    for record in summary["entity_by_category"]:
        if record["demography"] == "gender":
            continue
        positive_ratio = record.get("positive_ratio")
        if positive_ratio is None:
            continue
        key = (record["model_name"], record["demography"], record["parent_category"])
        category_entity_scores.setdefault(key, []).append(float(positive_ratio))

    category_sp_dev = {
        key: pstdev(values) for key, values in category_entity_scores.items() if values
    }
    demography_model_scores: dict[tuple[str, str], list[float]] = {}
    for (model_name, demography, _parent_category), score in category_sp_dev.items():
        demography_model_scores.setdefault((model_name, demography), []).append(score)

    demography_scores = {}
    for key, values in demography_model_scores.items():
        demography_scores[key] = sum(values) / len(values)

    rows = []
    for before_model, after_model, family in QA_MODEL_PAIRS:
        deltas = []
        for demography in QA_DEMOGRAPHY_ORDER:
            before = demography_scores[(before_model, demography)]
            after = demography_scores[(after_model, demography)]
            deltas.append(after - before)
            rows.append(
                {
                    "family": family,
                    "demography": QA_DEMOGRAPHY_LABELS[demography],
                    "before": before,
                    "after": after,
                    "delta": after - before,
                }
            )
    return rows


def render_markdown() -> str:
    _family_rows, family_summary = collect_model_family_bias_strength()
    language_rows = collect_decoder_language_effects()
    source_regional_rows = collect_source_regional_mcnemar()
    source_regional_summary = summarize_mcnemar_diagnostics(
        source_regional_rows, "pair"
    )
    pretraining_rates, pretraining_tests = collect_pretraining_claims()
    pretraining_summary = [
        row
        for row in summarize_mcnemar_diagnostics(pretraining_tests, "comparison")
        if int(row["language_count"]) > 1
    ]
    qa_rows = collect_qa_model_comparison()

    lines = [
        "# Paper Claim Statistics",
        "",
        "This file is generated by `indobias_pair/analysis/paper_claim_stats.py`.",
        "",
        "## Decoder vs Encoder Bias Strength",
        "",
        (
            f"Mean model-level `|protrope win rate - 50|`: decoders "
            f"{family_summary['decoder_mean']:.2f} over {family_summary['decoder_n']} models, "
            f"encoders {family_summary['encoder_mean']:.2f} over {family_summary['encoder_n']} models. "
            f"Exact Wilcoxon rank-sum p={format_p(family_summary['p'])}."
        ),
        "",
        "## Language Effects for Decoder Models",
        "",
        "| Comparison | Mean delta | Positive | Negative | Ties | W+ | W- | Wilcoxon p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall_language = language_rows["overall"]
    lines.append(
        f"| {overall_language['comparison']} | {overall_language['mean_delta']:+.2f} | "
        f"{overall_language['positive']} | {overall_language['negative']} | "
        f"{overall_language['ties']} | {overall_language['w_plus']:.1f} | "
        f"{overall_language['w_minus']:.1f} | {format_p(float(overall_language['p']))} |"
    )
    lines.extend(
        [
            "",
            "Descriptive pairwise language deltas:",
            "",
            "| Comparison | Mean delta | Positive | Negative | Ties |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in language_rows["pairwise"]:
        lines.append(
            f"| {row['comparison']} | {row['mean_delta']:+.2f} | {row['positive']} | "
            f"{row['negative']} | {row['ties']} |"
        )

    lines.extend(
        [
            "",
            "Ideology and Religion mean protrope win rate among decoder models:",
            "",
            "| Language | Mean |",
            "| --- | ---: |",
        ]
    )
    for row in language_rows["ideology"]:
        lines.append(f"| {row['language']} | {row['mean']:.2f} |")

    lines.extend(
        [
            "",
            "## Source Base vs Regional Base Adaptation",
            "",
            "| Pair | ID delta | JV delta | SU delta | MKS delta | Avg delta | McNemar diagnostics |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in source_regional_summary:
        deltas = row["language_deltas"]
        lines.append(
            f"| {row['comparison']} | {format_delta(deltas.get('ID'))} | "
            f"{format_delta(deltas.get('JV'))} | {format_delta(deltas.get('SU'))} | "
            f"{format_delta(deltas.get('MKS'))} | {row['average_delta']:+.2f} | "
            f"{row['diagnostics']} |"
        )

    lines.extend(
        [
            "",
            "Language-cell diagnostics for the same adaptation pairs:",
            "",
            "| Pair | Lang | Delta | 0->1 | 1->0 | McNemar p |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in source_regional_rows:
        lines.append(
            f"| {row['pair']} | {row['language']} | {row['delta']:+.2f} | "
            f"{row['non_pro_to_pro']} | {row['pro_to_non_pro']} | "
            f"{format_p(float(row['p']))} |"
        )

    lines.extend(
        [
            "",
            "## Pretraining Final Checkpoint",
            "",
            "| Run | ID | JV | SU | MKS |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pretraining_rates:
        lines.append(
            f"| {row['run']} | {row['ID']:.2f} | {row['JV']:.2f} | "
            f"{row['SU']:.2f} | {row['MKS']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Multilingual final-checkpoint summary:",
            "",
            "| Comparison | ID delta | JV delta | SU delta | MKS delta | Avg delta | McNemar diagnostics |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in pretraining_summary:
        deltas = row["language_deltas"]
        lines.append(
            f"| {row['comparison']} | {format_delta(deltas.get('ID'))} | "
            f"{format_delta(deltas.get('JV'))} | {format_delta(deltas.get('SU'))} | "
            f"{format_delta(deltas.get('MKS'))} | {row['average_delta']:+.2f} | "
            f"{row['diagnostics']} |"
        )

    lines.extend(
        [
            "",
            "Final-checkpoint McNemar diagnostics:",
            "",
            "| Comparison | Lang | Delta | 0->1 | 1->0 | McNemar p |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pretraining_tests:
        lines.append(
            f"| {row['comparison']} | {row['language']} | {row['delta']:+.2f} | "
            f"{row['non_pro_to_pro']} | {row['pro_to_non_pro']} | "
            f"{format_p(float(row['p']))} |"
        )

    lines.extend(
        [
            "",
            "## QA Model Comparison",
            "",
            "Descriptive demography-level mean SP_dev disparity deltas:",
            "",
            "| Family | Demography | Before | After | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in qa_rows:
        lines.append(
            f"| {row['family']} | {row['demography']} | {row['before']:.2f} | "
            f"{row['after']:.2f} | {row['delta']:+.2f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(render_markdown(), encoding="utf-8")
    print(f"Saved paper claim stats to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
