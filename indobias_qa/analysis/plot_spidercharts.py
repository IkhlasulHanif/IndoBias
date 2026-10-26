from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
SUMMARY_JSON = BASE_DIR / "results" / "analysis" / "sp_summary.json"
OUTPUT_DIR = BASE_DIR / "results" / "analysis" / "spidercharts"

CURATED_PANELS = [
    {
        "filename": "qwen3_8b_lembaga_top5_spider.png",
        "title": "Qwen3-8B: Selected Institutions (Anonymized)",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "lembaga",
        "source_file": "lembaga",
        "categories": [
            "Control of Corruption",
            "Government Effectiveness",
            "Political Stability",
            "Regulatory Quality",
            "Rule of Law",
            "Voice and Accountability",
        ],
        "series": [
            {"label": "Gov Institution A", "entity": "Kemenhan", "anonymized": True},
            {"label": "Gov Institution B", "entity": "BIN", "anonymized": True},
            {"label": "Gov Institution C", "entity": "Kemenkumham", "anonymized": True},
            {"label": "Gov Institution D", "entity": "Kominfo", "anonymized": True},
            {"label": "Gov Institution E", "entity": "BKN", "anonymized": True},
        ],
        "note": "Each colored line is one government institution. Labels are anonymized. Higher values mean higher stereotype positivity (SP).",
    },
    {
        "filename": "qwen3_8b_political_party_top5_spider.png",
        "title": "Qwen3-8B: Selected Political Parties",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "political_party",
        "source_file": "political_party",
        "categories": [
            "Control of Corruption",
            "Government Effectiveness",
            "Political Stability",
            "Regulatory Quality",
            "Rule of Law",
            "Voice and Accountability",
        ],
        "series": [
            {"label": "Gerindra", "entity": "Gerindra"},
            {"label": "Golkar", "entity": "Golkar"},
            {"label": "Partai Ummat", "entity": "Partai Ummat"},
            {"label": "PSI", "entity": "PSI"},
            {"label": "Gelora", "entity": "Gelora"},
        ],
        "note": "Each colored line is one political party. Higher values mean higher stereotype positivity (SP).",
    },
    {
        "filename": "qwen3_8b_ethnicity_top5_spider.png",
        "title": "Qwen3-8B: Selected Ethnic Groups",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "ethnicity",
        "source_file": "ethnicity_indonesia",
        "categories": [
            "Basic Human Needs",
            "Foundations of Wellbeing",
            "Opportunity",
        ],
        "series": [
            {"label": "Korowai", "entity": "Korowai"},
            {"label": "Bajau", "entity": "Bajau (Sama-Bajau)"},
            {"label": "Javanese", "entity": "Jawa"},
            {"label": "Sundanese", "entity": "Sunda"},
            {"label": "Chinese", "entity": "Tionghoa"},
        ],
        "note": "Each colored line is one ethnic group. Higher values mean higher stereotype positivity (SP).",
    },
    {
        "filename": "qwen3_8b_name_top3_spider.png",
        "title": "Qwen3-8B: Selected Names (Anonymized)",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "name",
        "source_file": "name",
        "legend_ncol": 1,
        "legend_y": 0.08,
        "categories": [
            "Basic Human Needs",
            "Foundations of Wellbeing",
            "Opportunity",
        ],
        "series": [
            {"label": "Presidential Candidate X", "entity": "Ganjar Pranowo", "anonymized": True},
            {"label": "Presidential Candidate Y", "entity": "Anies Baswedan", "anonymized": True},
            {"label": "Presidential Candidate Z", "entity": "Prabowo Subianto", "anonymized": True},
        ],
        "note": "Each colored line is one presidential candidate name. Labels are anonymized and order-randomized. Higher values mean higher stereotype positivity (SP).",
    },
    {
        "filename": "qwen3_8b_religion_top5_spider.png",
        "title": "Qwen3-8B: Selected Religions",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "religion",
        "source_file": "religion",
        "categories": [
            "Basic Human Needs",
            "Foundations of Wellbeing",
            "Opportunity",
        ],
        "series": [
            {"label": "Islam", "entity": "Islam"},
            {"label": "Catholicism", "entity": "Katolik"},
            {"label": "Protestanism", "entity": "Kristen"},
            {"label": "Fanomba Adu", "entity": "Fanömba Adu"},
            {"label": "Ugamo Malim", "entity": "Ugamo Malim"},
            {"label": "Buddhism", "entity": "Budha"},
        ],
        "note": "Each colored line is one religion. Higher values mean higher stereotype positivity (SP).",
    },
    {
        "filename": "qwen3_8b_university_top5_spider.png",
        "title": "Qwen3-8B: Selected Universities",
        "model_name": "Qwen/Qwen3-8B",
        "demography": "university",
        "source_file": "university",
        "categories": [
            "Achievement Orientation",
            "Conscientiousness",
            "Practical Intelligence",
            "Social Influence",
        ],
        "series": [
            {
                "label": "ITB (Bandung Institute of Technology)",
                "entity": "Institut Teknologi Bandung",
            },
            {
                "label": "IKJ (Jakarta Institute of the Arts)",
                "entity": "Institut Kesenian Jakarta",
            },
            {
                "label": "IPB (Bogor Agricultural University)",
                "entity": "Institut Pertanian Bogor",
            },
            {
                "label": "UI (University of Indonesia)",
                "entity": "Universitas Indonesia",
            },
            {
                "label": "UPH (Pelita Harapan University, private)",
                "entity": "Universitas Pelita Harapan",
            },
        ],
        "note": "Each colored line is one university. This panel uses representative public, private, and specialist campuses rather than aggregates.",
    },
]

COLOR_CYCLE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
]


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def build_index(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], dict[str, float]]:
    index: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (
            str(row["model_name"]),
            str(row["demography"]),
            str(row["source_file"]),
            str(row["group"]),
        )
        index.setdefault(key, {})[str(row["parent_category"])] = float(row["positive_ratio"])
    return index


def resolve_series_values(
    entity_index: dict[tuple[str, str, str, str], dict[str, float]],
    panel: dict[str, object],
    series: dict[str, object],
) -> tuple[str, list[float], dict[str, float], dict[str, object]]:
    model_name = str(panel["model_name"])
    demography = str(panel["demography"])
    source_file = str(panel["source_file"])
    categories = list(panel["categories"])
    label = str(series["label"])

    if "entity" in series:
        entity = str(series["entity"])
        values_by_cat = entity_index[(model_name, demography, source_file, entity)]
        filtered = {category: values_by_cat[category] for category in categories}
        metadata = {"type": "entity"}
        if bool(series.get("anonymized")):
            metadata["anonymized"] = True
        else:
            metadata["entity"] = entity
        return label, [filtered[category] for category in categories], filtered, metadata

    raise KeyError(f"Series must define 'entity': {series}")


def plot_panel(entity_index: dict[tuple[str, str, str, str], dict[str, float]], panel: dict[str, object]) -> list[dict[str, object]]:
    categories = list(panel["categories"])
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(8.8, 7.8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=14)
    ax.grid(alpha=0.35)
    ax.spines["polar"].set_alpha(0.35)

    summary_rows = []
    for idx, series in enumerate(panel["series"]):
        label, values, filtered_scores, metadata = resolve_series_values(entity_index, panel, series)
        closed_values = values + [values[0]]
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        ax.plot(angles, closed_values, linewidth=2.3, color=color, label=label)
        ax.fill(angles, closed_values, color=color, alpha=0.08)
        summary_rows.append(
            {
                "chart_file": str(panel["filename"]),
                "model_name": str(panel["model_name"]),
                "demography": str(panel["demography"]),
                "source_file": str(panel["source_file"]),
                "label": label,
                "series_type": metadata["type"],
                "sp_min": round(min(values), 2),
                "sp_max": round(max(values), 2),
                "sp_span": round(max(values) - min(values), 2),
                "category_scores": filtered_scores,
                "series_metadata": metadata,
            }
        )

    legend_ncol = int(panel.get("legend_ncol", 3))
    legend_y = float(panel.get("legend_y", -0.03))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / str(panel["filename"]), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def write_curated_summary(summary_rows: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "curated_entities.json"
    path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_legend_note() -> None:
    content = """Spider chart notes

SP means stereotype positivity: the percentage of valid responses that selected the positive stereotype label.

How to read the charts:
- Each colored line represents one entity.
- Larger values mean the model selected the positive stereotype more often for that category.
- A more uneven shape means the model treated the same entity differently across categories.
"""
    (OUTPUT_DIR / "README_spidercharts.txt").write_text(content, encoding="utf-8")


def main() -> None:
    data = load_summary()
    rows = data["entity_by_category"]
    entity_index = build_index(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary_rows = []
    for panel in CURATED_PANELS:
        all_summary_rows.extend(plot_panel(entity_index, panel))
    write_curated_summary(all_summary_rows)
    write_legend_note()

    print(f"Wrote spider charts to {OUTPUT_DIR}")
    for panel in CURATED_PANELS:
        print(f"- {panel['filename']}")


if __name__ == "__main__":
    main()
