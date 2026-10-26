from __future__ import annotations

import csv
from pathlib import Path

from indobias_qa.config import DEMOGRAPHY_TO_DIMENSION, GROUPS_DIR, infer_group_metadata


OUTPUT_PATH = Path("indobias_qa/analysis/group_inventory_table.tex")
DIMENSION_LABELS = {
    "sp": "SP",
    "wgi": "WGI",
    "onet": "O*NET",
}
DIMENSION_ORDER = {"sp": 0, "wgi": 1, "onet": 2, "unmapped": 3}
DEMOGRAPHY_LABELS = {
    "ethnicity": "Ethnicity",
    "gender": "Gender",
    "lembaga": "Institutions",
    "media": "Media",
    "name": "Names",
    "organisasi_bisnis": "Business Organizations",
    "political_party": "Political Parties",
    "religion": "Religion",
    "university": "Universities",
}


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def collect_rows() -> list[dict[str, str | int]]:
    rows = []
    for path in sorted(GROUPS_DIR.glob("*.csv")):
        demography, _country = infer_group_metadata(path.stem)
        dimension = DEMOGRAPHY_TO_DIMENSION.get(demography, "unmapped")
        rows.append(
            {
                "dimension": dimension,
                "demography": demography,
                "source_file": path.name,
                "count": count_rows(path),
            }
        )

    rows.sort(
        key=lambda row: (
            DIMENSION_ORDER[str(row["dimension"])],
            DEMOGRAPHY_LABELS.get(str(row["demography"]), str(row["demography"])),
        )
    )
    return rows


def render_table(rows: list[dict[str, str | int]]) -> str:
    total_entities = sum(int(row["count"]) for row in rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lllc}",
        r"\hline",
        r"\textbf{Dimension} & \textbf{Demography} & \textbf{Source File} & \textbf{Entities} \\",
        r"\hline",
    ]

    previous_dimension = None
    for row in rows:
        dimension = str(row["dimension"])
        dimension_cell = DIMENSION_LABELS.get(dimension, "Unmapped") if dimension != previous_dimension else ""
        demography = DEMOGRAPHY_LABELS.get(str(row["demography"]), str(row["demography"]))
        source_file = str(row["source_file"])
        count = int(row["count"])
        lines.append(
            " & ".join(
                [
                    latex_escape(dimension_cell),
                    latex_escape(demography),
                    r"\texttt{" + latex_escape(source_file) + "}",
                    str(count),
                ]
            )
            + r" \\"
        )
        previous_dimension = dimension

    lines.extend(
        [
            r"\hline",
            rf"\multicolumn{{3}}{{r}}{{\textbf{{Total}}}} & \textbf{{{total_entities}}} \\",
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Entity inventory in \texttt{indobias\_qa/data/groups}, grouped by demography and mapped stereotype dimension.}",
            r"\label{tab:qa_group_inventory}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect_rows()
    OUTPUT_PATH.write_text(render_table(rows), encoding="utf-8")
    print(f"Wrote group inventory table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
