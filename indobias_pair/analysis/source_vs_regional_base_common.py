from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DOMAIN_COLUMNS = [
    "Ideology and Religion",
    "Identity and Demographics",
    "Economic Status",
    "Cultural and Geographic",
    "Social and Family Roles",
]

# Results produced before the data was released use the original Indonesian
# labels in `domain` and `dimension`; the released data uses `domain` and
# `subdomain` with the paper's English names.
LEGACY_DOMAIN_LABELS = {
    "Ideologi dan Agama": "Ideology and Religion",
    "Identitas dan Demografi": "Identity and Demographics",
    "Status Ekonomi": "Economic Status",
    "Kultural dan Geografis": "Cultural and Geographic",
    "Peran Sosial dan Keluarga": "Social and Family Roles",
}
LEGACY_SUBDOMAIN_LABELS = {
    "ethnicity": "Ethnicity", "gender": "Gender", "generasi": "Generation",
    "kelas_sosial": "Social Class", "pekerjaan": "Occupation", "pendidikan": "Education", "ekonomi": "Economy",
    "kawasan_sosio_teritorial": "Region", "domisili": "Domicile", "afiliasi_budaya": "Cultural Affiliation",
    "rural_urban_divide": "Rural-Urban", "akses_dan_infrastruktur": "Access & Infrastructure",
    "peran_keluarga": "Family Role", "status_pernikahan": "Marital Status", "status_sosial": "Social Status",
    "religion": "Religion", "ideologi": "Politics", "afiliasi_politik": "Political Affiliation",
}


def read_result(path: Path) -> pd.DataFrame:
    """Read a pair-track result file with English `domain` and `subdomain` labels."""
    return normalize_labels(pd.read_excel(path))


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "dimension" in df.columns and "subdomain" not in df.columns:
        df = df.rename(columns={"dimension": "subdomain"})
        df["subdomain"] = df["subdomain"].replace(LEGACY_SUBDOMAIN_LABELS)
    df["domain"] = df["domain"].replace(LEGACY_DOMAIN_LABELS)
    return df
RAW_RESULT_DIR = PROJECT_DIR / "result/pairs/raw"

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
class BasePair:
    family: str
    source_label: str
    source_file: str
    regional_label: str
    regional_file: str


BASE_PAIRS = [
    BasePair(
        family="Indonesian",
        source_label="Llama-2-7B",
        source_file="decoder_meta-llama_Llama-2-7b-hf.xlsx",
        regional_label="Komodo-7B",
        regional_file="decoder_Yellow-AI-NLP_komodo-7b-base.xlsx",
    ),
    BasePair(
        family="SEA",
        source_label="Qwen2-7B",
        source_file="decoder_Qwen_Qwen2-7B.xlsx",
        regional_label="SeaLLM-v3-7B",
        regional_file="decoder_SeaLLMs_SeaLLMs-v3-7B.xlsx",
    ),
    BasePair(
        family="SEA",
        source_label="Qwen2.5-7B",
        source_file="decoder_Qwen_Qwen2_5-7B.xlsx",
        regional_label="Sailor2-8B",
        regional_file="decoder_sail_Sailor2-8B.xlsx",
    ),
]
