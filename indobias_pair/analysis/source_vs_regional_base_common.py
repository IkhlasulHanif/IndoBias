from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
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
