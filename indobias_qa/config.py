"""Shared project paths and dataset mapping tables."""

from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
GROUPS_DIR = DATA_DIR / "groups"
DIMENSIONS_FILE = DATA_DIR / "dimensions.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"

# Valid CLI values
VALID_DIMENSIONS = ("wgi", "onet", "qs_rankings", "sp")
VALID_DEMOGRAPHIES = (
    "ethnicity",
    "gender",
    "lembaga",
    "media",
    "name",
    "nationality",
    "organisasi_bisnis",
    "political_party",
    "race",
    "religion",
    "university",
)

# Runtime mapping: which stereotype dimension a demography should use
DEMOGRAPHY_TO_DIMENSION: dict[str, str] = {
    "ethnicity": "sp",
    "gender": "sp",
    "lembaga": "wgi",
    "name": "sp",
    "political_party": "wgi",
    "religion": "sp",
    "university": "onet",
}

# Loader mapping: infer metadata from CSV filenames
GROUP_FILE_TO_DEMOGRAPHY: dict[str, str] = {
    "ethnicity": "ethnicity",
    "ethnicity_indonesia": "ethnicity",
    "gender": "gender",
    "lembaga": "lembaga",
    "media": "media",
    "name": "name",
    "nationality": "nationality",
    "organisasi_bisnis": "organisasi_bisnis",
    "political_party": "political_party",
    "race": "race",
    "religion": "religion",
    "university": "university",
}

GROUP_FILE_TO_COUNTRY: dict[str, str] = {
    "ethnicity_indonesia": "Indonesia",
}


def infer_group_metadata(filename_stem: str) -> tuple[str, str | None]:
    """Infer demography and optional country from a groups filename stem."""
    normalized = filename_stem.lower()
    demography = GROUP_FILE_TO_DEMOGRAPHY.get(normalized)
    if demography is None:
        raise ValueError(f"Cannot infer demography from filename: {filename_stem}")

    country = GROUP_FILE_TO_COUNTRY.get(normalized)
    return demography, country
