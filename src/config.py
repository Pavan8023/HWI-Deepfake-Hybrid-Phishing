from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

APPROVED_RAW_DATASET_CATEGORIES: Final[tuple[str, ...]] = (
    "awareness",
    "ai_emails",
    "emails",
    "phishing_urls",
)

RAW_DATASET_CATEGORY_PATHS: Final[dict[str, Path]] = {
    category: RAW_DATA_DIR / category for category in APPROVED_RAW_DATASET_CATEGORIES
}

NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
REPORTS_DIR: Final[Path] = OUTPUTS_DIR / "reports"
DATASET_SUMMARIES_DIR: Final[Path] = REPORTS_DIR / "dataset_summaries"
RAW_FILE_MANIFEST_PATH: Final[Path] = REPORTS_DIR / "raw_file_manifest.csv"
UNSUPPORTED_RAW_FILES_PATH: Final[Path] = REPORTS_DIR / "unsupported_raw_files.csv"
STATISTICS_DIR: Final[Path] = OUTPUTS_DIR / "statistics"
PREDICTIONS_DIR: Final[Path] = OUTPUTS_DIR / "predictions"
SHAP_DIR: Final[Path] = OUTPUTS_DIR / "shap"
LIME_DIR: Final[Path] = OUTPUTS_DIR / "lime"
OUTPUT_MODELS_DIR: Final[Path] = OUTPUTS_DIR / "models"
OUTPUT_DASHBOARD_DIR: Final[Path] = OUTPUTS_DIR / "dashboard"

LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
REFERENCES_DIR: Final[Path] = PROJECT_ROOT / "references"
DASHBOARD_DIR: Final[Path] = PROJECT_ROOT / "dashboard"

EDA_FIGURES_DIR = FIGURES_DIR / "eda"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EDA_STATISTICS_DIR = STATISTICS_DIR / "eda"

EDA_SAMPLE_ROWS = 50_000
EDA_TOP_CATEGORIES = 20
EDA_TEXT_PREVIEW_LENGTH = 200

DATASET_METADATA_PATH: Final[Path] = REFERENCES_DIR / "dataset_metadata.json"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5

FEATURE_VALIDATION_REPORTS_DIR = (
    REPORTS_DIR / "feature_validation"
)

FEATURE_VALIDATION_STATISTICS_DIR = (
    STATISTICS_DIR / "feature_validation"
)

FEATURE_VALIDATION_FIGURES_DIR = (
    FIGURES_DIR / "feature_validation"
)

FEATURE_VALIDATION_ALPHA = 0.05
FEATURE_VALIDATION_HIGH_CORRELATION = 0.90

INITIAL_HWI_THRESHOLDS: Final[dict[str, int]] = {
    "low_max": 30,
    "medium_max": 70,
    "high_max": 100,
}

INITIAL_HWI_DIMENSION_WEIGHTS: Final[dict[str, float]] = {
    "awareness_risk": 0.25,
    "trust_risk": 0.20,
    "decision_risk": 0.20,
    "behavioural_risk": 0.20,
    "contextual_risk": 0.15,
}

SUPPORTED_DATASET_EXTENSIONS: Final[tuple[str, ...]] = (
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
)

CSV_ENCODINGS_TO_TRY: Final[tuple[str, ...]] = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin-1",
)

CANDIDATE_TARGET_COLUMN_KEYWORDS: Final[tuple[str, ...]] = (
    "label",
    "target",
    "class",
    "is_phishing",
    "phishing",
    "malicious",
    "attack",
    "outcome",
    "category",
    "type",
    "result",
)

GENERATED_DIRECTORIES: Final[tuple[Path, ...]] = (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    DATASET_SUMMARIES_DIR,
    STATISTICS_DIR,
    PREDICTIONS_DIR,
    SHAP_DIR,
    LIME_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_DASHBOARD_DIR,
    LOGS_DIR,
    EDA_FIGURES_DIR,
    EDA_REPORTS_DIR,
    EDA_STATISTICS_DIR,
    FEATURE_VALIDATION_REPORTS_DIR,
    FEATURE_VALIDATION_STATISTICS_DIR,
    FEATURE_VALIDATION_FIGURES_DIR,
)


def ensure_project_directories() -> None:
    """Create generated-output directories without touching raw datasets."""

    for directory in GENERATED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


ensure_project_directories()
