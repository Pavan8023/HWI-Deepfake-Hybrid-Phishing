from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
STATISTICS_DIR = OUTPUTS_DIR / "statistics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SHAP_DIR = OUTPUTS_DIR / "shap"
LIME_DIR = OUTPUTS_DIR / "lime"

LOGS_DIR = PROJECT_ROOT / "logs"
REFERENCES_DIR = PROJECT_ROOT / "references"

# ---------------------------------------------------------
# Reproducibility settings
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------
# HWI settings
# These are provisional and must be justified using literature.
# ---------------------------------------------------------

HWI_MIN = 0
HWI_MAX = 100

LOW_RISK_MAX = 30
MEDIUM_RISK_MAX = 70

HWI_DIMENSION_WEIGHTS = {
    "awareness_risk": 0.25,
    "trust_risk": 0.20,
    "decision_risk": 0.20,
    "behavioural_risk": 0.20,
    "contextual_risk": 0.15,
}

# ---------------------------------------------------------
# Create output directories automatically
# ---------------------------------------------------------

DIRECTORIES = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    STATISTICS_DIR,
    PREDICTIONS_DIR,
    SHAP_DIR,
    LIME_DIR,
    LOGS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)