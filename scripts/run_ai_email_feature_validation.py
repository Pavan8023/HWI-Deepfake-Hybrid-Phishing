from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (  # noqa: E402
    FEATURE_VALIDATION_FIGURES_DIR,
    FEATURE_VALIDATION_REPORTS_DIR,
    FEATURE_VALIDATION_STATISTICS_DIR,
    PROCESSED_DATA_DIR,
)
from src.feature_validation import (  # noqa: E402
    run_ai_email_feature_validation,
)


INPUT_PATH = (
    PROCESSED_DATA_DIR
    / "ai_email_features.csv"
)


def main() -> int:
    """Run validation on the real engineered AI-email dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset does not exist: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print("=" * 80)
    print("AI EMAIL FEATURE VALIDATION")
    print("=" * 80)
    print(f"Input: {INPUT_PATH}")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]}")
    print()

    summary = run_ai_email_feature_validation(
        dataframe,
        reports_directory=(
            FEATURE_VALIDATION_REPORTS_DIR
        ),
        statistics_directory=(
            FEATURE_VALIDATION_STATISTICS_DIR
        ),
        figures_directory=(
            FEATURE_VALIDATION_FIGURES_DIR
        ),
    )

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(
        "Numeric features:",
        summary["numeric_feature_count"],
    )
    print(
        "Usable features:",
        summary["usable_feature_count"],
    )
    print(
        "Constant features:",
        summary["constant_feature_count"],
    )
    print(
        "Significant four-group features:",
        summary[
            "significant_four_group_features"
        ],
    )
    print(
        "Significant phishing features:",
        summary[
            "significant_phishing_features"
        ],
    )
    print(
        "Significant source features:",
        summary[
            "significant_source_features"
        ],
    )
    print(
        "High-correlation pairs:",
        summary[
            "high_correlation_pair_count"
        ],
    )
    print()
    print(
        "Reports:",
        FEATURE_VALIDATION_REPORTS_DIR,
    )
    print(
        "Statistics:",
        FEATURE_VALIDATION_STATISTICS_DIR,
    )
    print(
        "Figures:",
        FEATURE_VALIDATION_FIGURES_DIR,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())