from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_ai_email_feature_dataset,
    validate_feature_frame,
)


RAW_AI_EMAIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai_emails"
)

OUTPUT_CSV_PATH = (
    PROCESSED_DATA_DIR
    / "ai_email_features.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROCESSED_DATA_DIR
    / "ai_email_features_summary.json"
)


def require_dataframe(
    loaded_data: object,
    dataset_name: str,
) -> pd.DataFrame:
    """Ensure the project loader returned a DataFrame."""

    if not isinstance(
        loaded_data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{dataset_name} did not load as a DataFrame."
        )

    return loaded_data


def main() -> int:
    """Build the harmonised AI-email feature table."""

    dataset_paths = {
        "human_legitimate": (
            RAW_AI_EMAIL_DIR
            / "human-generated"
            / "legit.csv"
        ),
        "human_phishing": (
            RAW_AI_EMAIL_DIR
            / "human-generated"
            / "phishing.csv"
        ),
        "llm_legitimate": (
            RAW_AI_EMAIL_DIR
            / "llm-generated"
            / "legit.csv"
        ),
        "llm_phishing": (
            RAW_AI_EMAIL_DIR
            / "llm-generated"
            / "phishing.csv"
        ),
    }

    for name, path in dataset_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset {name}: {path}"
            )

    human_legitimate = require_dataframe(
        load_dataset(
            dataset_paths["human_legitimate"]
        ),
        "human_legitimate",
    )

    human_phishing = require_dataframe(
        load_dataset(
            dataset_paths["human_phishing"]
        ),
        "human_phishing",
    )

    llm_legitimate = require_dataframe(
        load_dataset(
            dataset_paths["llm_legitimate"]
        ),
        "llm_legitimate",
    )

    llm_phishing = require_dataframe(
        load_dataset(
            dataset_paths["llm_phishing"]
        ),
        "llm_phishing",
    )

    feature_dataset = build_ai_email_feature_dataset(
        human_legitimate=human_legitimate,
        human_phishing=human_phishing,
        llm_legitimate=llm_legitimate,
        llm_phishing=llm_phishing,
    )

    validation_summary = validate_feature_frame(
        feature_dataset
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_dataset.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    with OUTPUT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print("=" * 80)
    print("AI EMAIL FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(feature_dataset):,}")
    print(f"Columns: {feature_dataset.shape[1]}")
    print(
        "Groups:",
        feature_dataset[
            "experiment_group"
        ].value_counts().to_dict(),
    )
    print(f"Missing values: {validation_summary['total_missing_values']}")
    print(f"Infinite values: {validation_summary['infinite_values']}")
    print(f"Feature dataset: {OUTPUT_CSV_PATH}")
    print(f"Summary: {OUTPUT_SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())