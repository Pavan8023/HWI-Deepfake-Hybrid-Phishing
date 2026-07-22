from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    OUTPUT_MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TEST_SIZE,
)
from src.data_loader import load_dataset  # noqa: E402
from src.preprocessing import (  # noqa: E402
    awareness_schema_to_dict,
    preprocess_awareness_dataset,
)


INPUT_PATH = (
    RAW_DATA_DIR
    / "awareness"
    / "phishing_awareness_dataset.csv"
)

X_TRAIN_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_x_train.csv"
)

X_TEST_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_x_test.csv"
)

Y_TRAIN_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_y_train.csv"
)

Y_TEST_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_y_test.csv"
)

TRAIN_COMPLETE_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_train_complete.csv"
)

TEST_COMPLETE_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_test_complete.csv"
)

SUMMARY_PATH = (
    PROCESSED_DATA_DIR
    / "awareness_preprocessing_summary.json"
)

PREPROCESSOR_PATH = (
    OUTPUT_MODELS_DIR
    / "awareness_preprocessor.joblib"
)


def require_dataframe(
    loaded_data: object,
) -> pd.DataFrame:
    """Confirm that the project loader returned one DataFrame."""

    if not isinstance(
        loaded_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "The awareness dataset did not load as a DataFrame."
        )

    return loaded_data


def build_complete_partition(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Combine processed predictors and target for inspection."""

    if len(features) != len(target):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    result = features.reset_index(
        drop=True
    ).copy()

    result["clicked_link"] = (
        target.reset_index(drop=True)
        .astype("int64")
    )

    return result


def ensure_no_target_leakage(
    dataframe: pd.DataFrame,
) -> None:
    """Confirm that the target is absent from predictor columns."""

    forbidden_exact_columns = {
        "clicked_link",
        "target",
        "label",
    }

    leaked_columns = sorted(
        forbidden_exact_columns.intersection(
            dataframe.columns
        )
    )

    if leaked_columns:
        raise ValueError(
            "Target leakage detected in processed predictors: "
            + ", ".join(leaked_columns)
        )


def serialise_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Convert preprocessing metadata into JSON-safe values."""

    return {
        key: (
            value.tolist()
            if hasattr(value, "tolist")
            else value
        )
        for key, value in metadata.items()
    }


def main() -> int:
    """Build and save the real awareness modelling dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Awareness dataset not found: {INPUT_PATH}"
        )

    print("=" * 80)
    print("AWARENESS MODEL DATASET BUILD")
    print("=" * 80)
    print(f"Input dataset: {INPUT_PATH}")
    print(
        "reported_email included:",
        False,
    )
    print(
        "Reason:",
        (
            "Excluded because its timing relative to clicking "
            "has not been confirmed."
        ),
    )
    print()

    loaded_data = load_dataset(
        INPUT_PATH
    )

    raw_dataframe = require_dataframe(
        loaded_data
    )

    print(
        f"Raw shape: "
        f"{raw_dataframe.shape[0]:,} rows × "
        f"{raw_dataframe.shape[1]} columns"
    )

    processed_split, preprocessor = (
        preprocess_awareness_dataset(
            raw_dataframe,
            include_reported_email=False,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    ensure_no_target_leakage(
        processed_split.x_train
    )
    ensure_no_target_leakage(
        processed_split.x_test
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

    processed_split.x_train.to_csv(
        X_TRAIN_PATH,
        index=False,
        encoding="utf-8",
    )

    processed_split.x_test.to_csv(
        X_TEST_PATH,
        index=False,
        encoding="utf-8",
    )

    processed_split.y_train.to_frame(
        name="clicked_link"
    ).to_csv(
        Y_TRAIN_PATH,
        index=False,
        encoding="utf-8",
    )

    processed_split.y_test.to_frame(
        name="clicked_link"
    ).to_csv(
        Y_TEST_PATH,
        index=False,
        encoding="utf-8",
    )

    train_complete = (
        build_complete_partition(
            processed_split.x_train,
            processed_split.y_train,
        )
    )

    test_complete = (
        build_complete_partition(
            processed_split.x_test,
            processed_split.y_test,
        )
    )

    train_complete.to_csv(
        TRAIN_COMPLETE_PATH,
        index=False,
        encoding="utf-8",
    )

    test_complete.to_csv(
        TEST_COMPLETE_PATH,
        index=False,
        encoding="utf-8",
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    summary = {
        "research_role": (
            "Behavioural susceptibility modelling dataset"
        ),
        "raw_dataset": str(INPUT_PATH),
        "raw_rows": int(
            len(raw_dataframe)
        ),
        "raw_columns": int(
            raw_dataframe.shape[1]
        ),
        "train_rows": int(
            len(processed_split.x_train)
        ),
        "test_rows": int(
            len(processed_split.x_test)
        ),
        "processed_feature_count": int(
            processed_split.x_train.shape[1]
        ),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target_column": "clicked_link",
        "positive_class_meaning": (
            "Clicked the link, subject to confirmation "
            "from dataset documentation"
        ),
        "negative_class_meaning": (
            "Did not click the link, subject to confirmation "
            "from dataset documentation"
        ),
        "reported_email_included": False,
        "reported_email_exclusion_reason": (
            "Potential post-outcome leakage because the "
            "event timing has not been confirmed."
        ),
        "excluded_raw_columns": [
            "user_id",
            "geo_location",
            "email_received_time",
            "reported_email",
            "clicked_link",
        ],
        "preprocessing_fit_scope": (
            "Training partition only"
        ),
        "schema": awareness_schema_to_dict(),
        "metadata": serialise_metadata(
            processed_split.metadata
        ),
        "output_files": {
            "x_train": str(X_TRAIN_PATH),
            "x_test": str(X_TEST_PATH),
            "y_train": str(Y_TRAIN_PATH),
            "y_test": str(Y_TEST_PATH),
            "train_complete": str(
                TRAIN_COMPLETE_PATH
            ),
            "test_complete": str(
                TEST_COMPLETE_PATH
            ),
            "preprocessor": str(
                PREPROCESSOR_PATH
            ),
        },
        "scientific_warning": (
            "The dataset estimates proxy-based behavioural "
            "susceptibility. It does not measure a clinical or "
            "psychological weakness construct."
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    train_distribution = (
        processed_split.y_train
        .value_counts()
        .sort_index()
        .to_dict()
    )

    test_distribution = (
        processed_split.y_test
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print()
    print("=" * 80)
    print("AWARENESS DATASET BUILD COMPLETE")
    print("=" * 80)
    print(
        f"Training rows: "
        f"{len(processed_split.x_train):,}"
    )
    print(
        f"Testing rows: "
        f"{len(processed_split.x_test):,}"
    )
    print(
        f"Processed features: "
        f"{processed_split.x_train.shape[1]}"
    )
    print(
        f"Training target distribution: "
        f"{train_distribution}"
    )
    print(
        f"Testing target distribution: "
        f"{test_distribution}"
    )
    print(
        "Missing values in X train:",
        int(
            processed_split.x_train
            .isna()
            .sum()
            .sum()
        ),
    )
    print(
        "Missing values in X test:",
        int(
            processed_split.x_test
            .isna()
            .sum()
            .sum()
        ),
    )
    print()
    print(f"X train: {X_TRAIN_PATH}")
    print(f"X test: {X_TEST_PATH}")
    print(f"Y train: {Y_TRAIN_PATH}")
    print(f"Y test: {Y_TEST_PATH}")
    print(
        f"Saved preprocessor: "
        f"{PREPROCESSOR_PATH}"
    )
    print(f"Summary: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())