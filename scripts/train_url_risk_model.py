from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
)
from src.data_loader import load_dataset  # noqa: E402
from src.model_training import (  # noqa: E402
    calculate_classification_metrics,
    extract_positive_probabilities,
)
from src.url_risk_model import (  # noqa: E402
    prepare_url_dataset,
    stratified_url_split,
    validate_url_split,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phishing_urls"
    / "malicious_phish.csv"
)

REPORTS_DIR = (
    OUTPUTS_DIR
    / "reports"
    / "url_risk_model"
)

PREDICTIONS_DIR = (
    OUTPUTS_DIR
    / "predictions"
)

COMPARISON_PATH = (
    REPORTS_DIR
    / "url_model_comparison.csv"
)

METRICS_PATH = (
    REPORTS_DIR
    / "url_model_metrics.json"
)

DATA_SUMMARY_PATH = (
    REPORTS_DIR
    / "url_dataset_summary.json"
)

SPLIT_SUMMARY_PATH = (
    REPORTS_DIR
    / "url_split_summary.json"
)

TEST_SCORES_PATH = (
    PREDICTIONS_DIR
    / "url_test_risk_scores.csv"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "url_best_calibrated_model.joblib"
)

MAXIMUM_ROWS = 200_000
CALIBRATION_FOLDS = 3


def require_dataframe(
    loaded: object,
) -> pd.DataFrame:
    """Ensure the loader returned one DataFrame."""

    if not isinstance(
        loaded,
        pd.DataFrame,
    ):
        raise TypeError(
            "The URL dataset did not load as a DataFrame."
        )

    return loaded


def build_estimators() -> dict[str, BaseEstimator]:
    """Build fast URL-risk baseline estimators."""

    logistic = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gradient_boosting = (
        HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.10,
            random_state=RANDOM_STATE,
        )
    )

    return {
        "logistic_regression": logistic,
        "hist_gradient_boosting": (
            gradient_boosting
        ),
    }


def calibrate_estimator(
    estimator: BaseEstimator,
) -> CalibratedClassifierCV:
    """Create a cross-validated sigmoid calibrator."""

    splitter = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=splitter,
        n_jobs=-1,
    )


def fit_and_evaluate(
    *,
    model_name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    calibrated: bool,
) -> dict[str, Any]:
    """Fit and evaluate one URL-risk classifier."""

    fitted = clone(
        estimator
    )

    fitted.fit(
        x_train,
        y_train,
    )

    probabilities = (
        extract_positive_probabilities(
            fitted,
            x_test,
        )
    )

    (
        metrics,
        predictions,
        matrix,
        report,
    ) = calculate_classification_metrics(
        y_test,
        probabilities,
        threshold=0.50,
    )

    return {
        "model_name": model_name,
        "estimator": fitted,
        "calibrated": calibrated,
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def main() -> int:
    """Train and save the URL technical-risk model."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"URL dataset not found: {INPUT_PATH}"
        )

    loaded = load_dataset(
        INPUT_PATH
    )

    raw_dataframe = require_dataframe(
        loaded
    )

    print("=" * 80)
    print("URL TECHNICAL RISK MODEL")
    print("=" * 80)
    print(f"Raw shape: {raw_dataframe.shape}")
    print(
        f"Maximum modelling rows: "
        f"{MAXIMUM_ROWS:,}"
    )
    print()

    (
        features,
        target,
        metadata,
        dataset_summary,
    ) = prepare_url_dataset(
        raw_dataframe,
        maximum_rows=MAXIMUM_ROWS,
        random_state=RANDOM_STATE,
    )

    split = stratified_url_split(
        features,
        target,
        metadata,
    )

    split_summary = validate_url_split(
        split
    )

    print(
        f"Model shape: "
        f"{len(features):,} rows × "
        f"{features.shape[1]} features"
    )
    print(
        "Training target:",
        split.y_train
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print(
        "Testing target:",
        split.y_test
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print()

    estimators = build_estimators()

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for base_name, base_estimator in (
        estimators.items()
    ):
        results[base_name] = (
            fit_and_evaluate(
                model_name=base_name,
                estimator=base_estimator,
                x_train=split.x_train,
                y_train=split.y_train,
                x_test=split.x_test,
                y_test=split.y_test,
                calibrated=False,
            )
        )

        calibrated_name = (
            f"{base_name}_calibrated"
        )

        results[calibrated_name] = (
            fit_and_evaluate(
                model_name=calibrated_name,
                estimator=calibrate_estimator(
                    base_estimator
                ),
                x_train=split.x_train,
                y_train=split.y_train,
                x_test=split.x_test,
                y_test=split.y_test,
                calibrated=True,
            )
        )

    comparison_records = [
        {
            "model_name": name,
            "calibrated": result[
                "calibrated"
            ],
            **result["metrics"],
        }
        for name, result in results.items()
    ]

    comparison = (
        pd.DataFrame(
            comparison_records
        )
        .sort_values(
            by=[
                "roc_auc",
                "brier_score",
                "f1",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    calibrated_results = (
        comparison.loc[
            comparison["calibrated"]
            .astype(bool)
        ]
        .sort_values(
            by=[
                "roc_auc",
                "brier_score",
                "f1",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
    )

    best_model_name = str(
        calibrated_results.iloc[0][
            "model_name"
        ]
    )

    best_result = results[
        best_model_name
    ]

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    with DATA_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    with SPLIT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    metrics_payload = {
        "selected_model": best_model_name,
        "score_name": (
            "url_technical_risk_score"
        ),
        "score_definition": (
            "Calibrated probability that the URL is "
            "malicious, multiplied by 100."
        ),
        "target_definition": {
            "0": "benign",
            "1": (
                "phishing, malware or defacement"
            ),
        },
        "maximum_modelling_rows": (
            MAXIMUM_ROWS
        ),
        "models": {
            name: {
                "calibrated": result[
                    "calibrated"
                ],
                "metrics": result[
                    "metrics"
                ],
                "confusion_matrix": result[
                    "confusion_matrix"
                ],
                "classification_report": result[
                    "classification_report"
                ],
            }
            for name, result in (
                results.items()
            )
        },
        "scientific_warning": (
            "This score measures technical URL "
            "maliciousness. It does not directly measure "
            "an individual user's susceptibility."
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    joblib.dump(
        best_result["estimator"],
        BEST_MODEL_PATH,
    )

    scores = (
        split.metadata_test.copy()
    )

    scores[
        "actual_malicious_label"
    ] = split.y_test

    scores[
        "predicted_malicious_label"
    ] = best_result[
        "predictions"
    ]

    scores[
        "url_risk_probability"
    ] = best_result[
        "probabilities"
    ]

    scores[
        "url_technical_risk_score"
    ] = (
        np.asarray(
            best_result[
                "probabilities"
            ],
            dtype=float,
        )
        * 100
    ).round(2)

    scores.to_csv(
        TEST_SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("URL MODEL COMPLETE")
    print("=" * 80)

    print(
        comparison[
            [
                "model_name",
                "calibrated",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(
        f"Selected model: "
        f"{best_model_name}"
    )
    print(
        "Selected ROC-AUC:",
        round(
            best_result[
                "metrics"
            ]["roc_auc"],
            4,
        ),
    )
    print(
        "Selected Brier score:",
        round(
            best_result[
                "metrics"
            ]["brier_score"],
            4,
        ),
    )
    print()
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Scores: {TEST_SCORES_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())