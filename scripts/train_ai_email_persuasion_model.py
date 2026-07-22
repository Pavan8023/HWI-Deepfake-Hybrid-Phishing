from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ai_email_model import (  # noqa: E402
    prepare_ai_email_model_frame,
    stratified_ai_email_split,
    validate_ai_email_split,
)
from src.config import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
)
from src.model_training import (  # noqa: E402
    DEFAULT_MODEL_CONFIGURATION,
    model_configuration_to_dict,
    select_best_hwi_model,
    train_awareness_model_suite,
)


INPUT_PATH = (
    PROCESSED_DATA_DIR
    / "ai_email_features.csv"
)

REPORTS_DIR = (
    OUTPUTS_DIR
    / "reports"
    / "ai_email_model"
)

PREDICTIONS_DIR = (
    OUTPUTS_DIR
    / "predictions"
)

COMPARISON_PATH = (
    REPORTS_DIR
    / "ai_email_model_comparison.csv"
)

METRICS_PATH = (
    REPORTS_DIR
    / "ai_email_model_metrics.json"
)

SPLIT_SUMMARY_PATH = (
    REPORTS_DIR
    / "ai_email_split_summary.json"
)

TEST_SCORES_PATH = (
    PREDICTIONS_DIR
    / "ai_email_test_persuasion_scores.csv"
)

FULL_SCORES_PATH = (
    PREDICTIONS_DIR
    / "ai_email_all_persuasion_scores.csv"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "ai_email_best_calibrated_model.joblib"
)


def main() -> int:
    """Train and save the AI-email phishing persuasion model."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"AI-email features not found: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    features, target, metadata = (
        prepare_ai_email_model_frame(
            dataframe
        )
    )

    split = stratified_ai_email_split(
        features,
        target,
        metadata,
    )

    split_summary = validate_ai_email_split(
        split
    )

    print("=" * 80)
    print("AI-EMAIL PERSUASION MODEL")
    print("=" * 80)
    print(f"Input shape: {dataframe.shape}")
    print(
        f"Selected numeric predictors: "
        f"{len(split.feature_names)}"
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

    (
        trained_results,
        comparison,
        cross_validation,
    ) = train_awareness_model_suite(
        x_train=split.x_train,
        y_train=split.y_train,
        x_test=split.x_test,
        y_test=split.y_test,
        configuration=(
            DEFAULT_MODEL_CONFIGURATION
        ),
    )

    best_model_name = (
        select_best_hwi_model(
            comparison
        )
    )

    best_result = trained_results[
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
            "email_persuasion_risk_score"
        ),
        "score_definition": (
            "Calibrated estimated probability that an email "
            "belongs to the phishing class, multiplied by 100."
        ),
        "configuration": (
            model_configuration_to_dict()
        ),
        "cross_validation": (
            cross_validation
        ),
        "models": {
            name: {
                "calibrated": result.calibrated,
                "metrics": result.metrics,
                "confusion_matrix": (
                    result.confusion_matrix
                ),
                "classification_report": (
                    result.classification_report
                ),
            }
            for name, result in (
                trained_results.items()
            )
        },
        "scientific_warning": (
            "This score represents phishing-oriented email "
            "characteristics. It is not a direct measurement "
            "of an individual person's weakness."
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
        best_result.estimator,
        BEST_MODEL_PATH,
    )

    test_scores = (
        split.metadata_test.copy()
    )

    test_scores[
        "actual_phishing_label"
    ] = split.y_test

    test_scores[
        "predicted_phishing_label"
    ] = best_result.predictions

    test_scores[
        "email_persuasion_probability"
    ] = best_result.probabilities

    test_scores[
        "email_persuasion_risk_score"
    ] = (
        best_result.probabilities
        * 100
    ).round(2)

    test_scores.to_csv(
        TEST_SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    full_model = best_result.estimator

    full_probabilities = (
        full_model.predict_proba(
            features
        )[:, 1]
    )

    full_predictions = (
        full_probabilities >= 0.50
    ).astype(int)

    full_scores = metadata.copy()

    full_scores[
        "actual_phishing_label"
    ] = target

    full_scores[
        "predicted_phishing_label"
    ] = full_predictions

    full_scores[
        "email_persuasion_probability"
    ] = full_probabilities

    full_scores[
        "email_persuasion_risk_score"
    ] = (
        full_probabilities
        * 100
    ).round(2)

    full_scores.to_csv(
        FULL_SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("AI-EMAIL MODEL COMPLETE")
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
        f"Selected model: {best_model_name}"
    )
    print(
        "Selected ROC-AUC:",
        round(
            best_result.metrics[
                "roc_auc"
            ],
            4,
        ),
    )
    print(
        "Selected Brier score:",
        round(
            best_result.metrics[
                "brier_score"
            ],
            4,
        ),
    )
    print()
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Test scores: {TEST_SCORES_PATH}")
    print(f"All scores: {FULL_SCORES_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())