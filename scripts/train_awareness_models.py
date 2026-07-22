from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
)
from src.hwi_engine import (  # noqa: E402
    enrich_with_model_hwi,
    save_hwi_outputs,
)
from src.model_training import (  # noqa: E402
    DEFAULT_MODEL_CONFIGURATION,
    model_configuration_to_dict,
    select_best_hwi_model,
    train_awareness_model_suite,
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

MODEL_REPORT_DIR = (
    OUTPUTS_DIR
    / "reports"
    / "model_training"
)
PREDICTIONS_DIR = (
    OUTPUTS_DIR
    / "predictions"
)

COMPARISON_PATH = (
    MODEL_REPORT_DIR
    / "awareness_model_comparison.csv"
)
METRICS_PATH = (
    MODEL_REPORT_DIR
    / "awareness_model_metrics.json"
)
CV_PATH = (
    MODEL_REPORT_DIR
    / "awareness_cross_validation.json"
)
BEST_MODEL_PATH = (
    MODELS_DIR
    / "awareness_best_calibrated_model.joblib"
)
HWI_DATASET_PATH = (
    PREDICTIONS_DIR
    / "awareness_test_hwi.csv"
)
HWI_SUMMARY_PATH = (
    MODEL_REPORT_DIR
    / "awareness_test_hwi_summary.json"
)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Load fixed awareness train/test partitions."""

    required_paths = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing modelling files: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    x_train = pd.read_csv(
        X_TRAIN_PATH
    )
    x_test = pd.read_csv(
        X_TEST_PATH
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH
    )["clicked_link"].astype("int64")

    y_test = pd.read_csv(
        Y_TEST_PATH
    )["clicked_link"].astype("int64")

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


def main() -> int:
    """Train, compare and save awareness susceptibility models."""

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_inputs()

    print("=" * 80)
    print("AWARENESS SUSCEPTIBILITY MODEL TRAINING")
    print("=" * 80)
    print(f"Training shape: {x_train.shape}")
    print(f"Testing shape: {x_test.shape}")
    print(
        "Training target:",
        y_train.value_counts()
        .sort_index()
        .to_dict(),
    )
    print(
        "Testing target:",
        y_test.value_counts()
        .sort_index()
        .to_dict(),
    )
    print()

    (
        trained_results,
        comparison,
        cross_validation,
    ) = train_awareness_model_suite(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
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

    MODEL_REPORT_DIR.mkdir(
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

    metrics_payload = {
        "best_hwi_model": best_model_name,
        "configuration": (
            model_configuration_to_dict()
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
            "Performance measures association with the observed "
            "clicked_link outcome in this dataset. It does not prove "
            "a universal psychological weakness construct."
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

    with CV_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cross_validation,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    joblib.dump(
        best_result.estimator,
        BEST_MODEL_PATH,
    )

    hwi_input = pd.DataFrame(
        {
            "actual_clicked_link": (
                y_test.reset_index(
                    drop=True
                )
            ),
            "predicted_clicked_link": (
                best_result.predictions
            ),
        }
    )

    hwi_dataset = enrich_with_model_hwi(
        hwi_input,
        best_result.probabilities,
    )

    save_hwi_outputs(
        hwi_dataset,
        dataset_output_path=(
            HWI_DATASET_PATH
        ),
        summary_output_path=(
            HWI_SUMMARY_PATH
        ),
        additional_metadata={
            "model_name": (
                best_model_name
            ),
            "dataset_partition": (
                "untouched_test_partition"
            ),
            "hwi_definition": (
                "Calibrated predicted probability "
                "of clicked_link multiplied by 100."
            ),
        },
    )

    print("=" * 80)
    print("MODEL TRAINING COMPLETE")
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
        ].round(4).to_string(
            index=False
        )
    )
    print()
    print(
        f"Selected HWI model: "
        f"{best_model_name}"
    )
    print(
        f"Selected ROC-AUC: "
        f"{best_result.metrics['roc_auc']:.4f}"
    )
    print(
        f"Selected Brier score: "
        f"{best_result.metrics['brier_score']:.4f}"
    )
    print()
    print(f"Comparison: {COMPARISON_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"HWI dataset: {HWI_DATASET_PATH}")
    print(f"HWI summary: {HWI_SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())