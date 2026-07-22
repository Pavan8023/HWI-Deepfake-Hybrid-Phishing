from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CV_FOLDS, RANDOM_STATE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ModelTrainingError(RuntimeError):
    """Raised when model training or evaluation cannot be completed safely."""


@dataclass(frozen=True)
class ModelConfiguration:
    """Shared configuration for awareness-model training."""

    random_state: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS
    calibration_method: CalibrationMethod = "sigmoid"
    probability_threshold: float = 0.50
    n_jobs: int = -1

    def validate(self) -> None:
        """Validate model-training settings."""

        if self.cv_folds < 2:
            raise ModelTrainingError(
                "cv_folds must be at least two."
            )

        if self.calibration_method not in {
            "sigmoid",
            "isotonic",
        }:
            raise ModelTrainingError(
                "calibration_method must be sigmoid or isotonic."
            )

        if not 0 < self.probability_threshold < 1:
            raise ModelTrainingError(
                "probability_threshold must be between zero and one."
            )

CalibrationMethod = Literal[
    "sigmoid",
    "isotonic",
]

DEFAULT_MODEL_CONFIGURATION = ModelConfiguration()


@dataclass
class TrainedModelResult:
    """A fitted model and its evaluation outputs."""

    model_name: str
    estimator: BaseEstimator
    calibrated: bool
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    probabilities: np.ndarray
    predictions: np.ndarray


def validate_modelling_inputs(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_name: str,
) -> None:
    """Validate predictors and target before model use."""

    if x.empty:
        raise ModelTrainingError(
            f"{dataset_name} predictors are empty."
        )

    if y.empty:
        raise ModelTrainingError(
            f"{dataset_name} target is empty."
        )

    if len(x) != len(y):
        raise ModelTrainingError(
            f"{dataset_name} predictor and target row counts differ."
        )

    if y.isna().any():
        raise ModelTrainingError(
            f"{dataset_name} target contains missing values."
        )

    if y.nunique() != 2:
        raise ModelTrainingError(
            f"{dataset_name} target must contain exactly two classes."
        )

    numeric_x = x.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_x.isna().any().any():
        raise ModelTrainingError(
            f"{dataset_name} predictors contain missing or non-numeric values."
        )

    values = numeric_x.to_numpy(
        dtype=float,
    )

    if np.isinf(values).any():
        raise ModelTrainingError(
            f"{dataset_name} predictors contain infinite values."
        )


def build_base_estimators(
    *,
    configuration: ModelConfiguration = DEFAULT_MODEL_CONFIGURATION,
) -> dict[str, BaseEstimator]:
    """Build the initial interpretable and nonlinear baseline models."""

    configuration.validate()

    logistic_regression = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=configuration.random_state,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=configuration.random_state,
        n_jobs=configuration.n_jobs,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }

def build_calibrated_estimator(
    estimator: BaseEstimator,
    *,
    configuration: ModelConfiguration = DEFAULT_MODEL_CONFIGURATION,
) -> CalibratedClassifierCV:
    """Wrap a classifier in cross-validated probability calibration."""

    configuration.validate()

    cross_validator = StratifiedKFold(
        n_splits=configuration.cv_folds,
        shuffle=True,
        random_state=configuration.random_state,
    )

    return CalibratedClassifierCV(
    estimator=clone(estimator),
    method=configuration.calibration_method,
    cv=cross_validator,
    n_jobs=configuration.n_jobs,
)


def extract_positive_probabilities(
    estimator: BaseEstimator,
    x: pd.DataFrame,
) -> np.ndarray:
    """Extract validated probabilities for the positive class."""

    if not hasattr(estimator, "predict_proba"):
        raise ModelTrainingError(
            "Estimator does not expose predict_proba()."
        )

    probability_matrix = estimator.predict_proba(x)  # type: ignore[attr-defined]

    probabilities = np.asarray(
        probability_matrix,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ModelTrainingError(
            "predict_proba must return a two-dimensional array."
        )

    if probabilities.shape[1] != 2:
        raise ModelTrainingError(
            "Binary classification requires two probability columns."
        )

    positive_probabilities = probabilities[:, 1]

    if (
        np.isnan(positive_probabilities).any()
        or np.isinf(positive_probabilities).any()
    ):
        raise ModelTrainingError(
            "Predicted probabilities contain invalid values."
        )

    if (
        (positive_probabilities < 0).any()
        or (positive_probabilities > 1).any()
    ):
        raise ModelTrainingError(
            "Predicted probabilities must remain between zero and one."
        )

    return positive_probabilities


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.50,
) -> tuple[
    dict[str, float],
    np.ndarray,
    list[list[int]],
    dict[str, Any],
]:
    """Calculate classification and probability-quality metrics."""

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between zero and one."
        )

    true_values = np.asarray(
        y_true,
        dtype=int,
    )

    probability_values = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(true_values) != len(probability_values):
        raise ModelTrainingError(
            "Target and probability row counts differ."
        )

    predictions = (
        probability_values >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                true_values,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_values,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                true_values,
                probability_values,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                true_values,
                probability_values,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                true_values,
                probability_values,
            )
        ),
        "log_loss": float(
            log_loss(
                true_values,
                np.column_stack(
                    [
                        1.0 - probability_values,
                        probability_values,
                    ]
                ),
                labels=[0, 1],
            )
        ),
        "positive_prediction_rate": float(
            predictions.mean()
        ),
        "probability_mean": float(
            probability_values.mean()
        ),
        "probability_standard_deviation": float(
            probability_values.std()
        ),
    }

    matrix = confusion_matrix(
        true_values,
        predictions,
        labels=[0, 1],
    ).tolist()

    raw_report = classification_report(
    true_values,
    predictions,
    labels=[0, 1],
    output_dict=True,
    zero_division=0,
)

    if not isinstance(raw_report, dict):
        raise ModelTrainingError(
            "classification_report did not return a dictionary."
        )

    report = cast(
        dict[str, Any],
        raw_report,
    )

    return (
        metrics,
        predictions,
        matrix,
        report,
    )


def fit_and_evaluate_estimator(
    *,
    model_name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    calibrated: bool,
    threshold: float = 0.50,
) -> TrainedModelResult:
    """Fit one estimator and evaluate it on the untouched test set."""

    validate_modelling_inputs(
        x_train,
        y_train,
        dataset_name="training",
    )

    validate_modelling_inputs(
        x_test,
        y_test,
        dataset_name="testing",
    )

    if list(x_train.columns) != list(x_test.columns):
        raise ModelTrainingError(
            "Training and testing predictor columns do not match."
        )

    fitted_estimator = clone(
        estimator
    )

    fitted_estimator.fit(
        x_train,
        y_train,
    )

    probabilities = extract_positive_probabilities(
        fitted_estimator,
        x_test,
    )

    (
        metrics,
        predictions,
        matrix,
        report,
    ) = calculate_classification_metrics(
        y_test,
        probabilities,
        threshold=threshold,
    )

    return TrainedModelResult(
        model_name=model_name,
        estimator=fitted_estimator,
        calibrated=calibrated,
        metrics=metrics,
        confusion_matrix=matrix,
        classification_report=report,
        probabilities=probabilities,
        predictions=predictions,
    )


def cross_validate_estimator(
    *,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    configuration: ModelConfiguration = DEFAULT_MODEL_CONFIGURATION,
) -> dict[str, float]:
    """Cross-validate a model using only the training partition."""

    configuration.validate()

    validate_modelling_inputs(
        x_train,
        y_train,
        dataset_name="cross-validation training",
    )

    splitter = StratifiedKFold(
        n_splits=configuration.cv_folds,
        shuffle=True,
        random_state=configuration.random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "neg_brier_score": "neg_brier_score",
    }

    results = cross_validate(
        clone(estimator),
        x_train,
        y_train,
        cv=splitter,
        scoring=scoring,
        n_jobs=configuration.n_jobs,
        return_train_score=False,
        error_score="raise",
    )

    summary: dict[str, float] = {}

    for scoring_name in scoring:
        test_key = f"test_{scoring_name}"
        values = np.asarray(
            results[test_key],
            dtype=float,
        )

        if scoring_name == "neg_brier_score":
            values = -values
            output_name = "brier_score"
        else:
            output_name = scoring_name

        summary[f"{output_name}_mean"] = float(
            values.mean()
        )
        summary[f"{output_name}_std"] = float(
            values.std()
        )

    return summary


def train_awareness_model_suite(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    configuration: ModelConfiguration = DEFAULT_MODEL_CONFIGURATION,
) -> tuple[
    dict[str, TrainedModelResult],
    pd.DataFrame,
    dict[str, dict[str, float]],
]:
    """
    Train uncalibrated and calibrated awareness-model baselines.

    Model selection should prioritise:
    1. probability discrimination;
    2. calibration quality;
    3. recall/F1;
    4. interpretability.
    """

    configuration.validate()

    base_estimators = build_base_estimators(
        configuration=configuration
    )

    trained_results: dict[
        str,
        TrainedModelResult,
    ] = {}

    cross_validation_results: dict[
        str,
        dict[str, float],
    ] = {}

    for base_name, estimator in base_estimators.items():
        cross_validation_results[base_name] = (
            cross_validate_estimator(
                estimator=estimator,
                x_train=x_train,
                y_train=y_train,
                configuration=configuration,
            )
        )

        base_result = fit_and_evaluate_estimator(
            model_name=base_name,
            estimator=estimator,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            calibrated=False,
            threshold=(
                configuration.probability_threshold
            ),
        )

        trained_results[base_name] = (
            base_result
        )

        calibrated_name = (
            f"{base_name}_calibrated"
        )

        calibrated_estimator = (
            build_calibrated_estimator(
                estimator,
                configuration=configuration,
            )
        )

        calibrated_result = (
            fit_and_evaluate_estimator(
                model_name=calibrated_name,
                estimator=calibrated_estimator,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                calibrated=True,
                threshold=(
                    configuration
                    .probability_threshold
                ),
            )
        )

        trained_results[calibrated_name] = (
            calibrated_result
        )

    comparison_records: list[
        dict[str, Any]
    ] = []

    for model_name, result in (
        trained_results.items()
    ):
        comparison_records.append(
            {
                "model_name": model_name,
                "calibrated": (
                    result.calibrated
                ),
                **result.metrics,
            }
        )

    comparison = pd.DataFrame(
        comparison_records
    )

    comparison = comparison.sort_values(
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
    ).reset_index(drop=True)

    return (
        trained_results,
        comparison,
        cross_validation_results,
    )


def select_best_hwi_model(
    comparison: pd.DataFrame,
) -> str:
    """
    Select the model used to produce HWI probabilities.

    Only calibrated models are eligible. Ranking prioritises ROC-AUC and
    then lower Brier score.
    """

    required_columns = {
        "model_name",
        "calibrated",
        "roc_auc",
        "brier_score",
        "f1",
    }

    missing_columns = required_columns.difference(
        comparison.columns
    )

    if missing_columns:
        raise ModelTrainingError(
            "Model-comparison table is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    calibrated_results = comparison.loc[
        comparison["calibrated"].astype(bool)
    ].copy()

    if calibrated_results.empty:
        raise ModelTrainingError(
            "No calibrated model is available for HWI generation."
        )

    ranked = calibrated_results.sort_values(
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

    return str(
        ranked.iloc[0]["model_name"]
    )


def model_configuration_to_dict(
    configuration: ModelConfiguration = DEFAULT_MODEL_CONFIGURATION,
) -> Mapping[str, Any]:
    """Convert configuration into JSON-compatible metadata."""

    configuration.validate()

    return asdict(configuration)