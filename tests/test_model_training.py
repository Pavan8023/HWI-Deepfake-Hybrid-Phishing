from __future__ import annotations
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model_training import (
    ModelConfiguration,
    ModelTrainingError,
    build_base_estimators,
    build_calibrated_estimator,
    calculate_classification_metrics,
    cross_validate_estimator,
    extract_positive_probabilities,
    fit_and_evaluate_estimator,
    model_configuration_to_dict,
    select_best_hwi_model,
    train_awareness_model_suite,
    validate_modelling_inputs,
)


def create_modelling_dataset(
    rows: int = 300,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Create a reproducible binary test dataset."""

    features, target = make_classification(
        n_samples=rows,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )

    dataframe = pd.DataFrame(
        features,
        columns=[
            f"feature_{index}"
            for index in range(8)
        ],
    )

    target_series = pd.Series(
        target,
        name="clicked_link",
        dtype="int64",
    )

    return dataframe, target_series


def test_validate_modelling_inputs() -> None:
    x, y = create_modelling_dataset()

    validate_modelling_inputs(
        x,
        y,
        dataset_name="test",
    )


def test_validate_modelling_inputs_rejects_missing() -> None:
    x, y = create_modelling_dataset()

    x.loc[0, "feature_0"] = np.nan

    with pytest.raises(ModelTrainingError):
        validate_modelling_inputs(
            x,
            y,
            dataset_name="test",
        )


def test_build_base_estimators() -> None:
    estimators = build_base_estimators()

    assert set(estimators) == {
        "logistic_regression",
        "random_forest",
    }


def test_build_calibrated_estimator() -> None:
    estimator = build_base_estimators()[
        "logistic_regression"
    ]

    calibrated = build_calibrated_estimator(
        estimator
    )

    parameters = calibrated.get_params()

    assert parameters["method"] == "sigmoid"


def test_extract_positive_probabilities() -> None:
    x, y = create_modelling_dataset()

    estimator = cast(
    Any,
    build_base_estimators()[
        "logistic_regression"
    ],
)

    estimator.fit(x, y)

    probabilities = (
        extract_positive_probabilities(
            estimator,
            x,
        )
    )

    assert probabilities.shape == (
        len(x),
    )

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_calculate_classification_metrics() -> None:
    y_true = np.array(
        [0, 0, 1, 1],
    )

    probabilities = np.array(
        [0.1, 0.4, 0.6, 0.9],
    )

    (
        metrics,
        predictions,
        matrix,
        report,
    ) = calculate_classification_metrics(
        y_true,
        probabilities,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert predictions.tolist() == [
        0,
        0,
        1,
        1,
    ]
    assert matrix == [
        [2, 0],
        [0, 2],
    ]
    assert "1" in report


def test_fit_and_evaluate_estimator() -> None:
    x, y = create_modelling_dataset()

    x_train = x.iloc[:240].reset_index(
        drop=True
    )
    x_test = x.iloc[240:].reset_index(
        drop=True
    )
    y_train = y.iloc[:240].reset_index(
        drop=True
    )
    y_test = y.iloc[240:].reset_index(
        drop=True
    )

    estimator = build_base_estimators()[
        "logistic_regression"
    ]

    result = fit_and_evaluate_estimator(
        model_name="logistic_regression",
        estimator=estimator,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        calibrated=False,
    )

    assert result.model_name == (
        "logistic_regression"
    )
    assert len(result.probabilities) == 60
    assert 0 <= result.metrics[
        "roc_auc"
    ] <= 1


def test_cross_validate_estimator() -> None:
    x, y = create_modelling_dataset()

    configuration = ModelConfiguration(
        cv_folds=3,
        n_jobs=1,
    )

    estimator = build_base_estimators(
        configuration=configuration
    )["logistic_regression"]

    summary = cross_validate_estimator(
        estimator=estimator,
        x_train=x,
        y_train=y,
        configuration=configuration,
    )

    assert "roc_auc_mean" in summary
    assert "brier_score_mean" in summary


def test_train_awareness_model_suite() -> None:
    x, y = create_modelling_dataset(
        rows=240
    )

    x_train = x.iloc[:180].reset_index(
        drop=True
    )
    x_test = x.iloc[180:].reset_index(
        drop=True
    )
    y_train = y.iloc[:180].reset_index(
        drop=True
    )
    y_test = y.iloc[180:].reset_index(
        drop=True
    )

    configuration = ModelConfiguration(
        cv_folds=3,
        n_jobs=1,
    )

    (
        results,
        comparison,
        cross_validation,
    ) = train_awareness_model_suite(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        configuration=configuration,
    )

    assert set(results) == {
        "logistic_regression",
        "logistic_regression_calibrated",
        "random_forest",
        "random_forest_calibrated",
    }

    assert len(comparison) == 4

    assert set(cross_validation) == {
        "logistic_regression",
        "random_forest",
    }


def test_select_best_hwi_model() -> None:
    comparison = pd.DataFrame(
        {
            "model_name": [
                "model_a_calibrated",
                "model_b_calibrated",
                "model_c",
            ],
            "calibrated": [
                True,
                True,
                False,
            ],
            "roc_auc": [
                0.80,
                0.80,
                0.90,
            ],
            "brier_score": [
                0.20,
                0.18,
                0.15,
            ],
            "f1": [
                0.75,
                0.74,
                0.82,
            ],
        }
    )

    best_model = select_best_hwi_model(
        comparison
    )

    assert best_model == (
        "model_b_calibrated"
    )


def test_model_configuration_to_dict() -> None:
    result = model_configuration_to_dict(
        ModelConfiguration(
            cv_folds=3,
            n_jobs=1,
        )
    )

    assert result["cv_folds"] == 3
    assert result["n_jobs"] == 1