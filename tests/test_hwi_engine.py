from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.hwi_engine import (
    HWIConfiguration,
    HWIError,
    HWIThresholds,
    IndicatorSpecification,
    build_transparent_proxy_hwi,
    classify_hwi_score,
    classify_hwi_scores,
    enrich_with_model_hwi,
    hwi_configuration_to_dict,
    percentile_risk_score,
    probability_to_hwi,
    save_hwi_outputs,
    validate_hwi_against_binary_outcome,
    validate_hwi_distribution,
    validate_indicator_specifications,
    validate_probability_values,
)


def test_validate_probability_values() -> None:
    probabilities = validate_probability_values(
        [0.0, 0.5, 1.0]
    )

    assert probabilities.tolist() == [
        0.0,
        0.5,
        1.0,
    ]


def test_probability_validation_rejects_invalid_values() -> None:
    with pytest.raises(HWIError):
        validate_probability_values(
            [-0.1, 0.5, 1.1]
        )


def test_probability_to_hwi() -> None:
    scores = probability_to_hwi(
        [0.10, 0.505, 0.90]
    )

    assert scores.tolist() == [
        10.0,
        50.5,
        90.0,
    ]


def test_classify_hwi_score() -> None:
    assert classify_hwi_score(20) == "low"
    assert classify_hwi_score(50) == "medium"
    assert classify_hwi_score(90) == "high"


def test_classify_hwi_scores() -> None:
    categories = classify_hwi_scores(
        [10, 50, 90]
    )

    assert categories.tolist() == [
        "low",
        "medium",
        "high",
    ]


def test_enrich_with_model_hwi() -> None:
    dataframe = pd.DataFrame(
        {
            "record_id": ["A", "B", "C"],
        }
    )

    enriched = enrich_with_model_hwi(
        dataframe,
        [0.10, 0.50, 0.90],
    )

    assert enriched["hwi_score"].tolist() == [
        10.0,
        50.0,
        90.0,
    ]

    assert enriched["hwi_category"].tolist() == [
        "low",
        "medium",
        "high",
    ]

    assert (
        enriched["hwi_method"]
        == "model_probability"
    ).all()


def test_percentile_risk_higher_is_risk() -> None:
    series = pd.Series(
        [10, 20, 30],
        name="indicator",
    )

    risk = percentile_risk_score(
        series,
        direction="higher_is_risk",
    )

    assert risk.iloc[0] < risk.iloc[1]
    assert risk.iloc[1] < risk.iloc[2]


def test_percentile_risk_lower_is_risk() -> None:
    series = pd.Series(
        [10, 20, 30],
        name="indicator",
    )

    risk = percentile_risk_score(
        series,
        direction="lower_is_risk",
    )

    assert risk.iloc[0] > risk.iloc[1]
    assert risk.iloc[1] > risk.iloc[2]


def test_validate_indicator_specifications() -> None:
    dataframe = pd.DataFrame(
        {
            "hover_time": [100, 200],
            "session_duration": [20, 30],
        }
    )

    specifications = [
        IndicatorSpecification(
            column="hover_time",
            weight=0.5,
            direction="lower_is_risk",
        ),
        IndicatorSpecification(
            column="session_duration",
            weight=0.5,
            direction="lower_is_risk",
        ),
    ]

    validate_indicator_specifications(
        dataframe,
        specifications,
    )


def test_indicator_validation_rejects_target_column() -> None:
    dataframe = pd.DataFrame(
        {
            "clicked_link": [0, 1],
        }
    )

    specifications = [
        IndicatorSpecification(
            column="clicked_link",
            weight=1.0,
            direction="higher_is_risk",
        )
    ]

    with pytest.raises(HWIError):
        validate_indicator_specifications(
            dataframe,
            specifications,
        )


def test_indicator_weights_must_sum_to_one() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    specifications = [
        IndicatorSpecification(
            column="a",
            weight=0.2,
            direction="higher_is_risk",
        ),
        IndicatorSpecification(
            column="b",
            weight=0.3,
            direction="higher_is_risk",
        ),
    ]

    with pytest.raises(HWIError):
        validate_indicator_specifications(
            dataframe,
            specifications,
        )


def test_build_transparent_proxy_hwi() -> None:
    dataframe = pd.DataFrame(
        {
            "hover_time": [100, 200, 300],
            "session_duration": [10, 20, 30],
        }
    )

    specifications = [
        IndicatorSpecification(
            column="hover_time",
            weight=0.5,
            direction="lower_is_risk",
        ),
        IndicatorSpecification(
            column="session_duration",
            weight=0.5,
            direction="lower_is_risk",
        ),
    ]

    result = build_transparent_proxy_hwi(
        dataframe,
        specifications,
    )

    assert "hwi_score" in result.columns
    assert "hwi_category" in result.columns

    first_score_value = result.at[
        0,
        "hwi_score",
    ]

    last_score_value = result.at[
        2,
        "hwi_score",
    ]

    assert isinstance(
        first_score_value,
        (int, float),
    )

    assert isinstance(
        last_score_value,
        (int, float),
    )

    first_score = float(first_score_value)
    last_score = float(last_score_value)

    assert first_score > last_score

    assert (
        result["hwi_method"]
        == "transparent_proxy"
    ).all()


def test_validate_hwi_distribution() -> None:
    dataframe = pd.DataFrame(
        {
            "hwi_score": [10, 50, 90],
            "hwi_category": [
                "low",
                "medium",
                "high",
            ],
        }
    )

    summary = validate_hwi_distribution(
        dataframe
    )

    assert summary["rows"] == 3
    assert summary["minimum"] == 10
    assert summary["maximum"] == 90
    assert summary["invalid_score_count"] == 0


def test_validate_hwi_against_binary_outcome() -> None:
    dataframe = pd.DataFrame(
        {
            "clicked_link": [
                "no",
                "no",
                "yes",
                "yes",
            ],
            "hwi_score": [
                10,
                20,
                70,
                90,
            ],
        }
    )

    result = (
        validate_hwi_against_binary_outcome(
            dataframe,
            outcome_column="clicked_link",
        )
    )

    assert result["positive_mean_hwi"] > (
        result["negative_mean_hwi"]
    )

    assert result["mean_difference"] > 0


def test_save_hwi_outputs(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "hwi_score": [10, 50, 90],
            "hwi_category": [
                "low",
                "medium",
                "high",
            ],
        }
    )

    output_paths = save_hwi_outputs(
        dataframe,
        dataset_output_path=(
            tmp_path / "enriched.csv"
        ),
        summary_output_path=(
            tmp_path / "summary.json"
        ),
    )

    assert output_paths["dataset"].exists()
    assert output_paths["summary"].exists()


def test_configuration_to_dict() -> None:
    configuration = HWIConfiguration(
        thresholds=HWIThresholds(
            low_max=25,
            medium_max=75,
        )
    )

    result = hwi_configuration_to_dict(
        configuration
    )

    assert result["thresholds"]["low_max"] == 25
    assert result["thresholds"]["medium_max"] == 75