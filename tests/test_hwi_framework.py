from __future__ import annotations

import pytest

from src.hwi_framework import (
    HWIFrameworkError,
    METHODOLOGICAL_WARNING,
    ScoreThresholds,
    classify_score,
    create_evidence_profile,
    create_profile_from_rows,
    profile_to_dataframe,
    profile_to_dict,
    validate_evidence_status,
    validate_score,
)


def test_validate_score() -> None:
    assert validate_score(
        50,
        score_name="test_score",
    ) == 50.0


def test_validate_score_rejects_out_of_range() -> None:
    with pytest.raises(HWIFrameworkError):
        validate_score(
            101,
            score_name="test_score",
        )


def test_classify_score() -> None:
    assert classify_score(20) == "low"
    assert classify_score(50) == "medium"
    assert classify_score(90) == "high"


def test_custom_thresholds() -> None:
    thresholds = ScoreThresholds(
        low_max=25,
        medium_max=75,
    )

    assert classify_score(
        25,
        thresholds=thresholds,
    ) == "low"

    assert classify_score(
        60,
        thresholds=thresholds,
    ) == "medium"

    assert classify_score(
        80,
        thresholds=thresholds,
    ) == "high"


def test_validate_evidence_status() -> None:
    assert (
        validate_evidence_status(
            "unsupported"
        )
        == "unsupported"
    )


def test_validate_evidence_status_rejects_unknown() -> None:
    with pytest.raises(HWIFrameworkError):
        validate_evidence_status(
            "perfect"
        )


def test_create_evidence_profile() -> None:
    profile = create_evidence_profile(
        behavioural_hwi_score=50,
        email_persuasion_risk_score=90,
        url_technical_risk_score=80,
        behavioural_evidence_status=(
            "unsupported"
        ),
    )

    assert (
        profile.behavioural_hwi_category
        == "medium"
    )

    assert (
        profile.email_persuasion_category
        == "high"
    )

    assert profile.url_risk_category == "high"

    assert (
        profile.behavioural_evidence_status
        == "unsupported"
    )

    assert (
        profile.email_evidence_status
        == "context_only"
    )

    assert (
        profile.url_evidence_status
        == "context_only"
    )

    assert (
        profile.methodological_warning
        == METHODOLOGICAL_WARNING
    )


def test_profile_does_not_contain_composite_score() -> None:
    profile = create_evidence_profile(
        behavioural_hwi_score=50,
        email_persuasion_risk_score=90,
        url_technical_risk_score=80,
    )

    result = profile_to_dict(profile)

    assert "composite_hwi" not in result
    assert "hybrid_score" not in result
    assert "overall_score" not in result


def test_profile_to_dataframe() -> None:
    profile = create_evidence_profile(
        behavioural_hwi_score=10,
        email_persuasion_risk_score=50,
        url_technical_risk_score=90,
    )

    frame = profile_to_dataframe(
        profile
    )

    assert len(frame) == 1
    assert (
        frame.loc[
            0,
            "behavioural_hwi_score",
        ]
        == 10
    )


def test_create_profile_from_rows() -> None:
    profile = create_profile_from_rows(
        behavioural_row={
            "hwi_score": 51.2,
        },
        email_row={
            "email_persuasion_risk_score": (
                94.5
            ),
        },
        url_row={
            "url_technical_risk_score": (
                82.1
            ),
        },
    )

    assert (
        profile.behavioural_hwi_score
        == 51.2
    )

    assert (
        profile.email_persuasion_risk_score
        == 94.5
    )

    assert (
        profile.url_technical_risk_score
        == 82.1
    )


def test_create_profile_from_rows_rejects_missing_score() -> None:
    with pytest.raises(HWIFrameworkError):
        create_profile_from_rows(
            behavioural_row={
                "hwi_score": 50,
            },
            email_row={},
            url_row={
                "url_technical_risk_score": 80,
            },
        )


def test_interpretation_mentions_separate_datasets() -> None:
    profile = create_evidence_profile(
        behavioural_hwi_score=50,
        email_persuasion_risk_score=90,
        url_technical_risk_score=80,
    )

    assert (
        "datasets do not contain verified "
        "shared identifiers"
        in profile.interpretation
    )