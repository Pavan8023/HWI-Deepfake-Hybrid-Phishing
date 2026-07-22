from __future__ import annotations

import pandas as pd
import pytest

from src.evidence_profiles import (
    build_evidence_profiles,
    profile_summary_markdown,
    profiles_to_dataframe,
    quantile_to_profile_name,
    select_nearest_quantile_rows,
    validate_quantiles,
)
from src.hwi_framework import HWIFrameworkError


def test_validate_quantiles() -> None:
    assert validate_quantiles(
        [0.9, 0.1, 0.5]
    ) == (
        0.1,
        0.5,
        0.9,
    )


def test_validate_quantiles_rejects_invalid() -> None:
    with pytest.raises(HWIFrameworkError):
        validate_quantiles(
            [-0.1, 0.5]
        )


def test_select_nearest_quantile_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "hwi_score": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
            ]
        }
    )

    selected = select_nearest_quantile_rows(
        dataframe,
        score_column="hwi_score",
        dataset_name="awareness",
    )

    assert len(selected) == 3
    assert (
        selected[
            "selected_row_index"
        ].nunique()
        == 3
    )


def test_quantile_to_profile_name() -> None:
    assert (
        quantile_to_profile_name(0.1)
        == "lower_observed_context"
    )

    assert (
        quantile_to_profile_name(0.5)
        == "middle_observed_context"
    )

    assert (
        quantile_to_profile_name(0.9)
        == "higher_observed_context"
    )


def test_build_evidence_profiles() -> None:
    behavioural = pd.DataFrame(
        {
            "hwi_score": [45, 50, 55],
            "selection_quantile": [
                0.1,
                0.5,
                0.9,
            ],
            "selected_row_index": [
                1,
                2,
                3,
            ],
        }
    )

    email = pd.DataFrame(
        {
            "email_persuasion_risk_score": [
                5,
                50,
                95,
            ],
            "selection_quantile": [
                0.1,
                0.5,
                0.9,
            ],
            "selected_row_index": [
                4,
                5,
                6,
            ],
        }
    )

    url = pd.DataFrame(
        {
            "url_technical_risk_score": [
                3,
                60,
                98,
            ],
            "selection_quantile": [
                0.1,
                0.5,
                0.9,
            ],
            "selected_row_index": [
                7,
                8,
                9,
            ],
        }
    )

    profiles, audit = (
        build_evidence_profiles(
            behavioural_rows=behavioural,
            email_rows=email,
            url_rows=url,
        )
    )

    assert len(profiles) == 3
    assert len(audit) == 3
    assert not audit[
        "records_are_paired"
    ].any()


def test_profiles_to_dataframe() -> None:
    behavioural = pd.DataFrame(
        {
            "hwi_score": [50],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    email = pd.DataFrame(
        {
            "email_persuasion_risk_score": [
                90
            ],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    url = pd.DataFrame(
        {
            "url_technical_risk_score": [
                80
            ],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    profiles, _ = build_evidence_profiles(
        behavioural_rows=behavioural,
        email_rows=email,
        url_rows=url,
    )

    frame = profiles_to_dataframe(
        profiles,
        profile_names=[
            "middle_observed_context"
        ],
    )

    assert len(frame) == 1
    assert (
        frame.loc[0, "profile_name"]
        == "middle_observed_context"
    )


def test_profile_summary_markdown() -> None:
    behavioural = pd.DataFrame(
        {
            "hwi_score": [50],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    email = pd.DataFrame(
        {
            "email_persuasion_risk_score": [
                90
            ],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    url = pd.DataFrame(
        {
            "url_technical_risk_score": [
                80
            ],
            "selection_quantile": [0.5],
            "selected_row_index": [0],
        }
    )

    profiles, _ = build_evidence_profiles(
        behavioural_rows=behavioural,
        email_rows=email,
        url_rows=url,
    )

    frame = profiles_to_dataframe(
        profiles,
        profile_names=[
            "middle_observed_context"
        ],
    )

    markdown = profile_summary_markdown(
        frame
    )

    assert "# HWI Evidence Profiles" in markdown
    assert (
        "do not represent verified paired records"
        in markdown
    )