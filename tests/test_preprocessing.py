from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    AwarenessSchema,
    PreprocessingError,
    awareness_schema_to_dict,
    build_awareness_preprocessor,
    classify_time_period,
    derive_datetime_features,
    encode_binary_target,
    fit_transform_awareness_split,
    identify_awareness_feature_types,
    prepare_awareness_dataframe,
    preprocess_awareness_dataset,
    stratified_awareness_split,
    validate_awareness_columns,
)


def create_awareness_dataframe(
    rows: int = 40,
) -> pd.DataFrame:
    """Create a balanced synthetic awareness dataset."""

    clicked_values = [
        "yes" if index % 2 else "no"
        for index in range(rows)
    ]

    return pd.DataFrame(
        {
            "user_id": [
                f"user_{index}"
                for index in range(rows)
            ],
            "email_subject": [
                (
                    "Urgent account verification"
                    if index % 2
                    else "Weekly project update"
                )
                for index in range(rows)
            ],
            "sender_email_domain": [
                (
                    "secure-login-example.com"
                    if index % 2
                    else "company.example"
                )
                for index in range(rows)
            ],
            "hover_time_ms": [
                1000 + index * 50
                for index in range(rows)
            ],
            "clicked_link": clicked_values,
            "reported_email": [
                "no" if index % 2 else "yes"
                for index in range(rows)
            ],
            "device_type": [
                (
                    "mobile"
                    if index % 3
                    else "desktop"
                )
                for index in range(rows)
            ],
            "browser_used": [
                (
                    "chrome"
                    if index % 2
                    else "firefox"
                )
                for index in range(rows)
            ],
            "email_received_time": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="6h",
            ).astype(str),
            "session_duration_sec": [
                20 + index
                for index in range(rows)
            ],
            "geo_location": [
                "53.3498,-6.2603"
                for _ in range(rows)
            ],
            "email_language": [
                "English"
                for _ in range(rows)
            ],
        }
    )


def test_validate_awareness_columns() -> None:
    dataframe = create_awareness_dataframe()

    validate_awareness_columns(dataframe)


def test_validate_awareness_columns_rejects_missing_column() -> None:
    dataframe = (
        create_awareness_dataframe()
        .drop(columns=["clicked_link"])
    )

    with pytest.raises(PreprocessingError):
        validate_awareness_columns(dataframe)


def test_encode_binary_target() -> None:
    encoded = encode_binary_target(
        pd.Series(
            ["yes", "no", "1", "0"]
        )
    )

    assert encoded.tolist() == [
        1,
        0,
        1,
        0,
    ]


def test_encode_binary_target_rejects_unknown_value() -> None:
    with pytest.raises(PreprocessingError):
        encode_binary_target(
            pd.Series(
                ["yes", "maybe", "no"]
            )
        )


def test_classify_time_period() -> None:
    assert classify_time_period(8) == "morning"
    assert classify_time_period(13) == "afternoon"
    assert classify_time_period(18) == "evening"
    assert classify_time_period(23) == "night"


def test_derive_datetime_features() -> None:
    result = derive_datetime_features(
        pd.Series(
            [
                "2026-01-03 09:30:00",
                "2026-01-05 22:15:00",
            ]
        )
    )

    assert result["received_hour"].tolist() == [
        9,
        22,
    ]

    assert result[
        "received_weekend"
    ].tolist() == [
        1,
        0,
    ]

    assert result[
        "received_time_period"
    ].tolist() == [
        "morning",
        "night",
    ]


def test_prepare_awareness_dataframe_excludes_leakage() -> None:
    dataframe = create_awareness_dataframe()

    features, target = (
        prepare_awareness_dataframe(
            dataframe
        )
    )

    assert "clicked_link" not in features.columns
    assert "user_id" not in features.columns
    assert "geo_location" not in features.columns
    assert "reported_email" not in features.columns
    assert "email_received_time" not in features.columns

    assert "received_hour" in features.columns
    assert (
        "received_day_of_week"
        in features.columns
    )
    assert "received_weekend" in features.columns

    assert set(target.unique()) == {0, 1}


def test_prepare_awareness_dataframe_can_include_reported_email() -> None:
    dataframe = create_awareness_dataframe()

    features, _ = prepare_awareness_dataframe(
        dataframe,
        include_reported_email=True,
    )

    assert "reported_email" in features.columns


def test_negative_numeric_values_become_missing() -> None:
    dataframe = create_awareness_dataframe()

    dataframe.loc[
        0,
        "hover_time_ms",
    ] = -10

    features, _ = prepare_awareness_dataframe(
        dataframe
    )

    assert pd.isna(
        features.loc[
            0,
            "hover_time_ms",
        ]
    )


def test_stratified_awareness_split() -> None:
    dataframe = create_awareness_dataframe(
        rows=100
    )

    features, target = (
        prepare_awareness_dataframe(
            dataframe
        )
    )

    split = stratified_awareness_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    assert len(split.x_train) == 80
    assert len(split.x_test) == 20

    assert split.y_train.mean() == (
        pytest.approx(0.5)
    )

    assert split.y_test.mean() == (
        pytest.approx(0.5)
    )


def test_identify_awareness_feature_types() -> None:
    dataframe = create_awareness_dataframe()

    features, _ = prepare_awareness_dataframe(
        dataframe
    )

    numeric_columns, categorical_columns = (
        identify_awareness_feature_types(
            features
        )
    )

    assert "hover_time_ms" in numeric_columns
    assert (
        "session_duration_sec"
        in numeric_columns
    )
    assert "device_type" in categorical_columns
    assert "email_subject" in categorical_columns


def test_build_awareness_preprocessor() -> None:
    preprocessor = build_awareness_preprocessor(
        numeric_columns=[
            "hover_time_ms",
        ],
        categorical_columns=[
            "device_type",
        ],
    )

    assert preprocessor is not None


def test_fit_transform_awareness_split() -> None:
    dataframe = create_awareness_dataframe(
        rows=100
    )

    features, target = (
        prepare_awareness_dataframe(
            dataframe
        )
    )

    split = stratified_awareness_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    processed, preprocessor = (
        fit_transform_awareness_split(
            split
        )
    )

    assert len(processed.x_train) == 80
    assert len(processed.x_test) == 20

    assert (
        processed.x_train.shape[1]
        == processed.x_test.shape[1]
    )

    assert len(
        processed.feature_names
    ) == processed.x_train.shape[1]

    assert (
        processed.metadata[
            "preprocessing_fit_scope"
        ]
        == "training_partition_only"
    )

    assert preprocessor is not None


def test_preprocessor_handles_unknown_test_category() -> None:
    training = pd.DataFrame(
        {
            "hover_time_ms": [
                100,
                200,
                300,
                400,
            ],
            "device_type": [
                "desktop",
                "mobile",
                "desktop",
                "mobile",
            ],
        }
    )

    testing = pd.DataFrame(
        {
            "hover_time_ms": [500],
            "device_type": ["tablet"],
        }
    )

    preprocessor = build_awareness_preprocessor(
        numeric_columns=[
            "hover_time_ms",
        ],
        categorical_columns=[
            "device_type",
        ],
    )

    train_result = preprocessor.fit_transform(
        training
    )

    test_result = preprocessor.transform(
        testing
    )

    assert train_result.shape[1] == (
        test_result.shape[1]
    )


def test_complete_awareness_preprocessing() -> None:
    dataframe = create_awareness_dataframe(
        rows=100
    )

    processed, preprocessor = (
        preprocess_awareness_dataset(
            dataframe,
            test_size=0.20,
            random_state=42,
        )
    )

    assert processed.x_train.shape[0] == 80
    assert processed.x_test.shape[0] == 20

    assert not np.isnan(
        processed.x_train.to_numpy(
            dtype=float
        )
    ).any()

    assert not np.isnan(
        processed.x_test.to_numpy(
            dtype=float
        )
    ).any()

    assert (
        processed.metadata[
            "reported_email_included"
        ]
        is False
    )

    assert preprocessor is not None


def test_awareness_schema_to_dict() -> None:
    schema_dict = awareness_schema_to_dict(
        AwarenessSchema()
    )

    assert (
        schema_dict["target_column"]
        == "clicked_link"
    )

    assert "user_id" in (
        schema_dict[
            "identifier_columns"
        ]
    )