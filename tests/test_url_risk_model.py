from __future__ import annotations

import pandas as pd
import pytest

from src.url_risk_model import (
    URLRiskModelError,
    character_entropy,
    encode_url_target,
    engineer_url_feature_frame,
    extract_url_features,
    prepare_url_dataset,
    stratified_url_split,
    validate_url_dataset,
    validate_url_split,
)


def create_url_dataset(
    rows: int = 100,
) -> pd.DataFrame:
    types = [
        "benign",
        "phishing",
        "malware",
        "defacement",
    ]

    return pd.DataFrame(
        {
            "url": [
                (
                    f"https://example{index}.com/page"
                    if index % 4 == 0
                    else (
                        f"http://secure-login-{index}.com/"
                        f"verify?account={index}"
                    )
                )
                for index in range(rows)
            ],
            "type": [
                types[index % 4]
                for index in range(rows)
            ],
        }
    )


def test_validate_url_dataset() -> None:
    validate_url_dataset(
        create_url_dataset()
    )


def test_validate_url_dataset_rejects_missing_column() -> None:
    dataframe = (
        create_url_dataset()
        .drop(columns=["type"])
    )

    with pytest.raises(URLRiskModelError):
        validate_url_dataset(
            dataframe
        )


def test_encode_url_target() -> None:
    encoded = encode_url_target(
        pd.Series(
            [
                "benign",
                "phishing",
                "malware",
                "defacement",
            ]
        )
    )

    assert encoded.tolist() == [
        0,
        1,
        1,
        1,
    ]


def test_character_entropy() -> None:
    assert character_entropy("") == 0.0
    assert character_entropy("aaaa") == 0.0
    assert character_entropy("abcd") > 0


def test_extract_url_features() -> None:
    features = extract_url_features(
        "https://secure-login123.example.com/"
        "verify?account=5"
    )

    assert features["https_indicator"] == 1
    assert features["digit_count"] >= 4
    assert features["hyphen_count"] >= 1
    assert features["subdomain_count"] >= 1
    assert features[
        "suspicious_keyword_count"
    ] >= 2


def test_ip_address_indicator() -> None:
    features = extract_url_features(
        "http://192.168.1.10/login"
    )

    assert features[
        "ip_address_indicator"
    ] == 1


def test_engineer_url_feature_frame() -> None:
    result = engineer_url_feature_frame(
        pd.Series(
            [
                "https://example.com",
                "http://secure-login.test/verify",
            ]
        )
    )

    assert len(result) == 2
    assert "url_length" in result.columns
    assert not result.isna().any().any()


def test_prepare_url_dataset_removes_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "url": [
                "https://example.com",
                "https://example.com",
                "http://bad.test/login",
                "http://malware.test/file",
            ],
            "type": [
                "benign",
                "benign",
                "phishing",
                "malware",
            ],
        }
    )

    features, target, metadata, summary = (
        prepare_url_dataset(
            dataframe,
            maximum_rows=None,
        )
    )

    assert len(features) == 3
    assert len(target) == 3
    assert len(metadata) == 3
    assert summary[
        "duplicates_removed"
    ] == 1


def test_stratified_url_split() -> None:
    dataframe = create_url_dataset(
        rows=200
    )

    features, target, metadata, _ = (
        prepare_url_dataset(
            dataframe,
            maximum_rows=None,
        )
    )

    split = stratified_url_split(
        features,
        target,
        metadata,
        test_size=0.20,
        random_state=42,
    )

    assert len(split.x_train) == 160
    assert len(split.x_test) == 40
    assert (
        split.x_train.shape[1]
        == split.x_test.shape[1]
    )


def test_validate_url_split() -> None:
    dataframe = create_url_dataset(
        rows=200
    )

    features, target, metadata, _ = (
        prepare_url_dataset(
            dataframe,
            maximum_rows=None,
        )
    )

    split = stratified_url_split(
        features,
        target,
        metadata,
    )

    summary = validate_url_split(
        split
    )

    assert summary["feature_count"] > 10
    assert summary["train_rows"] == 160
    assert summary["test_rows"] == 40