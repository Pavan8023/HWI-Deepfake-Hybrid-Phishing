from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


class URLRiskModelError(RuntimeError):
    """Raised when URL-risk modelling cannot proceed safely."""


MALICIOUS_TYPES = {
    "phishing",
    "malware",
    "defacement",
}

BENIGN_TYPE = "benign"

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "tiny.cc",
    "rebrand.ly",
}

SUSPICIOUS_URL_KEYWORDS = {
    "account",
    "admin",
    "bank",
    "confirm",
    "credential",
    "download",
    "invoice",
    "login",
    "password",
    "payment",
    "recover",
    "secure",
    "security",
    "signin",
    "support",
    "update",
    "verification",
    "verify",
    "wallet",
    "webscr",
}


@dataclass(frozen=True)
class URLRiskSplit:
    """Train/test data for the URL technical-risk model."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame
    feature_names: tuple[str, ...]


def validate_url_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the raw malicious-URL dataset."""

    if dataframe.empty:
        raise URLRiskModelError(
            "The URL dataset is empty."
        )

    required_columns = {
        "url",
        "type",
    }

    missing_columns = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise URLRiskModelError(
            "The URL dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataframe["url"].isna().all():
        raise URLRiskModelError(
            "The URL column contains no usable values."
        )

    observed_types = {
        str(value).strip().lower()
        for value in dataframe["type"].dropna()
    }

    allowed_types = {
        BENIGN_TYPE,
        *MALICIOUS_TYPES,
    }

    unsupported_types = sorted(
        observed_types.difference(
            allowed_types
        )
    )

    if unsupported_types:
        raise URLRiskModelError(
            "Unsupported URL classes: "
            + ", ".join(unsupported_types)
        )


def encode_url_target(
    series: pd.Series,
) -> pd.Series:
    """Encode benign as 0 and all malicious URL types as 1."""

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_mask = ~normalized.isin(
        {
            BENIGN_TYPE,
            *MALICIOUS_TYPES,
        }
    )

    if invalid_mask.any():
        invalid_values = (
            normalized.loc[invalid_mask]
            .dropna()
            .unique()
            .tolist()
        )

        raise URLRiskModelError(
            "URL target contains unsupported values: "
            + ", ".join(
                str(value)
                for value in invalid_values
            )
        )

    return (
        normalized.isin(MALICIOUS_TYPES)
        .astype("int64")
        .rename("malicious_label")
    )


def character_entropy(
    value: str,
) -> float:
    """Calculate Shannon entropy for a URL string."""

    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)

    return float(
        -sum(
            (count / length)
            * math.log2(count / length)
            for count in counts.values()
        )
    )


def normalize_url_for_parsing(
    raw_url: object,
) -> str:
    """Return a clean URL string suitable for lexical parsing."""

    if pd.isna(raw_url):
        return ""

    value = str(raw_url).strip()

    if not value:
        return ""

    if "://" not in value:
        return f"http://{value}"

    return value


def hostname_is_ip_address(
    hostname: str,
) -> int:
    """Return 1 when the hostname is an IPv4 or IPv6 address."""

    if not hostname:
        return 0

    try:
        ipaddress.ip_address(
            hostname.strip("[]")
        )
        return 1
    except ValueError:
        return 0


def count_suspicious_keywords(
    value: str,
) -> int:
    """Count suspicious URL tokens using boundary-aware matching."""

    lowered = value.lower()

    return int(
        sum(
            len(
                re.findall(
                    rf"(?<![a-z0-9])"
                    rf"{re.escape(keyword)}"
                    rf"(?![a-z0-9])",
                    lowered,
                )
            )
            for keyword in SUSPICIOUS_URL_KEYWORDS
        )
    )


def extract_url_features(
    raw_url: object,
) -> dict[str, float | int]:
    """Extract interpretable lexical and structural URL features."""

    original = (
        ""
        if pd.isna(raw_url)
        else str(raw_url).strip()
    )

    parsing_value = normalize_url_for_parsing(
        original
    )

    try:
        parsed = urlsplit(
            parsing_value
        )
    except ValueError:
        parsed = urlsplit(
            "http://invalid"
        )

    hostname = (
        parsed.hostname or ""
    ).lower()

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    dot_count = original.count(".")
    slash_count = original.count("/")
    hyphen_count = original.count("-")
    underscore_count = original.count("_")
    at_count = original.count("@")
    ampersand_count = original.count("&")
    equals_count = original.count("=")
    question_count = original.count("?")
    percent_count = original.count("%")

    digit_count = sum(
        character.isdigit()
        for character in original
    )

    alphabetic_count = sum(
        character.isalpha()
        for character in original
    )

    special_character_count = sum(
        not character.isalnum()
        for character in original
    )

    hostname_parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        0,
        len(hostname_parts) - 2,
    )

    query_parameter_count = len(
        parse_qsl(
            query,
            keep_blank_values=True,
        )
    )

    scheme = parsed.scheme.lower()

    return {
        "url_length": len(original),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "digit_count": digit_count,
        "alphabetic_count": alphabetic_count,
        "special_character_count": (
            special_character_count
        ),
        "dot_count": dot_count,
        "slash_count": slash_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "at_count": at_count,
        "ampersand_count": ampersand_count,
        "equals_count": equals_count,
        "question_count": question_count,
        "percent_encoding_count": (
            percent_count
        ),
        "subdomain_count": subdomain_count,
        "query_parameter_count": (
            query_parameter_count
        ),
        "https_indicator": int(
            scheme == "https"
        ),
        "http_indicator": int(
            scheme == "http"
        ),
        "ip_address_indicator": (
            hostname_is_ip_address(
                hostname
            )
        ),
        "shortener_indicator": int(
            hostname in SHORTENER_DOMAINS
        ),
        "suspicious_keyword_count": (
            count_suspicious_keywords(
                original
            )
        ),
        "double_slash_path_indicator": int(
            "//" in path
        ),
        "punycode_indicator": int(
            "xn--" in hostname
        ),
        "port_present_indicator": int(
            parsed.port is not None
        )
        if hostname
        else 0,
        "file_extension_indicator": int(
            bool(
                re.search(
                    r"\.[a-zA-Z0-9]{1,6}$",
                    path,
                )
            )
        ),
        "digit_ratio": (
            digit_count / len(original)
            if original
            else 0.0
        ),
        "special_character_ratio": (
            special_character_count
            / len(original)
            if original
            else 0.0
        ),
        "url_entropy": character_entropy(
            original
        ),
    }


def engineer_url_feature_frame(
    urls: pd.Series,
) -> pd.DataFrame:
    """Extract URL features for an entire Series."""

    if urls.empty:
        raise URLRiskModelError(
            "Cannot engineer features from an empty URL Series."
        )

    records = [
        extract_url_features(value)
        for value in urls
    ]

    frame = pd.DataFrame(
        records,
        index=urls.index,
    )

    return (
        frame.apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def prepare_url_dataset(
    dataframe: pd.DataFrame,
    *,
    maximum_rows: int | None = 200_000,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Deduplicate, optionally sample and engineer the URL dataset.

    Sampling is stratified by the binary malicious target.
    """

    validate_url_dataset(
        dataframe
    )

    working = dataframe[
        [
            "url",
            "type",
        ]
    ].copy()

    working["url"] = (
        working["url"]
        .astype("string")
        .str.strip()
    )

    working["type"] = (
        working["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    working = working.loc[
        working["url"].notna()
        & working["url"].ne("")
    ].copy()

    rows_before_deduplication = len(
        working
    )

    working = (
        working.drop_duplicates(
            subset=["url"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    rows_after_deduplication = len(
        working
    )

    working["malicious_label"] = (
        encode_url_target(
            working["type"]
        )
    )

    sampled = False

    if (
        maximum_rows is not None
        and len(working) > maximum_rows
    ):
        sampled_frame, _ = train_test_split(
            working,
            train_size=maximum_rows,
            random_state=random_state,
            stratify=working[
                "malicious_label"
            ],
        )

        working = sampled_frame.reset_index(
            drop=True
        )

        sampled = True

    features = engineer_url_feature_frame(
        working["url"]
    ).reset_index(drop=True)

    if features.isna().any().any():
        invalid_columns = (
            features.columns[
                features.isna().any()
            ]
            .tolist()
        )

        raise URLRiskModelError(
            "Engineered URL features contain invalid values: "
            + ", ".join(invalid_columns)
        )

    target = (
        working["malicious_label"]
        .astype("int64")
        .reset_index(drop=True)
    )

    metadata = (
        working[
            [
                "url",
                "type",
            ]
        ]
        .reset_index(drop=True)
    )

    summary = {
        "raw_rows": int(
            len(dataframe)
        ),
        "rows_before_deduplication": int(
            rows_before_deduplication
        ),
        "rows_after_deduplication": int(
            rows_after_deduplication
        ),
        "duplicates_removed": int(
            rows_before_deduplication
            - rows_after_deduplication
        ),
        "sampled": sampled,
        "model_rows": int(
            len(working)
        ),
        "feature_count": int(
            features.shape[1]
        ),
        "binary_target_distribution": {
            str(key): int(value)
            for key, value in (
                target.value_counts()
                .sort_index()
                .items()
            )
        },
        "original_type_distribution": {
            str(key): int(value)
            for key, value in (
                metadata["type"]
                .value_counts()
                .items()
            )
        },
    }

    return (
        features,
        target,
        metadata,
        summary,
    )


def stratified_url_split(
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> URLRiskSplit:
    """Create a reproducible stratified URL train/test split."""

    if not (
        len(features)
        == len(target)
        == len(metadata)
    ):
        raise URLRiskModelError(
            "Feature, target and metadata row counts do not match."
        )

    indices = np.arange(
        len(features)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
    )

    return URLRiskSplit(
        x_train=(
            features.iloc[train_indices]
            .reset_index(drop=True)
        ),
        x_test=(
            features.iloc[test_indices]
            .reset_index(drop=True)
        ),
        y_train=(
            target.iloc[train_indices]
            .reset_index(drop=True)
        ),
        y_test=(
            target.iloc[test_indices]
            .reset_index(drop=True)
        ),
        metadata_train=(
            metadata.iloc[train_indices]
            .reset_index(drop=True)
        ),
        metadata_test=(
            metadata.iloc[test_indices]
            .reset_index(drop=True)
        ),
        feature_names=tuple(
            features.columns
        ),
    )


def validate_url_split(
    split: URLRiskSplit,
) -> dict[str, Any]:
    """Validate the URL model split and report its structure."""

    if list(
        split.x_train.columns
    ) != list(
        split.x_test.columns
    ):
        raise URLRiskModelError(
            "Train and test URL features do not match."
        )

    if (
        split.x_train.isna().any().any()
        or split.x_test.isna().any().any()
    ):
        raise URLRiskModelError(
            "URL train/test features contain missing values."
        )

    return {
        "train_rows": int(
            len(split.x_train)
        ),
        "test_rows": int(
            len(split.x_test)
        ),
        "feature_count": int(
            split.x_train.shape[1]
        ),
        "train_target_distribution": {
            str(key): int(value)
            for key, value in (
                split.y_train
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "test_target_distribution": {
            str(key): int(value)
            for key, value in (
                split.y_test
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "feature_names": list(
            split.feature_names
        ),
    }