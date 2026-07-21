from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


class FeatureEngineeringError(RuntimeError):
    """Raised when feature engineering cannot be completed safely."""


URL_PATTERN = re.compile(
    r"""
    (?:
        https?://[^\s<>"']+
        |
        www\.[^\s<>"']+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EMAIL_ADDRESS_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

WORD_PATTERN = re.compile(
    r"\b[\w'-]+\b",
    flags=re.UNICODE,
)

SENTENCE_PATTERN = re.compile(
    r"[.!?]+"
)

PERSONALISATION_PATTERN = re.compile(
    r"\b(?:dear|hello|hi)\s+[A-Z][a-z]+\b"
)

GREETING_PATTERN = re.compile(
    r"^\s*(?:dear|hello|hi|greetings|good morning|good afternoon)\b",
    flags=re.IGNORECASE,
)

SIGNOFF_PATTERN = re.compile(
    r"\b(?:sincerely|regards|kind regards|best regards|thank you|thanks)\b",
    flags=re.IGNORECASE,
)

CURRENCY_SYMBOL_PATTERN = re.compile(
    r"[$€£¥₹]"
)


URGENCY_KEYWORDS = {
    "urgent",
    "urgently",
    "immediately",
    "immediate",
    "now",
    "today",
    "quickly",
    "promptly",
    "deadline",
    "expire",
    "expires",
    "expired",
    "limited time",
    "within 24 hours",
    "within 48 hours",
    "act now",
    "action required",
}

AUTHORITY_KEYWORDS = {
    "bank",
    "government",
    "police",
    "security team",
    "administrator",
    "admin",
    "manager",
    "director",
    "officer",
    "irs",
    "revenue",
    "support team",
    "customer service",
    "financial institution",
    "official",
    "department",
}

FEAR_KEYWORDS = {
    "suspended",
    "suspension",
    "blocked",
    "locked",
    "unauthorized",
    "fraud",
    "fraudulent",
    "warning",
    "risk",
    "danger",
    "breach",
    "compromised",
    "penalty",
    "terminate",
    "termination",
    "legal action",
    "lose access",
}

FINANCIAL_KEYWORDS = {
    "payment",
    "invoice",
    "refund",
    "tax",
    "bank",
    "account",
    "money",
    "funds",
    "transaction",
    "credit card",
    "debit card",
    "billing",
    "salary",
    "investment",
    "transfer",
    "fee",
    "balance",
}

CREDENTIAL_KEYWORDS = {
    "password",
    "username",
    "login",
    "log in",
    "sign in",
    "credentials",
    "verify",
    "verification",
    "verify account",
    "confirm account",
    "account details",
    "security code",
    "one-time password",
    "otp",
    "pin",
    "verification code",
}

REWARD_KEYWORDS = {
    "prize",
    "winner",
    "won",
    "reward",
    "gift",
    "bonus",
    "offer",
    "opportunity",
    "promotion",
    "cash",
    "lottery",
    "free",
}

TRUST_KEYWORDS = {
    "secure",
    "security",
    "official",
    "verified",
    "trusted",
    "protected",
    "safe",
    "confidential",
    "authorised",
    "authorized",
    "legitimate",
    "guaranteed",
    "privacy",
}

CALL_TO_ACTION_KEYWORDS = {
    "click",
    "click here",
    "open",
    "download",
    "verify",
    "confirm",
    "update",
    "reply",
    "respond",
    "visit",
    "follow the link",
    "sign in",
    "log in",
    "submit",
    "provide",
}


def normalise_text(value: Any) -> str:
    """
    Convert an arbitrary value into safe text.

    Missing values are converted to an empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value)


def tokenise_words(text: str) -> list[str]:
    """Return lowercase word-like tokens."""

    return [
        match.group(0).lower()
        for match in WORD_PATTERN.finditer(text)
    ]


def count_keyword_occurrences(
    text: str,
    keywords: Iterable[str],
) -> int:
    """
    Count keyword and phrase occurrences case-insensitively.

    Multi-word phrases are counted using escaped literal matching.
    """

    lowered_text = text.lower()
    total = 0

    for keyword in keywords:
        pattern = re.compile(
            rf"\b{re.escape(keyword.lower())}\b"
        )
        total += len(pattern.findall(lowered_text))

    return total


def calculate_character_entropy(text: str) -> float:
    """
    Calculate Shannon entropy for characters in text.

    Returns zero for empty text.
    """

    if not text:
        return 0.0

    counts = Counter(text)
    text_length = len(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / text_length
        entropy -= probability * math.log2(probability)

    return float(entropy)


def calculate_uppercase_ratio(text: str) -> float:
    """Calculate uppercase alphabetic characters divided by all letters."""

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    if not alphabetic_characters:
        return 0.0

    uppercase_count = sum(
        character.isupper()
        for character in alphabetic_characters
    )

    return uppercase_count / len(alphabetic_characters)


def calculate_lexical_diversity(words: list[str]) -> float:
    """Calculate unique words divided by total words."""

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def extract_email_text_features(
    text: Any,
) -> dict[str, int | float]:
    """
    Extract interpretable structural and persuasion features from email text.

    This function does not classify the email and does not calculate the HWI.
    """

    clean_text = normalise_text(text)
    words = tokenise_words(clean_text)

    word_count = len(words)
    sentence_matches = SENTENCE_PATTERN.findall(clean_text)
    sentence_count = len(sentence_matches)

    if clean_text.strip() and sentence_count == 0:
        sentence_count = 1

    alphabetic_words = [
        word
        for word in words
        if any(character.isalpha() for character in word)
    ]

    total_word_characters = sum(
        len(word)
        for word in alphabetic_words
    )

    average_word_length = (
        total_word_characters / len(alphabetic_words)
        if alphabetic_words
        else 0.0
    )

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count
        else 0.0
    )

    uppercase_character_count = sum(
        character.isupper()
        for character in clean_text
    )

    digit_count = sum(
        character.isdigit()
        for character in clean_text
    )

    url_matches = URL_PATTERN.findall(clean_text)
    email_matches = EMAIL_ADDRESS_PATTERN.findall(clean_text)

    return {
        "character_count": len(clean_text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_word_length": round(
            average_word_length,
            6,
        ),
        "average_sentence_length": round(
            average_sentence_length,
            6,
        ),
        "unique_word_count": len(set(words)),
        "lexical_diversity": round(
            calculate_lexical_diversity(words),
            6,
        ),
        "uppercase_character_count": uppercase_character_count,
        "uppercase_ratio": round(
            calculate_uppercase_ratio(clean_text),
            6,
        ),
        "digit_count": digit_count,
        "exclamation_count": clean_text.count("!"),
        "question_count": clean_text.count("?"),
        "comma_count": clean_text.count(","),
        "newline_count": (
            clean_text.count("\n")
            + clean_text.count("\r")
        ),
        "url_count": len(url_matches),
        "email_address_count": len(email_matches),
        "currency_symbol_count": len(
            CURRENCY_SYMBOL_PATTERN.findall(clean_text)
        ),
        "urgency_keyword_count": count_keyword_occurrences(
            clean_text,
            URGENCY_KEYWORDS,
        ),
        "authority_keyword_count": count_keyword_occurrences(
            clean_text,
            AUTHORITY_KEYWORDS,
        ),
        "fear_keyword_count": count_keyword_occurrences(
            clean_text,
            FEAR_KEYWORDS,
        ),
        "financial_keyword_count": count_keyword_occurrences(
            clean_text,
            FINANCIAL_KEYWORDS,
        ),
        "credential_keyword_count": count_keyword_occurrences(
            clean_text,
            CREDENTIAL_KEYWORDS,
        ),
        "reward_keyword_count": count_keyword_occurrences(
            clean_text,
            REWARD_KEYWORDS,
        ),
        "trust_keyword_count": count_keyword_occurrences(
            clean_text,
            TRUST_KEYWORDS,
        ),
        "call_to_action_count": count_keyword_occurrences(
            clean_text,
            CALL_TO_ACTION_KEYWORDS,
        ),
        "greeting_present": int(
            bool(GREETING_PATTERN.search(clean_text))
        ),
        "signoff_present": int(
            bool(SIGNOFF_PATTERN.search(clean_text))
        ),
        "personalisation_indicator": int(
            bool(PERSONALISATION_PATTERN.search(clean_text))
        ),
        "suspicious_link_indicator": int(
            len(url_matches) > 0
        ),
        "character_entropy": round(
            calculate_character_entropy(clean_text),
            6,
        ),
    }


def combine_subject_and_body(
    subject: Any,
    body: Any,
) -> str:
    """
    Combine subject and body while preserving available content.

    Empty components are ignored.
    """

    subject_text = normalise_text(subject).strip()
    body_text = normalise_text(body).strip()

    components = [
        component
        for component in (subject_text, body_text)
        if component
    ]

    return "\n".join(components)


def build_common_email_text(
    dataframe: pd.DataFrame,
    *,
    text_column: str | None = None,
    subject_column: str | None = None,
    body_column: str | None = None,
) -> pd.Series:
    """
    Build one common text field from different email schemas.

    Supported approaches:

    - use one existing text column;
    - combine subject and body columns.
    """

    if text_column is not None:
        if text_column not in dataframe.columns:
            raise FeatureEngineeringError(
                f"Text column not found: {text_column}"
            )

        return dataframe[text_column].map(
            normalise_text
        )

    if subject_column is None and body_column is None:
        raise FeatureEngineeringError(
            "Provide text_column or at least one of "
            "subject_column/body_column."
        )

    missing_columns = [
        column
        for column in (subject_column, body_column)
        if column is not None
        and column not in dataframe.columns
    ]

    if missing_columns:
        raise FeatureEngineeringError(
            "Required email columns are missing: "
            + ", ".join(missing_columns)
        )

    if subject_column is None:
        return dataframe[body_column].map(  # type: ignore[index]
            normalise_text
        )

    if body_column is None:
        return dataframe[subject_column].map(
            normalise_text
        )

    return pd.Series(
        [
            combine_subject_and_body(subject, body)
            for subject, body in zip(
                dataframe[subject_column],
                dataframe[body_column],
                strict=True,
            )
        ],
        index=dataframe.index,
        name="combined_text",
    )


def extract_email_feature_frame(
    texts: pd.Series,
) -> pd.DataFrame:
    """
    Extract email features for every text record.

    The output preserves the original Series index.
    """

    if not isinstance(texts, pd.Series):
        raise TypeError(
            "texts must be a pandas Series."
        )

    feature_records = [
        extract_email_text_features(value)
        for value in texts
    ]

    return pd.DataFrame(
        feature_records,
        index=texts.index,
    )


def engineer_email_features(
    dataframe: pd.DataFrame,
    *,
    text_column: str | None = None,
    subject_column: str | None = None,
    body_column: str | None = None,
    retain_original_text: bool = False,
) -> pd.DataFrame:
    """
    Create a model-ready handcrafted email feature table.

    The returned table contains only engineered features unless
    retain_original_text=True.
    """

    if dataframe.empty:
        raise FeatureEngineeringError(
            "Cannot engineer features from an empty DataFrame."
        )

    common_text = build_common_email_text(
        dataframe=dataframe,
        text_column=text_column,
        subject_column=subject_column,
        body_column=body_column,
    )

    feature_frame = extract_email_feature_frame(
        common_text
    )

    if retain_original_text:
        feature_frame.insert(
            0,
            "combined_text",
            common_text,
        )

    return feature_frame.reset_index(drop=True)


def add_experimental_labels(
    feature_frame: pd.DataFrame,
    *,
    source_type: str,
    email_class: str,
    original_label: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Add trustworthy experimental labels.

    source_type must be ``human`` or ``llm``.
    email_class must be ``legitimate`` or ``phishing``.
    """

    valid_source_types = {
        "human",
        "llm",
    }
    valid_email_classes = {
        "legitimate",
        "phishing",
    }

    if source_type not in valid_source_types:
        raise ValueError(
            f"Invalid source_type: {source_type}"
        )

    if email_class not in valid_email_classes:
        raise ValueError(
            f"Invalid email_class: {email_class}"
        )

    result = feature_frame.copy()

    result["source_type"] = source_type
    result["email_class"] = email_class
    result["source_label"] = int(
        source_type == "llm"
    )
    result["phishing_label"] = int(
        email_class == "phishing"
    )

    if original_label is not None:
        if len(original_label) != len(result):
            raise ValueError(
                "original_label length must match feature-frame length."
            )

        result["original_label"] = (
            original_label
            .reset_index(drop=True)
        )

    return result


def build_ai_email_feature_dataset(
    human_legitimate: pd.DataFrame,
    human_phishing: pd.DataFrame,
    llm_legitimate: pd.DataFrame,
    llm_phishing: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one harmonised feature table for the four AI-email experiment groups.

    Human email text is created from subject and body.
    LLM email text is read from the text column.

    The four source datasets remain identifiable through explicit labels.
    """

    human_legitimate_features = engineer_email_features(
        human_legitimate,
        subject_column="subject",
        body_column="body",
    )

    human_legitimate_features = add_experimental_labels(
        human_legitimate_features,
        source_type="human",
        email_class="legitimate",
        original_label=(
            human_legitimate["label"]
            if "label" in human_legitimate.columns
            else None
        ),
    )

    human_phishing_features = engineer_email_features(
        human_phishing,
        subject_column="subject",
        body_column="body",
    )

    human_phishing_features = add_experimental_labels(
        human_phishing_features,
        source_type="human",
        email_class="phishing",
        original_label=(
            human_phishing["label"]
            if "label" in human_phishing.columns
            else None
        ),
    )

    llm_legitimate_features = engineer_email_features(
        llm_legitimate,
        text_column="text",
    )

    llm_legitimate_features = add_experimental_labels(
        llm_legitimate_features,
        source_type="llm",
        email_class="legitimate",
        original_label=(
            llm_legitimate["label"]
            if "label" in llm_legitimate.columns
            else None
        ),
    )

    llm_phishing_features = engineer_email_features(
        llm_phishing,
        text_column="text",
    )

    llm_phishing_features = add_experimental_labels(
        llm_phishing_features,
        source_type="llm",
        email_class="phishing",
        original_label=(
            llm_phishing["label"]
            if "label" in llm_phishing.columns
            else None
        ),
    )

    combined = pd.concat(
        [
            human_legitimate_features,
            human_phishing_features,
            llm_legitimate_features,
            llm_phishing_features,
        ],
        ignore_index=True,
    )

    combined["experiment_group"] = (
        combined["source_type"]
        + "_"
        + combined["email_class"]
    )

    return combined


def validate_feature_frame(
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate an engineered email feature table.

    Returns a summary rather than changing the data.
    """

    numeric_columns = feature_frame.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    infinite_value_count = 0

    if numeric_columns:
        numeric_values = feature_frame[
            numeric_columns
        ].to_numpy(dtype=float, copy=True)

        infinite_value_count = int(
            np.isinf(numeric_values).sum()
        )

    return {
        "rows": int(len(feature_frame)),
        "columns": int(feature_frame.shape[1]),
        "duplicate_rows": int(
            feature_frame.duplicated().sum()
        ),
        "total_missing_values": int(
            feature_frame.isna().sum().sum()
        ),
        "infinite_values": infinite_value_count,
        "numeric_columns": numeric_columns,
        "column_names": feature_frame.columns.tolist(),
    }