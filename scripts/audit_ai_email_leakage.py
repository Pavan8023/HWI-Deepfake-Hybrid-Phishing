from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    combine_subject_and_body,
)


RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai_emails"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "ai_email_model"
)


def normalise_for_duplicate_check(
    text: object,
) -> str:
    """Normalise message text for exact/template duplicate checks."""

    value = str(text).lower()

    value = re.sub(
        r"https?://\S+|www\.\S+",
        "<url>",
        value,
    )

    value = re.sub(
        r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b",
        "<email>",
        value,
    )

    value = re.sub(
        r"\b\d+\b",
        "<number>",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def hash_text(text: str) -> str:
    """Create a deterministic SHA-256 text hash."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def require_dataframe(
    loaded: object,
    name: str,
) -> pd.DataFrame:
    """Ensure a loaded object is a DataFrame."""

    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(
            f"{name} did not load as a DataFrame."
        )

    return loaded


def load_group(
    *,
    path: Path,
    source_type: str,
    email_class: str,
) -> pd.DataFrame:
    """Load and standardise one AI-email experimental group."""

    dataframe = require_dataframe(
        load_dataset(path),
        path.name,
    )

    if "text" in dataframe.columns:
        combined_text = (
            dataframe["text"]
            .fillna("")
            .astype(str)
        )

    elif {
        "subject",
        "body",
    }.issubset(dataframe.columns):
        combined_text = pd.Series(
            [
                combine_subject_and_body(
                    subject,
                    body,
                )
                for subject, body in zip(
                    dataframe["subject"],
                    dataframe["body"],
                    strict=True,
                )
            ]
        )

    else:
        raise KeyError(
            f"No usable email text columns in {path}"
        )

    result = pd.DataFrame(
        {
            "combined_text": combined_text,
            "source_type": source_type,
            "email_class": email_class,
        }
    )

    result["experiment_group"] = (
        result["source_type"]
        + "_"
        + result["email_class"]
    )

    return result


def main() -> int:
    """Audit exact duplicates and normalised templates."""

    groups = [
        load_group(
            path=(
                RAW_DIRECTORY
                / "human-generated"
                / "legit.csv"
            ),
            source_type="human",
            email_class="legitimate",
        ),
        load_group(
            path=(
                RAW_DIRECTORY
                / "human-generated"
                / "phishing.csv"
            ),
            source_type="human",
            email_class="phishing",
        ),
        load_group(
            path=(
                RAW_DIRECTORY
                / "llm-generated"
                / "legit.csv"
            ),
            source_type="llm",
            email_class="legitimate",
        ),
        load_group(
            path=(
                RAW_DIRECTORY
                / "llm-generated"
                / "phishing.csv"
            ),
            source_type="llm",
            email_class="phishing",
        ),
    ]

    dataframe = pd.concat(
        groups,
        ignore_index=True,
    )

    dataframe["exact_hash"] = (
        dataframe["combined_text"]
        .map(hash_text)
    )

    dataframe["normalised_text"] = (
        dataframe["combined_text"]
        .map(normalise_for_duplicate_check)
    )

    dataframe["template_hash"] = (
        dataframe["normalised_text"]
        .map(hash_text)
    )

    exact_duplicate_rows = int(
        dataframe.duplicated(
            subset=["exact_hash"],
            keep=False,
        ).sum()
    )

    template_duplicate_rows = int(
        dataframe.duplicated(
            subset=["template_hash"],
            keep=False,
        ).sum()
    )

    conflicting_exact = (
        dataframe.groupby("exact_hash")[
            "email_class"
        ]
        .nunique()
    )

    conflicting_template = (
        dataframe.groupby("template_hash")[
            "email_class"
        ]
        .nunique()
    )

    exact_conflict_count = int(
        (conflicting_exact > 1).sum()
    )

    template_conflict_count = int(
        (conflicting_template > 1).sum()
    )

    group_summary = (
        dataframe.groupby(
            "experiment_group"
        )
        .agg(
            rows=("combined_text", "size"),
            exact_unique=("exact_hash", "nunique"),
            template_unique=(
                "template_hash",
                "nunique",
            ),
        )
        .reset_index()
    )

    group_summary[
        "exact_duplicate_rows"
    ] = (
        group_summary["rows"]
        - group_summary["exact_unique"]
    )

    group_summary[
        "template_duplicate_rows"
    ] = (
        group_summary["rows"]
        - group_summary["template_unique"]
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        OUTPUT_DIRECTORY
        / "ai_email_duplicate_audit.csv",
        index=False,
        encoding="utf-8",
    )

    duplicate_templates = (
        dataframe.loc[
            dataframe.duplicated(
                subset=["template_hash"],
                keep=False,
            ),
            [
                "experiment_group",
                "email_class",
                "template_hash",
                "normalised_text",
            ],
        ]
        .sort_values(
            [
                "template_hash",
                "experiment_group",
            ]
        )
    )

    duplicate_templates.to_csv(
        OUTPUT_DIRECTORY
        / "ai_email_duplicate_templates.csv",
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("AI-EMAIL DUPLICATE AND TEMPLATE AUDIT")
    print("=" * 80)
    print(f"Total rows: {len(dataframe):,}")
    print(
        f"Exact duplicate rows: "
        f"{exact_duplicate_rows:,}"
    )
    print(
        f"Template duplicate rows: "
        f"{template_duplicate_rows:,}"
    )
    print(
        f"Exact hashes with conflicting labels: "
        f"{exact_conflict_count:,}"
    )
    print(
        f"Template hashes with conflicting labels: "
        f"{template_conflict_count:,}"
    )
    print()
    print(group_summary.to_string(index=False))
    print()
    print(
        "Summary saved to:",
        OUTPUT_DIRECTORY
        / "ai_email_duplicate_audit.csv",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())