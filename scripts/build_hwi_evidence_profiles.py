from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evidence_profiles import (  # noqa: E402
    build_evidence_profiles,
    profile_summary_markdown,
    profiles_to_dataframe,
    quantile_to_profile_name,
    select_nearest_quantile_rows,
)


BEHAVIOURAL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "awareness_test_hwi.csv"
)

EMAIL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "ai_email_all_persuasion_scores.csv"
)

URL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "url_test_risk_scores.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "hwi_framework"
)

PROFILE_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "hwi_evidence_profiles.csv"
)

PROFILE_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "hwi_evidence_profiles.json"
)

SELECTION_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "hwi_profile_selection_audit.csv"
)

MARKDOWN_PATH = (
    OUTPUT_DIRECTORY
    / "hwi_evidence_profiles.md"
)


def detect_behavioural_score_column(
    dataframe: pd.DataFrame,
) -> str:
    """Detect the existing behavioural HWI score column."""

    candidates = (
        "hwi_score",
        "human_weakness_index",
        "model_hwi_score",
    )

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        "No behavioural HWI score column was found. "
        f"Available columns: {dataframe.columns.tolist()}"
    )


def require_file(
    path: Path,
) -> None:
    """Raise a clear error when an input is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


def main() -> int:
    """Build real, independently selected HWI evidence profiles."""

    for path in (
        BEHAVIOURAL_PATH,
        EMAIL_PATH,
        URL_PATH,
    ):
        require_file(path)

    behavioural = pd.read_csv(
        BEHAVIOURAL_PATH
    )

    email = pd.read_csv(
        EMAIL_PATH
    )

    url = pd.read_csv(
        URL_PATH
    )

    behavioural_score_column = (
        detect_behavioural_score_column(
            behavioural
        )
    )

    behavioural_selected = (
        select_nearest_quantile_rows(
            behavioural,
            score_column=(
                behavioural_score_column
            ),
            dataset_name=(
                "awareness_test_hwi"
            ),
        )
    )

    email_selected = (
        select_nearest_quantile_rows(
            email,
            score_column=(
                "email_persuasion_risk_score"
            ),
            dataset_name=(
                "ai_email_persuasion_scores"
            ),
        )
    )

    url_selected = (
        select_nearest_quantile_rows(
            url,
            score_column=(
                "url_technical_risk_score"
            ),
            dataset_name=(
                "url_technical_risk_scores"
            ),
        )
    )

    profiles, audit = (
        build_evidence_profiles(
            behavioural_rows=(
                behavioural_selected
            ),
            email_rows=email_selected,
            url_rows=url_selected,
            behavioural_score_column=(
                behavioural_score_column
            ),
        )
    )

    profile_names = [
        quantile_to_profile_name(
            float(value)
        )
        for value in behavioural_selected[
            "selection_quantile"
        ]
    ]

    profile_frame = (
        profiles_to_dataframe(
            profiles,
            profile_names=profile_names,
        )
    )

    markdown = (
        profile_summary_markdown(
            profile_frame
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_frame.to_csv(
        PROFILE_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
        SELECTION_AUDIT_PATH,
        index=False,
        encoding="utf-8",
    )

    with PROFILE_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile_frame.to_dict(
                orient="records"
            ),
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print("=" * 80)
    print("HWI EVIDENCE PROFILES COMPLETE")
    print("=" * 80)
    print(
        f"Behavioural score column: "
        f"{behavioural_score_column}"
    )
    print()
    print(
        profile_frame[
            [
                "profile_name",
                "behavioural_hwi_score",
                "behavioural_hwi_category",
                "behavioural_evidence_status",
                "email_persuasion_risk_score",
                "email_persuasion_category",
                "url_technical_risk_score",
                "url_risk_category",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Important: these records were selected "
        "independently and are not paired."
    )
    print(f"Profiles CSV: {PROFILE_CSV_PATH}")
    print(f"Profiles JSON: {PROFILE_JSON_PATH}")
    print(f"Selection audit: {SELECTION_AUDIT_PATH}")
    print(f"Markdown report: {MARKDOWN_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())