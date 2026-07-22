from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.hwi_framework import (
    EvidenceProfile,
    HWIFrameworkError,
    create_evidence_profile,
)


DEFAULT_PROFILE_QUANTILES: tuple[float, ...] = (
    0.10,
    0.50,
    0.90,
)


def validate_quantiles(
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    """Validate and return ordered scenario-selection quantiles."""

    if not quantiles:
        raise HWIFrameworkError(
            "At least one profile quantile is required."
        )

    validated = tuple(
        float(value)
        for value in quantiles
    )

    if any(
        not np.isfinite(value)
        or not 0 <= value <= 1
        for value in validated
    ):
        raise HWIFrameworkError(
            "Profile quantiles must be finite values between 0 and 1."
        )

    if len(set(validated)) != len(validated):
        raise HWIFrameworkError(
            "Profile quantiles must be unique."
        )

    return tuple(sorted(validated))


def validate_score_frame(
    dataframe: pd.DataFrame,
    *,
    score_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate one model-output table used for profile selection."""

    if dataframe.empty:
        raise HWIFrameworkError(
            f"{dataset_name} score dataset is empty."
        )

    if score_column not in dataframe.columns:
        raise HWIFrameworkError(
            f"{dataset_name} is missing score column "
            f"'{score_column}'."
        )

    validated = dataframe.copy()

    validated[score_column] = pd.to_numeric(
        validated[score_column],
        errors="coerce",
    )

    invalid_mask = (
        validated[score_column].isna()
        | ~np.isfinite(
            validated[score_column].to_numpy(
                dtype=float
            )
        )
        | validated[score_column].lt(0)
        | validated[score_column].gt(100)
    )

    if invalid_mask.any():
        raise HWIFrameworkError(
            f"{dataset_name} contains invalid scores in "
            f"'{score_column}'."
        )

    return validated.reset_index(drop=True)


def select_nearest_quantile_rows(
    dataframe: pd.DataFrame,
    *,
    score_column: str,
    quantiles: Sequence[float] = DEFAULT_PROFILE_QUANTILES,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Select rows nearest to requested score quantiles.

    Selection is performed independently for each dataset.
    """

    validated_quantiles = validate_quantiles(
        quantiles
    )

    validated_frame = validate_score_frame(
        dataframe,
        score_column=score_column,
        dataset_name=dataset_name,
    )

    scores = validated_frame[
        score_column
    ]

    selected_records: list[dict[str, Any]] = []

    used_indices: set[int] = set()

    for quantile in validated_quantiles:
        target_score = float(
            scores.quantile(quantile)
        )

        distances = (
            scores - target_score
        ).abs()

        ordered_indices = (
            distances.sort_values(
                kind="stable"
            )
            .index.tolist()
        )

        selected_index = next(
            (
                int(index)
                for index in ordered_indices
                if int(index) not in used_indices
            ),
            int(ordered_indices[0]),
        )

        used_indices.add(selected_index)

        record = (
            validated_frame.loc[
                selected_index
            ]
            .to_dict()
        )

        record["selection_quantile"] = quantile
        record["quantile_target_score"] = (
            target_score
        )
        record["selected_row_index"] = (
            selected_index
        )
        record["source_dataset"] = (
            dataset_name
        )

        selected_records.append(record)

    return pd.DataFrame(
        selected_records
    )


def quantile_to_profile_name(
    quantile: float,
) -> str:
    """Return a neutral profile name for an observed quantile."""

    if quantile < 0.34:
        return "lower_observed_context"

    if quantile < 0.67:
        return "middle_observed_context"

    return "higher_observed_context"


def build_evidence_profiles(
    *,
    behavioural_rows: pd.DataFrame,
    email_rows: pd.DataFrame,
    url_rows: pd.DataFrame,
    behavioural_score_column: str = "hwi_score",
    email_score_column: str = (
        "email_persuasion_risk_score"
    ),
    url_score_column: str = (
        "url_technical_risk_score"
    ),
) -> tuple[
    list[EvidenceProfile],
    pd.DataFrame,
]:
    """
    Build aligned demonstration profiles from independently selected rows.

    The row positions are aligned only for presentation. They are not
    interpreted as records from the same person or phishing event.
    """

    if not (
        len(behavioural_rows)
        == len(email_rows)
        == len(url_rows)
    ):
        raise HWIFrameworkError(
            "Selected evidence tables must contain equal row counts."
        )

    profiles: list[EvidenceProfile] = []
    audit_records: list[dict[str, Any]] = []

    for position in range(
        len(behavioural_rows)
    ):
        behavioural_row = (
            behavioural_rows.iloc[position]
        )
        email_row = email_rows.iloc[position]
        url_row = url_rows.iloc[position]

        quantile = float(
            behavioural_row[
                "selection_quantile"
            ]
        )

        profile_name = (
            quantile_to_profile_name(
                quantile
            )
        )

        behavioural_reference = (
            f"awareness_test_row_"
            f"{int(behavioural_row['selected_row_index'])}"
        )

        email_reference = (
            f"ai_email_score_row_"
            f"{int(email_row['selected_row_index'])}"
        )

        url_reference = (
            f"url_score_row_"
            f"{int(url_row['selected_row_index'])}"
        )

        profile = create_evidence_profile(
            behavioural_hwi_score=float(
                behavioural_row[
                    behavioural_score_column
                ]
            ),
            email_persuasion_risk_score=float(
                email_row[
                    email_score_column
                ]
            ),
            url_technical_risk_score=float(
                url_row[
                    url_score_column
                ]
            ),
            behavioural_evidence_status=(
                "unsupported"
            ),
            email_evidence_status=(
                "context_only"
            ),
            url_evidence_status=(
                "context_only"
            ),
            behavioural_record_reference=(
                behavioural_reference
            ),
            email_record_reference=(
                email_reference
            ),
            url_record_reference=url_reference,
        )

        profiles.append(profile)

        audit_records.append(
            {
                "profile_name": profile_name,
                "selection_quantile": quantile,
                "behavioural_selected_index": int(
                    behavioural_row[
                        "selected_row_index"
                    ]
                ),
                "email_selected_index": int(
                    email_row[
                        "selected_row_index"
                    ]
                ),
                "url_selected_index": int(
                    url_row[
                        "selected_row_index"
                    ]
                ),
                "records_are_paired": False,
                "selection_method": (
                    "Independent nearest-quantile "
                    "selection within each dataset"
                ),
            }
        )

    return profiles, pd.DataFrame(
        audit_records
    )


def profiles_to_dataframe(
    profiles: Sequence[EvidenceProfile],
    *,
    profile_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert evidence profiles into a reporting table."""

    records = [
        asdict(profile)
        for profile in profiles
    ]

    frame = pd.DataFrame(records)

    if profile_names is not None:
        if len(profile_names) != len(frame):
            raise HWIFrameworkError(
                "Profile-name count does not match profile count."
            )

        frame.insert(
            0,
            "profile_name",
            list(profile_names),
        )

    return frame


def profile_summary_markdown(
    profiles: pd.DataFrame,
) -> str:
    """Build a dissertation-ready Markdown summary."""

    required_columns = {
        "profile_name",
        "behavioural_hwi_score",
        "behavioural_hwi_category",
        "behavioural_evidence_status",
        "email_persuasion_risk_score",
        "email_persuasion_category",
        "url_technical_risk_score",
        "url_risk_category",
        "methodological_warning",
    }

    missing = required_columns.difference(
        profiles.columns
    )

    if missing:
        raise HWIFrameworkError(
            "Profile table is missing columns: "
            + ", ".join(sorted(missing))
        )

    lines = [
        "# HWI Evidence Profiles",
        "",
        "These profiles present independent evidence dimensions "
        "side by side. They do not represent verified paired "
        "records from the same user or phishing event.",
        "",
    ]

    for row in profiles.to_dict(
        orient="records"
    ):
        lines.extend(
            [
                f"## {row['profile_name']}",
                "",
                (
                    f"- Behavioural HWI estimate: "
                    f"{row['behavioural_hwi_score']:.2f} "
                    f"({row['behavioural_hwi_category']})"
                ),
                (
                    f"- Behavioural evidence status: "
                    f"{row['behavioural_evidence_status']}"
                ),
                (
                    f"- Email persuasion-risk score: "
                    f"{row['email_persuasion_risk_score']:.2f} "
                    f"({row['email_persuasion_category']})"
                ),
                (
                    f"- URL technical-risk score: "
                    f"{row['url_technical_risk_score']:.2f} "
                    f"({row['url_risk_category']})"
                ),
                "",
                f"Interpretation: {row['interpretation']}",
                "",
            ]
        )

    warning = str(
        profiles.iloc[0][
            "methodological_warning"
        ]
    )

    lines.extend(
        [
            "## Methodological warning",
            "",
            warning,
            "",
        ]
    )

    return "\n".join(lines)