from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


class HWIFrameworkError(RuntimeError):
    """Raised when an HWI evidence profile cannot be created safely."""


EvidenceStatus = Literal[
    "unsupported",
    "weak",
    "moderate",
    "strong",
    "context_only",
]


@dataclass(frozen=True)
class ScoreThresholds:
    """
    Provisional presentation bands for 0–100 scores.

    These are interpretation bands only and are not clinically
    or psychologically validated thresholds.
    """

    low_max: float = 30.0
    medium_max: float = 70.0

    def validate(self) -> None:
        """Validate threshold ordering and score range."""

        if not 0 <= self.low_max < self.medium_max <= 100:
            raise HWIFrameworkError(
                "Thresholds must satisfy "
                "0 <= low_max < medium_max <= 100."
            )


@dataclass(frozen=True)
class EvidenceProfile:
    """
    Structured output for one demonstration scenario.

    The three scores remain separate because they originate from
    different datasets and units of analysis.
    """

    behavioural_hwi_score: float
    behavioural_hwi_category: str
    behavioural_evidence_status: EvidenceStatus

    email_persuasion_risk_score: float
    email_persuasion_category: str
    email_evidence_status: EvidenceStatus

    url_technical_risk_score: float
    url_risk_category: str
    url_evidence_status: EvidenceStatus

    interpretation: str
    methodological_warning: str

    behavioural_record_reference: str | None = None
    email_record_reference: str | None = None
    url_record_reference: str | None = None


DEFAULT_THRESHOLDS = ScoreThresholds()


METHODOLOGICAL_WARNING = (
    "The behavioural HWI is an exploratory proxy derived from the selected "
    "awareness dataset. Email persuasion and URL technical-risk scores "
    "describe attack characteristics and are not direct measurements of "
    "individual human weakness. Records from separate datasets are combined "
    "for demonstration only and must not be interpreted as one verified "
    "real-world phishing event."
)


def validate_score(
    score: float | int | np.number,
    *,
    score_name: str,
) -> float:
    """Validate and return one numeric score on the 0–100 scale."""

    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise HWIFrameworkError(
            f"{score_name} must be numeric."
        ) from exc

    if not np.isfinite(numeric_score):
        raise HWIFrameworkError(
            f"{score_name} must be finite."
        )

    if not 0 <= numeric_score <= 100:
        raise HWIFrameworkError(
            f"{score_name} must be between 0 and 100."
        )

    return numeric_score


def classify_score(
    score: float | int | np.number,
    *,
    thresholds: ScoreThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Classify a 0–100 score as low, medium or high."""

    thresholds.validate()

    numeric_score = validate_score(
        score,
        score_name="score",
    )

    if numeric_score <= thresholds.low_max:
        return "low"

    if numeric_score <= thresholds.medium_max:
        return "medium"

    return "high"


def validate_evidence_status(
    status: str,
) -> EvidenceStatus:
    """Validate an evidence-strength label."""

    allowed_statuses: set[str] = {
        "unsupported",
        "weak",
        "moderate",
        "strong",
        "context_only",
    }

    if status not in allowed_statuses:
        raise HWIFrameworkError(
            "Unsupported evidence status: "
            f"{status}. Allowed values are: "
            + ", ".join(sorted(allowed_statuses))
        )

    return status  # type: ignore[return-value]


def build_profile_interpretation(
    *,
    behavioural_category: str,
    behavioural_status: EvidenceStatus,
    email_category: str,
    url_category: str,
) -> str:
    """
    Build a conservative scenario interpretation.

    The interpretation deliberately avoids calculating a composite score.
    """

    behavioural_text = (
        f"The behavioural model assigned a {behavioural_category} "
        "exploratory susceptibility category"
    )

    if behavioural_status == "unsupported":
        behavioural_text += (
            ", but the behavioural model did not demonstrate reliable "
            "discrimination and this result should not be treated as a "
            "validated individual susceptibility estimate"
        )

    elif behavioural_status == "weak":
        behavioural_text += (
            ", supported only by weak predictive evidence"
        )

    elif behavioural_status == "moderate":
        behavioural_text += (
            ", supported by moderate predictive evidence"
        )

    elif behavioural_status == "strong":
        behavioural_text += (
            ", supported by strong predictive evidence"
        )

    else:
        behavioural_text += (
            ", provided for contextual presentation only"
        )

    email_text = (
        f"The selected email record shows {email_category} "
        "phishing-oriented persuasion risk."
    )

    url_text = (
        f"The selected URL record shows {url_category} "
        "technical maliciousness risk."
    )

    separation_text = (
        "These dimensions are reported separately because the datasets "
        "do not contain verified shared identifiers."
    )

    return (
        f"{behavioural_text}. "
        f"{email_text} "
        f"{url_text} "
        f"{separation_text}"
    )


def create_evidence_profile(
    *,
    behavioural_hwi_score: float,
    email_persuasion_risk_score: float,
    url_technical_risk_score: float,
    behavioural_evidence_status: EvidenceStatus = "unsupported",
    email_evidence_status: EvidenceStatus = "context_only",
    url_evidence_status: EvidenceStatus = "context_only",
    thresholds: ScoreThresholds = DEFAULT_THRESHOLDS,
    behavioural_record_reference: str | None = None,
    email_record_reference: str | None = None,
    url_record_reference: str | None = None,
) -> EvidenceProfile:
    """
    Create one structured HWI evidence profile.

    No composite numeric score is calculated.
    """

    thresholds.validate()

    behavioural_score = validate_score(
        behavioural_hwi_score,
        score_name="behavioural_hwi_score",
    )

    email_score = validate_score(
        email_persuasion_risk_score,
        score_name="email_persuasion_risk_score",
    )

    url_score = validate_score(
        url_technical_risk_score,
        score_name="url_technical_risk_score",
    )

    validated_behavioural_status = (
        validate_evidence_status(
            behavioural_evidence_status
        )
    )

    validated_email_status = (
        validate_evidence_status(
            email_evidence_status
        )
    )

    validated_url_status = (
        validate_evidence_status(
            url_evidence_status
        )
    )

    behavioural_category = classify_score(
        behavioural_score,
        thresholds=thresholds,
    )

    email_category = classify_score(
        email_score,
        thresholds=thresholds,
    )

    url_category = classify_score(
        url_score,
        thresholds=thresholds,
    )

    interpretation = build_profile_interpretation(
        behavioural_category=behavioural_category,
        behavioural_status=validated_behavioural_status,
        email_category=email_category,
        url_category=url_category,
    )

    return EvidenceProfile(
        behavioural_hwi_score=round(
            behavioural_score,
            2,
        ),
        behavioural_hwi_category=(
            behavioural_category
        ),
        behavioural_evidence_status=(
            validated_behavioural_status
        ),
        email_persuasion_risk_score=round(
            email_score,
            2,
        ),
        email_persuasion_category=(
            email_category
        ),
        email_evidence_status=(
            validated_email_status
        ),
        url_technical_risk_score=round(
            url_score,
            2,
        ),
        url_risk_category=url_category,
        url_evidence_status=(
            validated_url_status
        ),
        interpretation=interpretation,
        methodological_warning=(
            METHODOLOGICAL_WARNING
        ),
        behavioural_record_reference=(
            behavioural_record_reference
        ),
        email_record_reference=(
            email_record_reference
        ),
        url_record_reference=(
            url_record_reference
        ),
    )


def profile_to_dict(
    profile: EvidenceProfile,
) -> dict[str, Any]:
    """Convert one evidence profile to a serializable dictionary."""

    return asdict(profile)


def profile_to_dataframe(
    profile: EvidenceProfile,
) -> pd.DataFrame:
    """Convert one profile into a single-row DataFrame."""

    return pd.DataFrame(
        [profile_to_dict(profile)]
    )


def create_profile_from_rows(
    *,
    behavioural_row: Mapping[str, Any],
    email_row: Mapping[str, Any],
    url_row: Mapping[str, Any],
    behavioural_score_column: str = "hwi_score",
    email_score_column: str = (
        "email_persuasion_risk_score"
    ),
    url_score_column: str = (
        "url_technical_risk_score"
    ),
    behavioural_evidence_status: EvidenceStatus = (
        "unsupported"
    ),
) -> EvidenceProfile:
    """
    Build a scenario profile from one row from each output dataset.

    Rows are selected for demonstration only and are not treated as
    records from the same verified event.
    """

    missing: list[str] = []

    if behavioural_score_column not in behavioural_row:
        missing.append(
            f"behavioural:{behavioural_score_column}"
        )

    if email_score_column not in email_row:
        missing.append(
            f"email:{email_score_column}"
        )

    if url_score_column not in url_row:
        missing.append(
            f"url:{url_score_column}"
        )

    if missing:
        raise HWIFrameworkError(
            "Required scenario score fields are missing: "
            + ", ".join(missing)
        )

    return create_evidence_profile(
        behavioural_hwi_score=float(
            behavioural_row[
                behavioural_score_column
            ]
        ),
        email_persuasion_risk_score=float(
            email_row[email_score_column]
        ),
        url_technical_risk_score=float(
            url_row[url_score_column]
        ),
        behavioural_evidence_status=(
            behavioural_evidence_status
        ),
        email_evidence_status="context_only",
        url_evidence_status="context_only",
        behavioural_record_reference=(
            str(
                behavioural_row.get(
                    "record_reference",
                    "",
                )
            )
            or None
        ),
        email_record_reference=(
            str(
                email_row.get(
                    "record_reference",
                    "",
                )
            )
            or None
        ),
        url_record_reference=(
            str(
                url_row.get(
                    "record_reference",
                    "",
                )
            )
            or None
        ),
    )