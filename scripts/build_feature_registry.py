from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "references"
    / "feature_registry.csv"
)


FEATURES: list[dict[str, Any]] = [
    # Awareness dataset
    {
        "track": "awareness",
        "feature": "hover_time_seconds",
        "source_columns": "hover_time_ms",
        "feature_type": "numeric",
        "research_role": "behavioural_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "session_duration_minutes",
        "source_columns": "session_duration_sec",
        "feature_type": "numeric",
        "research_role": "behavioural_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "hover_session_ratio",
        "source_columns": "hover_time_ms|session_duration_sec",
        "feature_type": "numeric",
        "research_role": "interaction_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "clicked_link",
        "source_columns": "clicked_link",
        "feature_type": "binary",
        "research_role": "candidate_outcome",
        "hwi_candidate": False,
        "ml_candidate": False,
        "target": True,
        "status": "requires_label_confirmation",
    },
    {
        "track": "awareness",
        "feature": "reported_binary",
        "source_columns": "reported_email",
        "feature_type": "binary",
        "research_role": "security_response_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "requires_timing_confirmation",
    },
    {
        "track": "awareness",
        "feature": "received_hour",
        "source_columns": "email_received_time",
        "feature_type": "numeric",
        "research_role": "context",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "received_day_of_week",
        "source_columns": "email_received_time",
        "feature_type": "categorical",
        "research_role": "context",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "subject_urgency_count",
        "source_columns": "email_subject",
        "feature_type": "numeric",
        "research_role": "urgency_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "awareness",
        "feature": "subject_authority_count",
        "source_columns": "email_subject",
        "feature_type": "numeric",
        "research_role": "authority_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },

    # Email features shared by human, LLM and main corpora
    {
        "track": "email_common",
        "feature": "character_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "text_structure",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "word_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "text_structure",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "lexical_diversity",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "linguistic_complexity",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "url_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "action_request",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "urgency_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "urgency_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "authority_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "authority_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "fear_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "fear_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "financial_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "financial_lure_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "credential_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "credential_request_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "trust_keyword_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "trust_manipulation_proxy",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "email_common",
        "feature": "call_to_action_count",
        "source_columns": "combined_text",
        "feature_type": "numeric",
        "research_role": "action_pressure",
        "hwi_candidate": True,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },

    # URL track
    {
        "track": "phishing_urls",
        "feature": "url_length",
        "source_columns": "url",
        "feature_type": "numeric",
        "research_role": "technical_threat",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "dot_count",
        "source_columns": "url",
        "feature_type": "numeric",
        "research_role": "technical_threat",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "digit_ratio",
        "source_columns": "url",
        "feature_type": "numeric",
        "research_role": "technical_threat",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "https_indicator",
        "source_columns": "url",
        "feature_type": "binary",
        "research_role": "technical_context",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "ip_address_indicator",
        "source_columns": "url",
        "feature_type": "binary",
        "research_role": "technical_threat",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "suspicious_keyword_count",
        "source_columns": "url",
        "feature_type": "numeric",
        "research_role": "technical_threat",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
    {
        "track": "phishing_urls",
        "feature": "url_entropy",
        "source_columns": "url",
        "feature_type": "numeric",
        "research_role": "obfuscation",
        "hwi_candidate": False,
        "ml_candidate": True,
        "target": False,
        "status": "planned",
    },
]


def main() -> int:
    """Write the feature registry to a CSV file."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "track",
        "feature",
        "source_columns",
        "feature_type",
        "research_role",
        "hwi_candidate",
        "ml_candidate",
        "target",
        "status",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(FEATURES)

    print(f"Feature registry written to: {OUTPUT_PATH}")
    print(f"Features registered: {len(FEATURES)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())