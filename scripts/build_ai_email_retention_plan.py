from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "feature_validation"
)

STATISTICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "feature_validation"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "references"
    / "ai_email_feature_retention.csv"
)


def main() -> int:
    """Build a transparent retention plan from validation evidence."""

    quality = pd.read_csv(
        REPORTS_DIR
        / "ai_email_feature_quality_report.csv"
    )

    recommendations = pd.read_csv(
        REPORTS_DIR
        / "ai_email_feature_recommendations.csv"
    )

    phishing = pd.read_csv(
        STATISTICS_DIR
        / "ai_email_phishing_feature_comparison.csv"
    )

    source = pd.read_csv(
        STATISTICS_DIR
        / "ai_email_source_feature_comparison.csv"
    )

    high_correlations_path = (
        STATISTICS_DIR
        / "ai_email_high_correlations.csv"
    )

    if high_correlations_path.exists():
        high_correlations = pd.read_csv(
            high_correlations_path
        )
    else:
        high_correlations = pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    phishing = phishing[
        [
            "feature",
            "effect_size_category",
            "p_value_holm",
            "significant_holm",
        ]
    ].rename(
        columns={
            "effect_size_category": (
                "phishing_effect_category"
            ),
            "p_value_holm": "phishing_p_holm",
            "significant_holm": (
                "phishing_significant"
            ),
        }
    )

    source = source[
        [
            "feature",
            "effect_size_category",
            "p_value_holm",
            "significant_holm",
        ]
    ].rename(
        columns={
            "effect_size_category": (
                "source_effect_category"
            ),
            "p_value_holm": "source_p_holm",
            "significant_holm": (
                "source_significant"
            ),
        }
    )

    plan = (
        recommendations
        .merge(
            quality[
                [
                    "feature",
                    "missing_percentage",
                    "constant_feature",
                    "unique_count",
                ]
            ],
            on="feature",
            how="left",
            suffixes=("", "_quality"),
        )
        .merge(
            phishing,
            on="feature",
            how="left",
        )
        .merge(
            source,
            on="feature",
            how="left",
        )
    )

    required_effect_columns = {
        "phishing_effect",
        "source_effect",
        "four_group_effect",
    }

    missing_effect_columns = (
        required_effect_columns.difference(
            plan.columns
        )
    )

    if missing_effect_columns:
        raise KeyError(
            "Required effect columns are missing after "
            "merging validation reports: "
            + ", ".join(
                sorted(missing_effect_columns)
            )
        )

    correlated_features: set[str] = set()

    if not high_correlations.empty:
        if {
            "feature_1",
            "feature_2",
        }.issubset(high_correlations.columns):
            correlated_features.update(
                high_correlations[
                    "feature_1"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

            correlated_features.update(
                high_correlations[
                    "feature_2"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

    final_decisions: list[str] = []
    decision_reasons: list[str] = []

    for _, row in plan.iterrows():
        feature = str(row["feature"])

        constant_feature = bool(
            row.get(
                "constant_feature",
                False,
            )
        )

        missing_value = row.get(
            "missing_percentage",
            0.0,
        )

        missing_percentage = (
            float(missing_value)
            if pd.notna(missing_value)
            else 0.0
        )

        phishing_effect_value = row.get(
            "phishing_effect",
            0.0,
        )

        phishing_effect = (
            float(phishing_effect_value)
            if pd.notna(
                phishing_effect_value
            )
            else 0.0
        )

        source_effect_value = row.get(
            "source_effect",
            0.0,
        )

        source_effect = (
            float(source_effect_value)
            if pd.notna(source_effect_value)
            else 0.0
        )

        phishing_significant = bool(
            row.get(
                "phishing_significant",
                False,
            )
        )

        source_significant = bool(
            row.get(
                "source_significant",
                False,
            )
        )

        if constant_feature:
            decision = "remove"
            reason = (
                "Constant feature with no variation."
            )

        elif missing_percentage > 50:
            decision = "review"
            reason = (
                "More than 50% missing values."
            )

        elif (
            phishing_significant
            and phishing_effect >= 0.10
        ):
            decision = "retain_phishing"
            reason = (
                "Statistically significant phishing-class "
                "difference with non-negligible effect."
            )

        elif (
            source_significant
            and source_effect >= 0.10
        ):
            decision = "retain_source"
            reason = (
                "Statistically significant human-versus-LLM "
                "difference with non-negligible effect."
            )

        elif feature in correlated_features:
            decision = "review_redundancy"
            reason = (
                "Highly correlated with another engineered "
                "feature."
            )

        else:
            decision = "review"
            reason = (
                "Weak univariate evidence; retain only with "
                "clear theoretical justification."
            )

        final_decisions.append(decision)
        decision_reasons.append(reason)

    plan["final_decision"] = final_decisions
    plan["decision_reason"] = decision_reasons

    preferred_column_order = [
        "feature",
        "final_decision",
        "decision_reason",
        "recommendation",
        "missing_percentage",
        "constant_feature",
        "unique_count",
        "phishing_effect",
        "phishing_effect_category",
        "phishing_p_holm",
        "phishing_significant",
        "source_effect",
        "source_effect_category",
        "source_p_holm",
        "source_significant",
        "high_correlation_flag",
        "four_group_effect",
    ]

    existing_columns = [
        column
        for column in preferred_column_order
        if column in plan.columns
    ]

    plan = plan[
        existing_columns
    ].copy()

    sort_columns = [
        column
        for column in [
            "final_decision",
            "phishing_effect",
            "source_effect",
        ]
        if column in plan.columns
    ]

    if sort_columns:
        ascending_map = {
            "final_decision": True,
            "phishing_effect": False,
            "source_effect": False,
        }

        plan = plan.sort_values(
            by=sort_columns,
            ascending=[
                ascending_map[column]
                for column in sort_columns
            ],
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("AI EMAIL FEATURE RETENTION PLAN")
    print("=" * 80)
    print(f"Output: {OUTPUT_PATH}")
    print()

    print(
        plan["final_decision"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        plan[
            [
                "feature",
                "final_decision",
                "decision_reason",
            ]
        ].to_string(index=False)
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())    