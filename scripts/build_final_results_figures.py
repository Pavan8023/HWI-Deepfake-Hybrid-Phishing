from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PERFORMANCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_results"
    / "model_performance_summary.csv"
)

HWI_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_results"
    / "hwi_framework_summary.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "final_results"
)

PERFORMANCE_FIGURE_PATH = (
    OUTPUT_DIRECTORY
    / "model_performance_comparison.png"
)

CALIBRATION_FIGURE_PATH = (
    OUTPUT_DIRECTORY
    / "calibration_quality_comparison.png"
)

PROFILE_FIGURE_PATH = (
    OUTPUT_DIRECTORY
    / "evidence_profile_scores.png"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "final_results_figure_manifest.csv"
)


def require_file(path: Path) -> None:
    """Confirm that an expected input file exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


def build_performance_figure(
    performance: pd.DataFrame,
) -> Path:
    """Compare discrimination metrics across the three tracks."""

    required_columns = {
        "track",
        "roc_auc",
        "pr_auc",
        "f1",
    }

    missing = required_columns.difference(
        performance.columns
    )

    if missing:
        raise KeyError(
            "Performance table is missing columns: "
            + ", ".join(sorted(missing))
        )

    plot_frame = (
        performance[
            [
                "track",
                "roc_auc",
                "pr_auc",
                "f1",
            ]
        ]
        .set_index("track")
    )

    axis = plot_frame.plot(
        kind="bar",
        figsize=(11, 7),
    )

    axis.set_title(
        "Selected Model Performance by Evidence Track"
    )
    axis.set_xlabel("Evidence track")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1.05)
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.legend(
        title="Metric",
    )

    plt.tight_layout()
    plt.savefig(
        PERFORMANCE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return PERFORMANCE_FIGURE_PATH


def build_calibration_figure(
    performance: pd.DataFrame,
) -> Path:
    """Compare Brier scores across the selected models."""

    required_columns = {
        "track",
        "brier_score",
    }

    missing = required_columns.difference(
        performance.columns
    )

    if missing:
        raise KeyError(
            "Performance table is missing columns: "
            + ", ".join(sorted(missing))
        )

    plot_frame = performance[
        [
            "track",
            "brier_score",
        ]
    ].copy()

    axis = plot_frame.plot(
        x="track",
        y="brier_score",
        kind="bar",
        legend=False,
        figsize=(10, 6),
    )

    axis.set_title(
        "Probability Error by Evidence Track"
    )
    axis.set_xlabel("Evidence track")
    axis.set_ylabel(
        "Brier score (lower is better)"
    )
    axis.tick_params(
        axis="x",
        rotation=15,
    )

    plt.tight_layout()
    plt.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return CALIBRATION_FIGURE_PATH


def build_profile_figure(
    profiles: pd.DataFrame,
) -> Path:
    """Plot the three independent evidence scores in each profile."""

    required_columns = {
        "profile_name",
        "behavioural_hwi_score",
        "email_persuasion_risk_score",
        "url_technical_risk_score",
    }

    missing = required_columns.difference(
        profiles.columns
    )

    if missing:
        raise KeyError(
            "HWI profile table is missing columns: "
            + ", ".join(sorted(missing))
        )

    plot_frame = (
        profiles[
            [
                "profile_name",
                "behavioural_hwi_score",
                "email_persuasion_risk_score",
                "url_technical_risk_score",
            ]
        ]
        .set_index("profile_name")
    )

    axis = plot_frame.plot(
        kind="bar",
        figsize=(12, 7),
    )

    axis.set_title(
        "Independent Evidence Scores in Demonstration Profiles"
    )
    axis.set_xlabel("Demonstration profile")
    axis.set_ylabel("Score on 0–100 scale")
    axis.set_ylim(0, 105)
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.legend(
        [
            "Behavioural HWI estimate",
            "Email persuasion risk",
            "URL technical risk",
        ],
        title="Independent evidence dimension",
    )

    note = (
        "Records were selected independently from separate datasets; "
        "the bars do not represent one verified user or attack."
    )

    plt.figtext(
        0.5,
        0.01,
        note,
        ha="center",
        wrap=True,
    )

    plt.tight_layout(
        rect=(0, 0.05, 1, 1)
    )

    plt.savefig(
        PROFILE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return PROFILE_FIGURE_PATH


def main() -> int:
    """Generate verified dissertation-ready result figures."""

    require_file(
        PERFORMANCE_PATH
    )
    require_file(
        HWI_SUMMARY_PATH
    )

    performance = pd.read_csv(
        PERFORMANCE_PATH
    )

    profiles = pd.read_csv(
        HWI_SUMMARY_PATH
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = [
        {
            "figure_name": (
                "Model performance comparison"
            ),
            "path": str(
                build_performance_figure(
                    performance
                )
            ),
            "supported_interpretation": (
                "Comparison of selected models across "
                "separate empirical tracks."
            ),
        },
        {
            "figure_name": (
                "Calibration quality comparison"
            ),
            "path": str(
                build_calibration_figure(
                    performance
                )
            ),
            "supported_interpretation": (
                "Comparison of probability error; "
                "lower Brier score indicates lower error."
            ),
        },
        {
            "figure_name": (
                "Evidence profile scores"
            ),
            "path": str(
                build_profile_figure(
                    profiles
                )
            ),
            "supported_interpretation": (
                "Independent demonstration scores shown "
                "side by side without a composite score."
            ),
        },
    ]

    pd.DataFrame(
        generated
    ).to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINAL RESULTS FIGURES COMPLETE")
    print("=" * 80)

    for record in generated:
        print(
            f"{record['figure_name']}: "
            f"{record['path']}"
        )

    print(
        f"Figure manifest: {MANIFEST_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())