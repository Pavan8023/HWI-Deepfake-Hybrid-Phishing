from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "pipeline"
)

PIPELINE_LOG_PATH = (
    PIPELINE_REPORT_DIRECTORY
    / "full_pipeline.log"
)

PIPELINE_SUMMARY_PATH = (
    PIPELINE_REPORT_DIRECTORY
    / "full_pipeline_summary.json"
)


def project_path(*parts: str) -> Path:
    """Build a project-relative path using pathlib."""

    return PROJECT_ROOT.joinpath(*parts)


class PipelineError(RuntimeError):
    """Raised when one or more pipeline stages fail."""


@dataclass(frozen=True)
class PipelineStage:
    """Definition of one executable pipeline stage."""

    name: str
    command: tuple[str, ...]
    description: str
    expected_outputs: tuple[Path, ...] = ()
    heavy: bool = False


@dataclass
class StageResult:
    """Execution result for one pipeline stage."""

    name: str
    description: str
    command: list[str]
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    return_code: int | None
    expected_outputs: list[str]
    missing_outputs: list[str]
    error: str | None = None


def python_command(
    relative_script_path: str,
    *arguments: str,
) -> tuple[str, ...]:
    """Build a command using the active Python interpreter."""

    return (
        sys.executable,
        str(project_path(relative_script_path)),
        *arguments,
    )


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        name="test_suite_before_pipeline",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ),
        description=(
            "Run the complete automated test suite before "
            "regenerating research outputs."
        ),
    ),
    PipelineStage(
        name="exploratory_data_analysis",
        command=python_command(
            "scripts/run_eda.py",
            "--category",
            "awareness",
        ),
        description=(
            "Run the current production EDA workflow for the "
            "awareness category."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "eda",
                "awareness_eda_comparison.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "eda",
                "awareness_eda_results.json",
            ),
        ),
        heavy=True,
    ),
    PipelineStage(
        name="feature_registry",
        command=python_command(
            "scripts/build_feature_registry.py"
        ),
        description=(
            "Regenerate the feature registry and feature documentation."
        ),
        expected_outputs=(
            project_path(
                "references",
                "feature_registry.csv",
            ),
        ),
    ),
    PipelineStage(
        name="ai_email_feature_dataset",
        command=python_command(
            "scripts/build_ai_email_features.py"
        ),
        description=(
            "Build the harmonised AI-email engineered feature dataset, "
            "including duplicate-safe template group identifiers."
        ),
        expected_outputs=(
            project_path(
                "data",
                "processed",
                "ai_email_features.csv",
            ),
            project_path(
                "data",
                "processed",
                "ai_email_features_summary.json",
            ),
        ),
    ),
    PipelineStage(
        name="ai_email_feature_validation",
        command=python_command(
            "scripts/run_ai_email_feature_validation.py"
        ),
        description=(
            "Run statistical validation of engineered AI-email features."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "feature_validation",
                "ai_email_feature_validation_summary.json",
            ),
            project_path(
                "outputs",
                "reports",
                "feature_validation",
                "ai_email_feature_quality_report.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "feature_validation",
                "ai_email_feature_recommendations.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_group_descriptive_statistics.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_kruskal_wallis.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_pairwise_mann_whitney.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_phishing_feature_comparison.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_source_feature_comparison.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_spearman_correlation.csv",
            ),
            project_path(
                "outputs",
                "statistics",
                "feature_validation",
                "ai_email_high_correlations.csv",
            ),
        ),
    ),
    PipelineStage(
        name="ai_email_retention_plan",
        command=python_command(
            "scripts/build_ai_email_retention_plan.py"
        ),
        description=(
            "Regenerate the evidence-based AI-email feature "
            "retention plan."
        ),
        expected_outputs=(
            project_path(
                "references",
                "ai_email_feature_retention.csv",
            ),
        ),
    ),
    PipelineStage(
        name="awareness_preprocessing",
        command=python_command(
            "scripts/build_awareness_model_dataset.py"
        ),
        description=(
            "Prepare the behavioural awareness train/test datasets."
        ),
        expected_outputs=(
            project_path(
                "data",
                "processed",
                "awareness_x_train.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_x_test.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_y_train.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_y_test.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_train_complete.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_test_complete.csv",
            ),
            project_path(
                "data",
                "processed",
                "awareness_preprocessing_summary.json",
            ),
            project_path(
                "outputs",
                "models",
                "awareness_preprocessor.joblib",
            ),
        ),
    ),
    PipelineStage(
        name="awareness_model_training",
        command=python_command(
            "scripts/train_awareness_models.py"
        ),
        description=(
            "Train, calibrate and select the behavioural "
            "awareness model."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "model_training",
                "awareness_model_comparison.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "model_training",
                "awareness_model_metrics.json",
            ),
            project_path(
                "outputs",
                "reports",
                "model_training",
                "awareness_cross_validation.json",
            ),
            project_path(
                "models",
                "awareness_best_calibrated_model.joblib",
            ),
            project_path(
                "outputs",
                "predictions",
                "awareness_test_hwi.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "model_training",
                "awareness_test_hwi_summary.json",
            ),
        ),
        heavy=True,
    ),
    PipelineStage(
        name="awareness_model_evaluation",
        command=python_command(
            "scripts/evaluate_awareness_model.py"
        ),
        description=(
            "Generate final behavioural evaluation metrics, "
            "confidence interval and figures."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "evaluation",
                "awareness_final_evaluation.json",
            ),
            project_path(
                "outputs",
                "reports",
                "evaluation",
                "awareness_probability_summary.csv",
            ),
            project_path(
                "outputs",
                "figures",
                "evaluation",
                "awareness_roc_curve.png",
            ),
            project_path(
                "outputs",
                "figures",
                "evaluation",
                "awareness_precision_recall_curve.png",
            ),
            project_path(
                "outputs",
                "figures",
                "evaluation",
                "awareness_calibration_curve.png",
            ),
            project_path(
                "outputs",
                "figures",
                "evaluation",
                "awareness_confusion_matrix.png",
            ),
            project_path(
                "outputs",
                "figures",
                "evaluation",
                "awareness_probability_distribution.png",
            ),
        ),
    ),
    PipelineStage(
        name="ai_email_model_training",
        command=python_command(
            "scripts/train_ai_email_persuasion_model.py"
        ),
        description=(
            "Train the duplicate-safe AI-email phishing and "
            "persuasion-risk model."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "ai_email_model",
                "ai_email_model_comparison.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "ai_email_model",
                "ai_email_model_metrics.json",
            ),
            project_path(
                "outputs",
                "reports",
                "ai_email_model",
                "ai_email_split_summary.json",
            ),
            project_path(
                "models",
                "ai_email_best_calibrated_model.joblib",
            ),
            project_path(
                "outputs",
                "predictions",
                "ai_email_test_persuasion_scores.csv",
            ),
            project_path(
                "outputs",
                "predictions",
                "ai_email_all_persuasion_scores.csv",
            ),
        ),
        heavy=True,
    ),
    PipelineStage(
        name="ai_email_duplicate_audit",
        command=python_command(
            "scripts/audit_ai_email_leakage.py"
        ),
        description=(
            "Regenerate the AI-email exact-duplicate and "
            "template-duplicate audit."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "ai_email_model",
                "ai_email_duplicate_audit.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "ai_email_model",
                "ai_email_duplicate_templates.csv",
            ),
        ),
    ),
    PipelineStage(
        name="url_risk_model_training",
        command=python_command(
            "scripts/train_url_risk_model.py"
        ),
        description=(
            "Build lexical URL features and train the calibrated "
            "technical-risk model."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "url_risk_model",
                "url_model_comparison.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "url_risk_model",
                "url_model_metrics.json",
            ),
            project_path(
                "outputs",
                "reports",
                "url_risk_model",
                "url_dataset_summary.json",
            ),
            project_path(
                "outputs",
                "reports",
                "url_risk_model",
                "url_split_summary.json",
            ),
            project_path(
                "models",
                "url_best_calibrated_model.joblib",
            ),
            project_path(
                "outputs",
                "predictions",
                "url_test_risk_scores.csv",
            ),
        ),
        heavy=True,
    ),
    PipelineStage(
        name="hwi_evidence_profiles",
        command=python_command(
            "scripts/build_hwi_evidence_profiles.py"
        ),
        description=(
            "Create independently selected behavioural, email and "
            "URL evidence profiles without calculating a composite score."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "hwi_framework",
                "hwi_evidence_profiles.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "hwi_framework",
                "hwi_evidence_profiles.json",
            ),
            project_path(
                "outputs",
                "reports",
                "hwi_framework",
                "hwi_profile_selection_audit.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "hwi_framework",
                "hwi_evidence_profiles.md",
            ),
        ),
    ),
    PipelineStage(
        name="final_results_summary",
        command=python_command(
            "scripts/build_final_results_summary.py"
        ),
        description=(
            "Consolidate verified metrics and interpretations into "
            "dissertation-ready tables and reports."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "reports",
                "final_results",
                "model_performance_summary.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "final_results",
                "dataset_and_model_summary.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "final_results",
                "hwi_framework_summary.csv",
            ),
            project_path(
                "outputs",
                "reports",
                "final_results",
                "final_results_summary.json",
            ),
            project_path(
                "outputs",
                "reports",
                "final_results",
                "dissertation_results_summary.md",
            ),
        ),
    ),
    PipelineStage(
        name="final_results_figures",
        command=python_command(
            "scripts/build_final_results_figures.py"
        ),
        description=(
            "Generate final dissertation-ready comparison and "
            "evidence-profile figures."
        ),
        expected_outputs=(
            project_path(
                "outputs",
                "figures",
                "final_results",
                "model_performance_comparison.png",
            ),
            project_path(
                "outputs",
                "figures",
                "final_results",
                "calibration_quality_comparison.png",
            ),
            project_path(
                "outputs",
                "figures",
                "final_results",
                "evidence_profile_scores.png",
            ),
            project_path(
                "outputs",
                "figures",
                "final_results",
                "final_results_figure_manifest.csv",
            ),
        ),
    ),
    PipelineStage(
        name="test_suite_after_pipeline",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ),
        description=(
            "Run the complete automated test suite after output generation."
        ),
    ),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete HWI dissertation pipeline from "
            "raw data to final reports."
        )
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help=(
            "Skip the automated test stages before and after "
            "the pipeline."
        ),
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help=(
            "Skip EDA regeneration when existing EDA outputs "
            "are already available."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip tests and EDA. Model training and final "
            "report generation still run."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to later stages after a failure. "
            "The default behaviour is fail-fast."
        ),
    )

    parser.add_argument(
        "--from-stage",
        type=str,
        default=None,
        help="Begin execution from the named pipeline stage.",
    )

    parser.add_argument(
        "--to-stage",
        type=str,
        default=None,
        help="Stop execution after the named pipeline stage.",
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available pipeline stage names and exit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned commands without executing them."
        ),
    )

    return parser.parse_args()


def timestamp() -> str:
    """Return the current local timestamp."""

    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def write_log(message: str) -> None:
    """Print and append a message to the pipeline log."""

    print(message, flush=True)

    PIPELINE_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PIPELINE_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(message + "\n")


def format_command(
    command: Sequence[str],
) -> str:
    """Format a command for readable logging."""

    return subprocess.list2cmdline(
        list(command)
    )


def validate_stage_range(
    stages: Sequence[PipelineStage],
    *,
    from_stage: str | None,
    to_stage: str | None,
) -> list[PipelineStage]:
    """Select a valid inclusive range of pipeline stages."""

    stage_names = [
        stage.name
        for stage in stages
    ]

    start_index = 0
    end_index = len(stages)

    if from_stage is not None:
        if from_stage not in stage_names:
            raise PipelineError(
                f"Unknown --from-stage value: {from_stage}"
            )

        start_index = stage_names.index(
            from_stage
        )

    if to_stage is not None:
        if to_stage not in stage_names:
            raise PipelineError(
                f"Unknown --to-stage value: {to_stage}"
            )

        end_index = (
            stage_names.index(to_stage)
            + 1
        )

    if start_index >= end_index:
        raise PipelineError(
            "--from-stage occurs after --to-stage."
        )

    return list(
        stages[start_index:end_index]
    )


def filter_stages(
    stages: Sequence[PipelineStage],
    *,
    skip_tests: bool,
    skip_eda: bool,
    quick: bool,
) -> list[PipelineStage]:
    """Apply optional execution filters."""

    selected: list[PipelineStage] = []

    for stage in stages:
        is_test_stage = (
            stage.name.startswith(
                "test_suite_"
            )
        )

        is_eda_stage = (
            stage.name
            == "exploratory_data_analysis"
        )

        if quick and (
            is_test_stage
            or is_eda_stage
        ):
            continue

        if skip_tests and is_test_stage:
            continue

        if skip_eda and is_eda_stage:
            continue

        selected.append(stage)

    return selected


def check_expected_outputs(
    stage: PipelineStage,
) -> list[str]:
    """Return expected output paths that do not exist."""

    return [
        str(path)
        for path in stage.expected_outputs
        if not path.exists()
    ]


def execute_stage(
    stage: PipelineStage,
    *,
    dry_run: bool,
) -> StageResult:
    """Execute one pipeline stage and capture its result."""

    started_at = timestamp()
    start_time = time.perf_counter()

    write_log("")
    write_log("=" * 80)
    write_log(f"STAGE: {stage.name}")
    write_log("=" * 80)
    write_log(stage.description)
    write_log(
        "Command: "
        + format_command(stage.command)
    )

    if dry_run:
        write_log("Status: DRY RUN")

        return StageResult(
            name=stage.name,
            description=stage.description,
            command=list(stage.command),
            status="dry_run",
            started_at=started_at,
            completed_at=timestamp(),
            duration_seconds=0.0,
            return_code=None,
            expected_outputs=[
                str(path)
                for path in stage.expected_outputs
            ],
            missing_outputs=[],
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(
        PROJECT_ROOT
    )

    try:
        completed = subprocess.run(
            stage.command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            check=False,
        )
    except OSError as exc:
        duration = (
            time.perf_counter()
            - start_time
        )

        write_log(
            f"Status: FAILED TO START - {exc}"
        )

        return StageResult(
            name=stage.name,
            description=stage.description,
            command=list(stage.command),
            status="failed",
            started_at=started_at,
            completed_at=timestamp(),
            duration_seconds=round(
                duration,
                3,
            ),
            return_code=None,
            expected_outputs=[
                str(path)
                for path in stage.expected_outputs
            ],
            missing_outputs=[
                str(path)
                for path in stage.expected_outputs
                if not path.exists()
            ],
            error=str(exc),
        )

    duration = (
        time.perf_counter()
        - start_time
    )

    missing_outputs = (
        check_expected_outputs(stage)
    )

    if completed.returncode != 0:
        status = "failed"
        error = (
            f"Command exited with code "
            f"{completed.returncode}."
        )
    elif missing_outputs:
        status = "failed"
        error = (
            "The command completed, but one or more "
            "expected outputs were not created."
        )
    else:
        status = "passed"
        error = None

    write_log(
        f"Status: {status.upper()}"
    )
    write_log(
        f"Duration: {duration:.2f} seconds"
    )

    if missing_outputs:
        write_log(
            "Missing expected outputs:"
        )

        for path in missing_outputs:
            write_log(f"  - {path}")

    return StageResult(
        name=stage.name,
        description=stage.description,
        command=list(stage.command),
        status=status,
        started_at=started_at,
        completed_at=timestamp(),
        duration_seconds=round(
            duration,
            3,
        ),
        return_code=completed.returncode,
        expected_outputs=[
            str(path)
            for path in stage.expected_outputs
        ],
        missing_outputs=missing_outputs,
        error=error,
    )


def write_summary(
    results: Sequence[StageResult],
    *,
    pipeline_started_at: str,
    pipeline_completed_at: str,
    total_duration_seconds: float,
    command_line: Sequence[str],
) -> None:
    """Write the machine-readable pipeline summary."""

    passed = sum(
        result.status == "passed"
        for result in results
    )

    failed = sum(
        result.status == "failed"
        for result in results
    )

    skipped = sum(
        result.status == "skipped"
        for result in results
    )

    dry_run = sum(
        result.status == "dry_run"
        for result in results
    )

    payload = {
        "project": (
            "Human Weakness Index dissertation"
        ),
        "pipeline_started_at": (
            pipeline_started_at
        ),
        "pipeline_completed_at": (
            pipeline_completed_at
        ),
        "total_duration_seconds": round(
            total_duration_seconds,
            3,
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "project_root": str(PROJECT_ROOT),
        "command_line": list(command_line),
        "summary": {
            "stages_executed": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "dry_run": dry_run,
            "pipeline_successful": (
                failed == 0
            ),
        },
        "scientific_scope": {
            "behavioural_hwi": (
                "Exploratory behavioural susceptibility proxy; "
                "predictive discrimination unsupported."
            ),
            "ai_email_score": (
                "Email-level phishing and persuasion-risk "
                "indicator, not human susceptibility."
            ),
            "url_risk_score": (
                "URL-level technical maliciousness indicator, "
                "not human susceptibility."
            ),
            "composite_hwi_calculated": False,
            "independent_records_paired": False,
        },
        "stages": [
            asdict(result)
            for result in results
        ],
    }

    PIPELINE_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PIPELINE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            payload,
            summary_file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


def print_stage_list() -> None:
    """Display all available pipeline stage names."""

    print("Available pipeline stages:")
    print()

    for number, stage in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        heavy_label = (
            " [heavy]"
            if stage.heavy
            else ""
        )

        print(
            f"{number:02d}. "
            f"{stage.name}{heavy_label}"
        )
        print(
            f"    {stage.description}"
        )


def main() -> int:
    """Run the complete dissertation pipeline."""

    arguments = parse_arguments()

    if arguments.list_stages:
        print_stage_list()
        return 0

    selected_stages = validate_stage_range(
        PIPELINE_STAGES,
        from_stage=arguments.from_stage,
        to_stage=arguments.to_stage,
    )

    selected_stages = filter_stages(
        selected_stages,
        skip_tests=arguments.skip_tests,
        skip_eda=arguments.skip_eda,
        quick=arguments.quick,
    )

    if not selected_stages:
        raise PipelineError(
            "No pipeline stages remain after filtering."
        )

    PIPELINE_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PIPELINE_LOG_PATH.write_text(
        "",
        encoding="utf-8",
    )

    pipeline_started_at = timestamp()
    overall_start = time.perf_counter()

    write_log("=" * 80)
    write_log("HWI DISSERTATION FULL PIPELINE")
    write_log("=" * 80)
    write_log(
        f"Project root: {PROJECT_ROOT}"
    )
    write_log(
        f"Python: {sys.executable}"
    )
    write_log(
        f"Started: {pipeline_started_at}"
    )
    write_log(
        f"Stages selected: "
        f"{len(selected_stages)}"
    )
    write_log("")
    write_log(
        "Scientific scope: the behavioural HWI remains "
        "exploratory. Email and URL scores are separate "
        "attack-context indicators. No unsupported composite "
        "score is calculated, and records are not treated as paired."
    )

    results: list[StageResult] = []

    for stage in selected_stages:
        result = execute_stage(
            stage,
            dry_run=arguments.dry_run,
        )

        results.append(result)

        if (
            result.status == "failed"
            and not arguments.continue_on_error
        ):
            write_log("")
            write_log(
                "Pipeline stopped because the stage "
                f"'{stage.name}' failed."
            )
            break

    total_duration = (
        time.perf_counter()
        - overall_start
    )

    pipeline_completed_at = timestamp()

    write_summary(
        results,
        pipeline_started_at=(
            pipeline_started_at
        ),
        pipeline_completed_at=(
            pipeline_completed_at
        ),
        total_duration_seconds=(
            total_duration
        ),
        command_line=sys.argv,
    )

    failed_results = [
        result
        for result in results
        if result.status == "failed"
    ]

    write_log("")
    write_log("=" * 80)
    write_log("PIPELINE EXECUTION SUMMARY")
    write_log("=" * 80)

    for result in results:
        write_log(
            f"{result.name:<40} "
            f"{result.status.upper():<10} "
            f"{result.duration_seconds:>10.2f}s"
        )

    write_log("")
    write_log(
        f"Total duration: "
        f"{total_duration:.2f} seconds"
    )
    write_log(
        f"Log: {PIPELINE_LOG_PATH}"
    )
    write_log(
        f"Summary: {PIPELINE_SUMMARY_PATH}"
    )

    if failed_results:
        write_log("")
        write_log(
            f"PIPELINE FAILED: "
            f"{len(failed_results)} "
            "stage(s) failed."
        )
        return 1

    write_log("")
    write_log(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nPipeline cancelled by the user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except PipelineError as exc:
        print(
            f"Pipeline configuration error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
