from src.config import (
    APPROVED_RAW_DATASET_CATEGORIES,
    CV_FOLDS,
    GENERATED_DIRECTORIES,
    INITIAL_HWI_DIMENSION_WEIGHTS,
    INITIAL_HWI_THRESHOLDS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATASET_CATEGORY_PATHS,
    RAW_DATA_DIR,
    SUPPORTED_DATASET_EXTENSIONS,
    TEST_SIZE,
)


def test_project_root_contains_expected_directories() -> None:
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
    assert (PROJECT_ROOT / "data").exists()


def test_approved_raw_categories_match_scope() -> None:
    assert APPROVED_RAW_DATASET_CATEGORIES == (
        "awareness",
        "ai_emails",
        "emails",
        "phishing_urls",
    )


def test_removed_categories_are_not_approved() -> None:
    assert "webpage" not in APPROVED_RAW_DATASET_CATEGORIES
    assert "breaches" not in APPROVED_RAW_DATASET_CATEGORIES


def test_raw_dataset_category_paths_match_categories() -> None:
    assert tuple(RAW_DATASET_CATEGORY_PATHS.keys()) == APPROVED_RAW_DATASET_CATEGORIES
    for category, path in RAW_DATASET_CATEGORY_PATHS.items():
        assert path == RAW_DATA_DIR / category


def test_generated_directories_exist() -> None:
    for directory in GENERATED_DIRECTORIES:
        assert directory.exists()
        assert directory.is_dir()


def test_generated_directories_do_not_recreate_removed_raw_categories() -> None:
    assert not (RAW_DATA_DIR / "webpage").exists()
    assert not (RAW_DATA_DIR / "breaches").exists()


def test_supported_extensions_cover_required_types() -> None:
    assert {".csv", ".xlsx", ".xls", ".json", ".parquet"}.issubset(
        set(SUPPORTED_DATASET_EXTENSIONS)
    )


def test_reproducibility_defaults_are_stable() -> None:
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.20
    assert CV_FOLDS == 5


def test_initial_hwi_configuration_is_complete() -> None:
    assert INITIAL_HWI_THRESHOLDS["low_max"] == 30
    assert INITIAL_HWI_THRESHOLDS["medium_max"] == 70
    assert abs(sum(INITIAL_HWI_DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9
