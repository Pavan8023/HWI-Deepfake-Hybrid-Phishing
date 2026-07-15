from src.config import (
    CV_FOLDS,
    GENERATED_DIRECTORIES,
    INITIAL_HWI_DIMENSION_WEIGHTS,
    INITIAL_HWI_THRESHOLDS,
    PROJECT_ROOT,
    RANDOM_STATE,
    SUPPORTED_DATASET_EXTENSIONS,
    TEST_SIZE,
)


def test_project_root_contains_expected_directories() -> None:
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
    assert (PROJECT_ROOT / "data").exists()


def test_generated_directories_exist() -> None:
    for directory in GENERATED_DIRECTORIES:
        assert directory.exists()
        assert directory.is_dir()


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
