"""Shared fixtures for GUI tests."""

import sys
from pathlib import Path

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import (
    TEST_MZML,
    TEST_MZXML,
    TEST_TARGETS_FN_V0,
    TEST_TARGETS_FN_V2_CSV_SEC,
    RESULTS_FN,
)


@pytest.fixture
def test_mzml_file() -> Path:
    """Path to test mzML file."""
    return Path(TEST_MZML)


@pytest.fixture
def test_mzxml_file() -> Path:
    """Path to test mzXML file."""
    return Path(TEST_MZXML)


@pytest.fixture
def test_ms_files() -> list[str]:
    """List of test MS files."""
    return [TEST_MZML, TEST_MZXML]


@pytest.fixture
def test_targets_csv() -> Path:
    """Path to test targets CSV file."""
    return Path(TEST_TARGETS_FN_V2_CSV_SEC)


@pytest.fixture
def test_targets_csv_content(test_targets_csv: Path) -> bytes:
    """Content of test targets CSV file as bytes."""
    return test_targets_csv.read_bytes()


@pytest.fixture
def test_results_csv() -> Path:
    """Path to test results CSV file."""
    return Path(RESULTS_FN)
