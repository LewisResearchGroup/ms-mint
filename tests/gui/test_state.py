"""Tests for MintState class - TDD approach."""

import pytest
import pandas as pd
from pathlib import Path

from ms_mint.gui.state import MintState


class TestMintStateInitialization:
    """Test MintState initialization."""

    def test_initial_state_defaults(self):
        """MintState should initialize with empty/default values."""
        state = MintState()

        assert state.ms_files.value == []
        assert len(state.targets.value) == 0
        assert len(state.results.value) == 0
        assert state.progress.value == 0.0
        assert state.status.value == "waiting"
        assert state.messages.value == []

    def test_initial_run_parameters(self):
        """MintState should have default run parameters."""
        state = MintState()

        assert state.nthreads.value == 4
        assert state.rt_margin.value == 0.5
        assert state.mode.value == "standard"

    def test_mint_instance_created(self):
        """MintState should create a Mint instance internally."""
        state = MintState()

        assert state._mint is not None
        assert hasattr(state._mint, "run")
        assert hasattr(state._mint, "results")


class TestMintStateFileLoading:
    """Test file loading functionality."""

    def test_load_ms_files(self, test_ms_files: list[str]):
        """Loading MS files should update state and add message."""
        state = MintState()
        state.load_ms_files(test_ms_files)

        assert len(state.ms_files.value) == len(test_ms_files)
        assert len(state.messages.value) > 0
        assert "loaded" in state.messages.value[0].lower()

    def test_load_ms_files_updates_mint(self, test_ms_files: list[str]):
        """Loading MS files should update underlying Mint instance."""
        state = MintState()
        state.load_ms_files(test_ms_files)

        assert state._mint.n_files == len(test_ms_files)

    def test_clear_ms_files(self, test_mzml_file: Path):
        """Clearing MS files should reset file list."""
        state = MintState()
        state.load_ms_files([str(test_mzml_file)])  # Use single file to avoid duplicate label issue
        state.clear_ms_files()

        assert state.ms_files.value == []
        assert state._mint.n_files == 0


class TestMintStateTargetLoading:
    """Test target loading functionality."""

    def test_load_targets_from_bytes(self, test_targets_csv_content: bytes):
        """Loading targets from bytes should update state."""
        state = MintState()
        state.load_targets_from_bytes(test_targets_csv_content, "targets.csv")

        assert len(state.targets.value) > 0
        assert len(state.messages.value) > 0
        assert "targets" in state.messages.value[0].lower()

    def test_load_targets_from_file(self, test_targets_csv: Path):
        """Loading targets from file path should update state."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))

        assert len(state.targets.value) > 0

    def test_clear_targets(self, test_targets_csv: Path):
        """Clearing targets should reset target list."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))
        state.clear_targets()

        assert len(state.targets.value) == 0


class TestMintStateProcessing:
    """Test processing functionality."""

    def test_progress_callback_updates_state(self):
        """Progress callback should update progress value."""
        state = MintState()
        state._progress_callback(50.0)

        assert state.progress.value == 50.0

    def test_run_updates_status(self, test_ms_files: list[str], test_targets_csv: Path):
        """Running processing should update status."""
        state = MintState()
        state.load_ms_files([test_ms_files[0]])  # Use single file for speed
        state.load_targets_from_file(str(test_targets_csv))

        state.run()

        assert state.status.value == "done"
        assert len(state.results.value) > 0

    def test_run_with_parameters(self, test_ms_files: list[str], test_targets_csv: Path):
        """Run should use configured parameters."""
        state = MintState()
        state.load_ms_files([test_ms_files[0]])
        state.load_targets_from_file(str(test_targets_csv))

        state.nthreads.value = 1
        state.rt_margin.value = 1.0
        state.mode.value = "express"

        state.run()

        assert state.status.value == "done"


class TestMintStateMessages:
    """Test message handling."""

    def test_add_message(self):
        """Adding message should prepend to message list."""
        state = MintState()
        state.add_message("First message")
        state.add_message("Second message")

        assert len(state.messages.value) == 2
        assert state.messages.value[0] == "Second message"
        assert state.messages.value[1] == "First message"

    def test_clear_messages(self):
        """Clearing messages should empty the list."""
        state = MintState()
        state.add_message("Test message")
        state.clear_messages()

        assert state.messages.value == []


class TestMintStateExport:
    """Test export functionality."""

    def test_export_results(self, test_ms_files: list[str], test_targets_csv: Path, tmp_path: Path):
        """Exporting results should create file."""
        state = MintState()
        state.load_ms_files([test_ms_files[0]])
        state.load_targets_from_file(str(test_targets_csv))
        state.run()

        output_file = tmp_path / "results.xlsx"
        state.export(str(output_file))

        assert output_file.exists()

    def test_export_formats(self, test_ms_files: list[str], test_targets_csv: Path, tmp_path: Path):
        """Export should support multiple formats."""
        state = MintState()
        state.load_ms_files([test_ms_files[0]])
        state.load_targets_from_file(str(test_targets_csv))
        state.run()

        for ext in [".xlsx", ".csv", ".parquet"]:
            output_file = tmp_path / f"results{ext}"
            state.export(str(output_file))
            assert output_file.exists(), f"Export to {ext} failed"


class TestMintStateReset:
    """Test reset functionality."""

    def test_reset_clears_all(self, test_ms_files: list[str], test_targets_csv: Path):
        """Reset should clear all state."""
        state = MintState()
        state.load_ms_files(test_ms_files)
        state.load_targets_from_file(str(test_targets_csv))
        state.add_message("Test")

        state.reset()

        assert state.ms_files.value == []
        assert len(state.targets.value) == 0
        assert len(state.results.value) == 0
        assert state.progress.value == 0.0
        assert state.status.value == "waiting"
        assert state.messages.value == []


class TestMintStateTargetReordering:
    """Test target reordering functionality."""

    def test_reorder_targets(self, test_targets_csv: Path):
        """Reordering targets should change the order."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))

        # Get original order
        original_order = state.targets.value["peak_label"].tolist()

        # Reverse the order
        new_order = list(reversed(original_order))
        state.reorder_targets(new_order)

        # Check that the order changed
        reordered = state.targets.value["peak_label"].tolist()
        assert reordered == new_order

    def test_reorder_targets_message(self, test_targets_csv: Path):
        """Reordering should add a message."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))
        state.clear_messages()

        original_order = state.targets.value["peak_label"].tolist()
        state.reorder_targets(original_order)

        assert len(state.messages.value) > 0
        assert "reordered" in state.messages.value[0].lower()


class TestMintStateTargetActivation:
    """Test target activation/deactivation functionality."""

    def test_initial_inactive_targets(self):
        """Initially all targets should be active (empty inactive list)."""
        state = MintState()
        assert state.inactive_targets.value == []

    def test_set_inactive_targets(self, test_targets_csv: Path):
        """Setting inactive targets should update the list."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))

        # Get first target label
        first_label = state.targets.value["peak_label"].iloc[0]
        state.set_inactive_targets([first_label])

        assert first_label in state.inactive_targets.value

    def test_active_peak_labels(self, test_targets_csv: Path):
        """active_peak_labels should exclude inactive targets."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))

        all_labels = state.targets.value["peak_label"].tolist()
        first_label = all_labels[0]

        # Deactivate first target
        state.set_inactive_targets([first_label])

        active = state.active_peak_labels
        assert first_label not in active
        assert len(active) == len(all_labels) - 1

    def test_set_inactive_targets_message(self, test_targets_csv: Path):
        """Setting inactive targets should add a message."""
        state = MintState()
        state.load_targets_from_file(str(test_targets_csv))
        state.clear_messages()

        state.set_inactive_targets(["test"])

        assert len(state.messages.value) > 0
        assert "activation" in state.messages.value[0].lower()
