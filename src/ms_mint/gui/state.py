"""Reactive state management for ms-mint GUI.

This module provides the MintState class which wraps a Mint instance
and provides reactive state variables for the Solara GUI.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import solara

if TYPE_CHECKING:
    from ms_mint.Mint import Mint as MintType

def get_session_file() -> Path:
    """Get session file path in current working directory."""
    return Path.cwd() / ".ms_mint_session.json"


class MintState:
    """Central state store for the Mint GUI application.

    This class wraps a Mint instance and provides reactive state variables
    that can be used with Solara components. It uses composition rather than
    inheritance to keep GUI state separate from business logic.

    Attributes:
        ms_files: Reactive list of loaded MS file paths.
        targets: Reactive DataFrame of target compounds.
        results: Reactive DataFrame of processing results.
        progress: Reactive float for progress bar (0-100).
        status: Reactive string for processing status.
        messages: Reactive list of log messages.
        nthreads: Reactive int for number of processing threads.
        rt_margin: Reactive float for retention time margin.
        mode: Reactive string for processing mode.
    """

    def __init__(self, mint: Optional[MintType] = None) -> None:
        """Initialize MintState with optional Mint instance.

        Args:
            mint: Optional Mint instance to wrap. If None, creates a new one.
        """
        # Import here to avoid circular imports
        from ms_mint.Mint import Mint

        # Reactive state variables
        self.ms_files: solara.Reactive[list[str]] = solara.reactive([])
        self.targets: solara.Reactive[pd.DataFrame] = solara.reactive(pd.DataFrame())
        self.results: solara.Reactive[pd.DataFrame] = solara.reactive(pd.DataFrame())
        self.progress: solara.Reactive[float] = solara.reactive(0.0)
        self.status: solara.Reactive[str] = solara.reactive("waiting")
        self.messages: solara.Reactive[list[str]] = solara.reactive([])

        # Run parameters
        self.nthreads: solara.Reactive[int] = solara.reactive(4)
        self.rt_margin: solara.Reactive[float] = solara.reactive(0.5)
        self.mode: solara.Reactive[str] = solara.reactive("standard")

        # Display settings
        self.rt_unit: solara.Reactive[str] = solara.reactive("seconds")  # "seconds" or "minutes"

        # Target filtering
        self.inactive_targets: solara.Reactive[list[str]] = solara.reactive([])

        # Create or use provided Mint instance
        if mint is None:
            self._mint = Mint(progress_callback=self._progress_callback)
        else:
            self._mint = mint
            self._mint.progress_callback = self._progress_callback

    def _progress_callback(self, value: float) -> None:
        """Callback for progress updates from Mint.

        Args:
            value: Progress value (0-100).
        """
        self.progress.value = value

    def _sync_from_mint(self) -> None:
        """Synchronize reactive state from underlying Mint instance."""
        self.ms_files.value = list(self._mint.ms_files)
        if self._mint.targets is not None and len(self._mint.targets) > 0:
            self.targets.value = self._mint.targets.reset_index()
        else:
            self.targets.value = pd.DataFrame()
        if self._mint.results is not None and len(self._mint.results) > 0:
            self.results.value = self._mint.results.copy()
        else:
            self.results.value = pd.DataFrame()

    def load_ms_files(self, files: list[str]) -> None:
        """Load MS files into state and underlying Mint instance.

        Args:
            files: List of file paths to load.
        """
        self._mint.ms_files = files
        self.ms_files.value = list(self._mint.ms_files)
        # Set working directory to the directory of the first MS file
        if files:
            self._mint.wdir = Path(files[0]).parent
        self.add_message(f"{len(files)} MS file(s) loaded.")

    def clear_ms_files(self) -> None:
        """Clear all loaded MS files."""
        self._mint.clear_ms_files()
        self.ms_files.value = []
        self.add_message("MS files cleared.")

    def load_targets_from_bytes(self, content: bytes, filename: str) -> None:
        """Load targets from uploaded file bytes.

        Args:
            content: File content as bytes.
            filename: Original filename for format detection.
        """
        import pandas as pd

        # Determine format from filename
        buffer = io.BytesIO(content)
        if filename.endswith(".xlsx"):
            df = pd.read_excel(buffer)
        else:
            df = pd.read_csv(buffer)

        # Set targets using DataFrame
        self._mint.targets = df
        self._sync_targets()
        self.add_message(f"{len(self._mint.targets)} targets loaded from {filename}.")

    def load_targets_from_file(self, filepath: str) -> None:
        """Load targets from a file path.

        Args:
            filepath: Path to target file (CSV or XLSX).
        """
        self._mint.load_targets(filepath)
        self._targets_file = filepath  # Track for session persistence
        self._sync_targets()
        self.add_message(f"{len(self._mint.targets)} targets loaded.")

    def _sync_targets(self) -> None:
        """Sync targets from Mint to reactive state."""
        if self._mint.targets is not None and len(self._mint.targets) > 0:
            # Use copy() to ensure a new DataFrame object is created for reactivity
            self.targets.value = self._mint.targets.reset_index().copy()
        else:
            self.targets.value = pd.DataFrame()

    def clear_targets(self) -> None:
        """Clear all loaded targets."""
        self._mint.clear_targets()
        self.targets.value = pd.DataFrame()
        self.add_message("Targets cleared.")

    def reorder_targets(self, new_order: list) -> None:
        """Reorder targets by the given peak label order.

        Args:
            new_order: List of peak labels in the desired order.
        """
        if self._mint.targets is None:
            return

        # Get current index and convert new_order to match its dtype
        current_index = self._mint.targets.index

        # Convert new_order items to match the index dtype
        if current_index.dtype == "object":
            # Index is strings, convert new_order to strings
            new_order = [str(x) for x in new_order]
        elif current_index.dtype == "int64":
            # Index is integers, convert new_order to integers
            new_order = [int(x) for x in new_order]

        # Reorder the targets DataFrame by the new order
        self._mint.targets = self._mint.targets.loc[new_order]
        self._sync_targets()
        self.add_message(f"Targets reordered ({len(new_order)} targets)")

    def set_inactive_targets(self, inactive: list[str]) -> None:
        """Set which targets are inactive (not used in plots/processing).

        Args:
            inactive: List of peak labels to mark as inactive.
        """
        self.inactive_targets.set(inactive)
        n_active = len(self._mint.peak_labels) - len(inactive) if self._mint.targets is not None else 0
        self.add_message(f"Target activation updated: {n_active} active, {len(inactive)} inactive")

    @property
    def active_peak_labels(self) -> list[str]:
        """Get list of active (non-inactive) peak labels."""
        if self._mint.targets is None:
            return []
        all_labels = list(self._mint.peak_labels)
        inactive = set(self.inactive_targets.value)
        return [label for label in all_labels if label not in inactive]

    def run(self) -> None:
        """Execute Mint processing with current parameters."""
        self.status.value = "running"
        self.progress.value = 0
        self.add_message("Processing started...")

        try:
            self._mint.run(
                nthreads=self.nthreads.value,
                rt_margin=self.rt_margin.value,
                mode=self.mode.value,
            )
            self._sync_results()
            self.status.value = "done"
            self.add_message(f"Processing complete. {len(self.results.value)} results.")
        except Exception as e:
            self.status.value = "error"
            self.add_message(f"Error: {e}")
            raise

    def _sync_results(self) -> None:
        """Sync results from Mint to reactive state."""
        if self._mint.results is not None and len(self._mint.results) > 0:
            self.results.value = self._mint.results.copy()
        else:
            self.results.value = pd.DataFrame()

    def export(self, filepath: str) -> None:
        """Export results to file.

        Args:
            filepath: Output file path. Format determined by extension.
        """
        self._mint.export(filepath)
        self.add_message(f"Results exported to {filepath}.")

    def add_message(self, text: str) -> None:
        """Add a message to the log (prepends for newest-first order).

        Args:
            text: Message text to add.
        """
        self.messages.value = [text] + self.messages.value

    def clear_messages(self) -> None:
        """Clear all messages from the log."""
        self.messages.value = []

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._mint.reset()
        self.ms_files.value = []
        self.targets.value = pd.DataFrame()
        self.results.value = pd.DataFrame()
        self.progress.value = 0.0
        self.status.value = "waiting"
        self.messages.value = []

    def save_session(self, filepath: Optional[Path] = None) -> None:
        """Save current session state to a JSON file.

        Args:
            filepath: Path to save session. Defaults to .ms_mint_session.json in cwd
        """
        filepath = Path(filepath) if filepath else get_session_file()
        session_data = {
            "ms_files": self.ms_files.value,
            "targets_file": getattr(self, "_targets_file", None),
            "nthreads": self.nthreads.value,
            "rt_margin": self.rt_margin.value,
            "mode": self.mode.value,
            "rt_unit": self.rt_unit.value,
        }

        # Save targets to parquet if no file path available
        if len(self.targets.value) > 0 and not getattr(self, "_targets_file", None):
            targets_file = str(filepath.parent / ".ms_mint_targets.parquet")
            self.targets.value.to_parquet(targets_file)
            session_data["targets_parquet"] = targets_file

        # Save results to parquet if available
        if len(self.results.value) > 0:
            results_file = str(filepath.with_suffix(".parquet"))
            self.results.value.to_parquet(results_file)
            session_data["results_file"] = results_file

        # Save metadata to parquet if available
        if self._mint.meta is not None and len(self._mint.meta) > 0:
            meta_file = str(filepath.parent / ".ms_mint_metadata.parquet")
            self._mint.meta.to_parquet(meta_file)
            session_data["metadata_file"] = meta_file

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        self.add_message(f"Session saved to {filepath}")

    def load_session(self, filepath: Optional[Path] = None) -> bool:
        """Load session state from a JSON file.

        Args:
            filepath: Path to load session from. Defaults to .ms_mint_session.json in cwd

        Returns:
            True if session was loaded successfully, False otherwise.
        """
        filepath = Path(filepath) if filepath else get_session_file()
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                session_data = json.load(f)

            # Restore MS files
            ms_files = session_data.get("ms_files", [])
            if ms_files:
                # Filter to only existing files
                existing_files = [f for f in ms_files if Path(f).exists()]
                if existing_files:
                    self.load_ms_files(existing_files)

            # Restore targets - try file path first, then parquet backup
            targets_file = session_data.get("targets_file")
            targets_parquet = session_data.get("targets_parquet")
            if targets_file and Path(targets_file).exists():
                self.load_targets_from_file(targets_file)
                self._targets_file = targets_file
            elif targets_parquet and Path(targets_parquet).exists():
                targets_df = pd.read_parquet(targets_parquet)
                self._mint.targets = targets_df
                self._sync_targets()
                self.add_message(f"{len(targets_df)} targets restored")

            # Restore results from parquet
            results_file = session_data.get("results_file")
            if results_file and Path(results_file).exists():
                self.results.value = pd.read_parquet(results_file)
                self._mint._results = self.results.value.copy()
                self.add_message(f"{len(self.results.value)} results restored")

            # Restore metadata from parquet
            metadata_file = session_data.get("metadata_file")
            if metadata_file and Path(metadata_file).exists():
                self._mint.meta = pd.read_parquet(metadata_file)
                self.add_message(f"Metadata restored ({len(self._mint.meta)} rows)")

            # Restore parameters
            if "nthreads" in session_data:
                self.nthreads.value = session_data["nthreads"]
            if "rt_margin" in session_data:
                self.rt_margin.value = session_data["rt_margin"]
            if "mode" in session_data:
                self.mode.value = session_data["mode"]
            if "rt_unit" in session_data:
                self.rt_unit.value = session_data["rt_unit"]

            self.add_message("Session restored")
            return True
        except Exception as e:
            self.add_message(f"Failed to load session: {e}")
            return False


# Global state instance for use across components
_state: Optional[MintState] = None


def get_state(auto_load_session: bool = True) -> MintState:
    """Get or create the global MintState instance.

    Args:
        auto_load_session: If True, automatically load previous session on first access.

    Returns:
        The global MintState instance.
    """
    global _state
    if _state is None:
        _state = MintState()
        if auto_load_session:
            _state.load_session()
    return _state


def reset_state() -> None:
    """Reset the global state instance."""
    global _state
    _state = None
