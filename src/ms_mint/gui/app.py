"""Main MintGui Solara component.

This module provides the main entry point for the ms-mint GUI application.
"""

from __future__ import annotations

import solara

from .state import MintState, get_state
from .components.file_selector import MSFileSelector
from .components.target_loader import TargetLoader
from .components.run_panel import RunPanel
from .components.message_log import MessageLog
from .components.results_panel import ResultsPanel
from .components.visualization import VisualizationPanel
from .components.optimization_panel import OptimizationPanel
from .components.metadata_panel import MetadataPanel
from .components.crosstab_panel import CrosstabPanel


@solara.component
def MintGui():
    """Main ms-mint GUI application component.

    This component provides the complete interactive interface for ms-mint,
    including file loading, processing controls, results display, and
    visualizations.

    Example:
        ```python
        from ms_mint.gui import MintGui

        # In Jupyter notebook
        MintGui()

        # Or run as standalone app:
        # solara run ms_mint.gui.app:MintGui
        ```
    """
    state = get_state()

    # Computed values
    can_run = len(state.ms_files.value) > 0 and len(state.targets.value) > 0
    can_export = len(state.results.value) > 0

    # Callbacks
    def on_files_loaded(files: list[str]) -> None:
        state.load_ms_files(files)

    def on_targets_loaded(content: bytes, filename: str) -> None:
        state.load_targets_from_bytes(content, filename)

    def on_clear_targets() -> None:
        state.clear_targets()

    def on_run() -> None:
        state.run()

    def on_export(filename: str) -> None:
        state.export(filename)

    def on_clear_messages() -> None:
        state.clear_messages()

    def on_targets_updated() -> None:
        """Sync targets after optimization."""
        state._sync_targets()

    def on_targets_reordered(new_order: list[str]) -> None:
        """Reorder targets by the given peak label order."""
        state.reorder_targets(new_order)

    def on_targets_activation_changed(inactive: list[str]) -> None:
        """Update target activation status."""
        state.set_inactive_targets(inactive)

    def on_save_targets() -> None:
        """Save targets to file."""
        state.save_targets()

    def on_metadata_updated() -> None:
        """Refresh after metadata update."""
        pass  # Metadata is stored in mint instance

    def on_save_session() -> None:
        state.save_session()

    def on_load_session() -> None:
        state.load_session()

    # Active tab
    tab_index = solara.use_reactive(0)

    # Working directory input
    wdir_input = solara.use_reactive(str(state._mint.wdir))

    def update_wdir(value: str):
        from pathlib import Path
        path = Path(value)
        if path.is_dir():
            state._mint.wdir = path
            wdir_input.set(str(path))

    # Sidebar for settings
    with solara.Sidebar():
        with solara.Card("Settings", margin=0):
            solara.Markdown("### Working Directory")
            solara.InputText(
                label="wdir",
                value=wdir_input,
                on_value=update_wdir,
            )
            solara.Markdown("---")
            solara.Markdown("### Display")
            solara.Select(
                label="RT Unit",
                value=state.rt_unit.value,
                values=["seconds", "minutes"],
                on_value=state.rt_unit.set,
            )
            solara.Markdown("---")
            solara.Markdown("### Performance")
            solara.SliderInt(
                label="Threads",
                value=state.nthreads,
                min=1,
                max=16,
            )
            solara.Markdown("---")
            solara.Markdown("### Session")
            solara.Button(
                "Load Session",
                on_click=on_load_session,
                color="primary",
                outlined=True,
                style={"width": "100%"},
            )
            solara.Button(
                "Save Session",
                on_click=on_save_session,
                color="secondary",
                outlined=True,
                style={"width": "100%", "margin-top": "8px"},
            )

    with solara.Column(style={"padding": "16px"}):
        solara.Title("ms-mint")

        # All content in tabs
        with solara.lab.Tabs(value=tab_index):
            with solara.lab.Tab("MS Files"):
                MSFileSelector(
                    ms_files=state.ms_files,
                    on_files_loaded=on_files_loaded,
                )
                # Metadata section under MS Files
                solara.Markdown("---")
                MetadataPanel(
                    mint=state._mint,
                    ms_files=state.ms_files,
                    on_metadata_updated=on_metadata_updated,
                )

            with solara.lab.Tab("Targets"):
                TargetLoader(
                    targets=state.targets,
                    on_targets_loaded=on_targets_loaded,
                    on_targets_reordered=on_targets_reordered,
                    on_targets_activation_changed=on_targets_activation_changed,
                    on_save_targets=on_save_targets,
                    on_clear=on_clear_targets,
                )
                OptimizationPanel(
                    mint=state._mint,
                    ms_files=state.ms_files,
                    targets=state.targets,
                    on_targets_updated=on_targets_updated,
                    rt_unit=state.rt_unit,
                    nthreads=state.nthreads,
                )

            with solara.lab.Tab("Processing"):
                with solara.Row():
                    with solara.Column(style={"flex": "2"}):
                        RunPanel(
                            status=state.status,
                            progress=state.progress,
                            nthreads=state.nthreads,
                            rt_margin=state.rt_margin,
                            mode=state.mode,
                            can_run=can_run,
                            can_export=can_export,
                            on_run=on_run,
                            on_export=on_export,
                        )
                    with solara.Column(style={"flex": "1"}):
                        MessageLog(
                            messages=state.messages,
                            on_clear=on_clear_messages,
                        )

            with solara.lab.Tab("Results"):
                ResultsPanel(
                    results=state.results,
                )
                solara.Markdown("---")
                CrosstabPanel(
                    mint=state._mint,
                    results=state.results,
                    inactive_targets=state.inactive_targets,
                )

            with solara.lab.Tab("Visualization"):
                VisualizationPanel(
                    mint=state._mint,
                    results=state.results,
                    targets=state.targets,
                    rt_unit=state.rt_unit,
                    nthreads=state.nthreads,
                    inactive_targets=state.inactive_targets,
                )


# For standalone deployment: solara run ms_mint.gui.app:Page
Page = MintGui
