"""Crosstab panel component for creating and exporting pivot tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import solara

if TYPE_CHECKING:
    from ms_mint.Mint import Mint


@solara.component
def CrosstabPanel(
    mint: "Mint",
    results: solara.Reactive[pd.DataFrame],
    inactive_targets: solara.Reactive[list[str]] = None,
):
    """Component for creating and exporting crosstabs (pivot tables).

    Args:
        mint: The Mint instance with crosstab method.
        results: Reactive DataFrame of results.
        inactive_targets: Reactive list of inactive target labels to filter out.
    """
    # Settings
    var_name = solara.use_reactive("peak_max")
    apply = solara.use_reactive("log2p1")
    scaler = solara.use_reactive("none")

    # Export settings
    export_message = solara.use_reactive("")

    # Crosstab data (computed on demand)
    crosstab_data = solara.use_reactive(None)

    def compute_crosstab():
        """Compute crosstab with current settings."""
        try:
            apply_val = apply.value if apply.value != "none" else None
            scaler_val = scaler.value if scaler.value != "none" else None

            ct = mint.crosstab(
                var_name=var_name.value,
                apply=apply_val,
                scaler=scaler_val,
            )

            # Filter out inactive targets
            if inactive_targets is not None and inactive_targets.value:
                inactive_set = set(inactive_targets.value)
                active_cols = [c for c in ct.columns if c not in inactive_set]
                ct = ct[active_cols]

            crosstab_data.set(ct)
            export_message.set("")
        except Exception as e:
            export_message.set(f"Error: {e}")
            crosstab_data.set(None)

    def export_csv():
        """Export crosstab to CSV."""
        if crosstab_data.value is None:
            export_message.set("No data to export. Click 'Generate' first.")
            return
        try:
            # Create output directory
            out_dir = mint.wdir / "analyses" / "crosstabs"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            date_str = datetime.now().strftime("%y%m%d")
            apply_str = apply.value if apply.value != "none" else "raw"
            scaler_str = scaler.value if scaler.value != "none" else "unscaled"
            filename = f"{date_str}-crosstab-{var_name.value}-{apply_str}-{scaler_str}.csv"
            filepath = out_dir / filename

            # Export
            crosstab_data.value.to_csv(filepath)
            export_message.set(f"Exported: {filepath.name}")
        except Exception as e:
            export_message.set(f"Export error: {e}")

    def export_xlsx():
        """Export crosstab to XLSX."""
        if crosstab_data.value is None:
            export_message.set("No data to export. Click 'Generate' first.")
            return
        try:
            # Create output directory
            out_dir = mint.wdir / "analyses" / "crosstabs"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            date_str = datetime.now().strftime("%y%m%d")
            apply_str = apply.value if apply.value != "none" else "raw"
            scaler_str = scaler.value if scaler.value != "none" else "unscaled"
            filename = f"{date_str}-crosstab-{var_name.value}-{apply_str}-{scaler_str}.xlsx"
            filepath = out_dir / filename

            # Export
            crosstab_data.value.to_excel(filepath)
            export_message.set(f"Exported: {filepath.name}")
        except Exception as e:
            export_message.set(f"Export error: {e}")

    if len(results.value) == 0:
        solara.Info("No results available. Run processing first.")
        return

    with solara.Card("Crosstab", margin=0):
        with solara.Column():
            # Settings row
            with solara.Row():
                solara.Select(
                    label="Variable",
                    value=var_name.value,
                    values=["peak_max", "peak_area", "peak_area_top3", "peak_mean"],
                    on_value=var_name.set,
                )
                solara.Select(
                    label="Transform",
                    value=apply.value,
                    values=["none", "log2p1", "log10p1"],
                    on_value=apply.set,
                )
                solara.Select(
                    label="Scaler",
                    value=scaler.value,
                    values=["none", "standard", "robust", "minmax"],
                    on_value=scaler.set,
                )

            # Action buttons
            with solara.Row():
                solara.Button(
                    "Generate",
                    on_click=compute_crosstab,
                    color="primary",
                )
                solara.Button(
                    "Export CSV",
                    on_click=export_csv,
                    color="secondary",
                    disabled=crosstab_data.value is None,
                )
                solara.Button(
                    "Export XLSX",
                    on_click=export_xlsx,
                    color="secondary",
                    disabled=crosstab_data.value is None,
                )

            # Message
            if export_message.value:
                if "Error" in export_message.value:
                    solara.Error(export_message.value)
                else:
                    solara.Success(export_message.value)

            # Preview
            if crosstab_data.value is not None:
                ct = crosstab_data.value
                solara.Info(f"Crosstab: {ct.shape[0]} samples x {ct.shape[1]} targets")

                # Show preview (first 10 rows, first 10 columns)
                preview = ct.iloc[:10, :10]
                solara.DataFrame(preview.reset_index())

                if ct.shape[0] > 10 or ct.shape[1] > 10:
                    solara.Text(
                        f"Showing first 10x10 of {ct.shape[0]}x{ct.shape[1]}",
                        style={"fontStyle": "italic", "fontSize": "12px"},
                    )
