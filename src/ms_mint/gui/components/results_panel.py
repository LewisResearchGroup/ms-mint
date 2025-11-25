"""Results panel component for displaying processing results."""

from __future__ import annotations

import pandas as pd
import solara


@solara.component
def ResultsPanel(
    results: solara.Reactive[pd.DataFrame],
):
    """Component for displaying results.

    Args:
        results: Reactive DataFrame of results.
    """
    show_all_columns = solara.use_reactive(False)
    max_rows = solara.use_reactive(50)

    if len(results.value) == 0:
        solara.Info("No results yet. Load files and targets, then run processing.")
        return

    with solara.Card("Results", margin=0):
        with solara.Column():
            solara.Info(f"{len(results.value)} rows x {len(results.value.columns)} columns")

            # Display options
            with solara.Row():
                solara.Checkbox(label="Show all columns", value=show_all_columns)
                solara.SliderInt(label="Max rows", value=max_rows, min=10, max=200, step=10)

            # Results table
            if show_all_columns.value:
                preview_df = results.value.head(max_rows.value)
            else:
                # Show key columns
                display_cols = []
                priority_cols = [
                    "ms_file_label",
                    "peak_label",
                    "mz_mean",
                    "rt",
                    "rt_min",
                    "rt_max",
                    "peak_area",
                    "peak_area_top3",
                    "peak_max",
                    "peak_mean",
                    "peak_rt_of_max",
                    "peak_n_datapoints",
                ]
                for col in priority_cols:
                    if col in results.value.columns:
                        display_cols.append(col)
                preview_df = results.value[display_cols].head(max_rows.value) if display_cols else results.value.head(max_rows.value)

            # Round numeric columns for cleaner display
            preview_df = preview_df.copy()
            for col in preview_df.select_dtypes(include=["float64", "float32"]).columns:
                if col in ["mz_mean"]:
                    preview_df[col] = preview_df[col].round(4)
                elif col in ["rt", "rt_min", "rt_max", "peak_rt_of_max"]:
                    preview_df[col] = preview_df[col].round(2)
                else:
                    preview_df[col] = preview_df[col].round(0).astype("Int64")

            solara.DataFrame(preview_df, items_per_page=max_rows.value)

            if len(results.value) > max_rows.value:
                solara.Text(
                    f"Showing {max_rows.value} of {len(results.value)} rows",
                    style={"fontStyle": "italic", "fontSize": "12px"},
                )
