"""Target list optimization tools for retention time refinement."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Mint import Mint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .Chromatogram import Chromatogram
from .io import ms_file_to_df
from .processing import extract_chromatogram_from_ms1


class TargetOptimizer:
    """Optimizer for MS-MINT target lists.

    This class provides methods to optimize retention time parameters
    in target lists based on actual data from MS files.

    Attributes:
        mint: Mint instance to optimize.
        results: Results of the most recent optimization.
    """

    def __init__(self, mint: Mint | None = None) -> None:
        """Initialize a TargetOptimizer instance.

        Args:
            mint: Mint instance to optimize.
        """
        self.mint = mint
        self.reset()

    def reset(self) -> TargetOptimizer:
        """Reset the optimizer results.

        Returns:
            Self for method chaining.
        """
        self.results: pd.DataFrame | None = None
        return self

    def _optimize_targets(
        self,
        process_target: Callable[[str, pd.Series, pd.DataFrame, pd.DataFrame], tuple[Any, ...] | None],
        plot_target: Callable[[int, int, int, str, float, Any], None] | None,
        fns: list[str | Path] | None = None,
        targets: pd.DataFrame | None = None,
        peak_labels: list[str] | None = None,
        plot: bool = False,
        height: int = 3,
        aspect: int = 2,
        col_wrap: int = 3,
        max_plots: int = 1000,
        desc: str = "Processing targets",
    ) -> tuple[Mint, Figure] | Mint:
        """Template method for target optimization workflows.

        Handles common setup, iteration, and finalization logic. Delegates
        target-specific processing and plotting to the provided callables.

        Args:
            process_target: Callable that processes a single target. Receives
                (peak_label, row, _targets, ms1) and returns data to update
                _targets, or None to skip.
            plot_target: Callable for plotting a single target. Receives
                (plot_index, n_rows, col_wrap, peak_label, mz, process_result).
            fns: List of filenames to use. If None, uses all files in mint.
            targets: Target list to optimize. If None, uses mint.targets.
            peak_labels: Subset of peak_labels to process. If None, processes all.
            plot: Whether to generate plots.
            height: Height of each subplot in inches.
            aspect: Width-to-height ratio of each subplot.
            col_wrap: Maximum number of columns in the plot.
            max_plots: Maximum number of subplots to generate.
            desc: Description for the progress bar.

        Returns:
            If plot=True, returns a tuple of (mint instance, matplotlib figure).
            If plot=False, returns only the mint instance.
        """
        if targets is None:
            targets = self.mint.targets.reset_index()

        if fns is None:
            fns = self.mint.ms_files

        if peak_labels is None:
            peak_labels = targets.peak_label.values

        _targets = targets.set_index("peak_label").copy()

        ms1 = pd.concat(
            [ms_file_to_df(fn) for fn in self.mint.tqdm(fns, desc="Reading files")]
        ).sort_values(["scan_time", "mz"])

        fig = None
        if plot:
            n_plots = min(len(peak_labels), max_plots)
            n_rows = int(np.ceil(n_plots / col_wrap))
            fig = plt.figure(figsize=(col_wrap * height * aspect, n_rows * height))

        plot_index = 0
        n_rows = int(np.ceil(min(len(peak_labels), max_plots) / col_wrap)) if plot else 0

        for peak_label, row in self.mint.tqdm(
            _targets.iterrows(), total=len(targets), desc=desc
        ):
            if peak_label not in peak_labels:
                logging.warning(f"{peak_label} not in {peak_labels}")
                continue

            result = process_target(peak_label, row, _targets, ms1)

            if result is not None and plot and plot_index < max_plots:
                plot_index += 1
                if plot_target is not None:
                    plot_target(plot_index, n_rows, col_wrap, peak_label, row.mz_mean, result)

        self.results = _targets.reset_index()

        if self.mint is not None:
            self.mint.targets = self.results

        if plot:
            plt.tight_layout()
            return self.mint, fig
        else:
            return self.mint

    def rt_min_max(
        self,
        fns: list[str | Path] | None = None,
        targets: pd.DataFrame | None = None,
        peak_labels: list[str] | None = None,
        minimum_intensity: float = 1e4,
        plot: bool = False,
        sigma: float = 20,
        filters: list[Any] | None = None,
        post_opt: bool = False,
        post_opt_kwargs: dict[str, Any] | None = None,
        rel_height: float = 0.9,
        height: int = 3,
        aspect: int = 2,
        col_wrap: int = 3,
        **kwargs,
    ) -> tuple[Mint, Figure] | Mint:
        """Optimize rt_min and rt_max values based on expected retention times.

        For this optimization all rt values in the target list must be present.
        This method analyzes chromatograms to find peaks around expected retention
        times and sets optimal rt_min and rt_max values.

        Args:
            fns: List of filenames to use for optimization. If None, uses all files in mint.
            targets: Target list to optimize. If None, uses mint.targets.
            peak_labels: Subset of peak_labels to optimize. If None, optimizes all targets.
            minimum_intensity: Minimum intensity required, otherwise skip target.
            plot: Whether to plot optimizations (up to 1000 plots).
            sigma: Sigma value for peak selection (Gaussian weighting parameter).
            filters: Filter instances to apply in respective order.
            post_opt: Whether to optimize retention times after peak selection.
            post_opt_kwargs: Parameters for post-optimization.
            rel_height: Relative height for peak width determination.
            height: Height of each subplot in inches.
            aspect: Width-to-height ratio of each subplot.
            col_wrap: Maximum number of columns in the plot.
            **kwargs: Additional parameters passed to find_peaks method.

        Returns:
            If plot=True, returns a tuple of (mint instance, matplotlib figure).
            If plot=False, returns only the mint instance.
        """
        def process_target(
            peak_label: str, row: pd.Series, _targets: pd.DataFrame, ms1: pd.DataFrame
        ) -> Chromatogram | None:
            mz = row.mz_mean
            rt = row.rt

            _slice = extract_chromatogram_from_ms1(ms1, mz).groupby("scan_time").sum()
            chrom = Chromatogram(_slice.index, _slice.values, expected_rt=rt, filters=filters)

            if chrom.x.max() < minimum_intensity:
                logging.warning(
                    f"Peak intensity for {peak_label} below threshold ({minimum_intensity})"
                )
                return None

            chrom.apply_filters()
            chrom.find_peaks(rel_height=rel_height, **kwargs)
            chrom.select_peak_with_gaussian_weight(rt, sigma)

            if post_opt:
                _post_opt_kwargs = post_opt_kwargs or {}
                chrom.optimise_peak_times_with_diff(**_post_opt_kwargs)

            if chrom.selected_peak_ndxs is None or len(chrom.selected_peak_ndxs) == 0:
                logging.warning(f"No peaks detected for {peak_label}")
                return None

            ndx = chrom.selected_peak_ndxs[0]
            rt_min = chrom.peaks.at[ndx, "rt_min"]
            rt_max = chrom.peaks.at[ndx, "rt_max"]

            _targets.loc[peak_label, ["rt_min", "rt_max"]] = rt_min, rt_max
            return chrom

        def plot_target(
            plot_index: int, n_rows: int, col_wrap: int,
            peak_label: str, mz: float, chrom: Chromatogram
        ) -> None:
            plt.subplot(n_rows, col_wrap, plot_index)
            chrom.plot()
            plt.gca().get_legend().remove()
            plt.title(f"{peak_label}\nm/z={mz:.3f}")

        return self._optimize_targets(
            process_target=process_target,
            plot_target=plot_target,
            fns=fns,
            targets=targets,
            peak_labels=peak_labels,
            plot=plot,
            height=height,
            aspect=aspect,
            col_wrap=col_wrap,
            max_plots=1000,
            desc="Optimizing targets",
        )

    def detect_largest_peak_rt(
        self,
        fns: list[str | Path] | None = None,
        targets: pd.DataFrame | None = None,
        peak_labels: list[str] | None = None,
        minimum_intensity: float = 1e4,
        plot: bool = False,
        height: int = 3,
        aspect: int = 2,
        col_wrap: int = 3,
        **kwargs,
    ) -> tuple[Mint, Figure] | Mint:
        """Detect the largest peak and set the RT value (not RT_min and RT_max).

        Uses a simple maximum intensity approach rather than complex peak detection
        to find the retention time of the most intense peak for each target.

        Args:
            fns: List of filenames to use for peak detection. If None, uses all files in mint.
            targets: Target list to update. If None, uses mint.targets.
            peak_labels: Subset of peak_labels to update. If None, updates all targets.
            minimum_intensity: Minimum intensity required, otherwise skip target.
            plot: Whether to plot results (up to 100 plots).
            height: Height of each subplot in inches.
            aspect: Width-to-height ratio of each subplot.
            col_wrap: Maximum number of columns in the plot.
            **kwargs: Additional parameters (not used but accepted for compatibility).

        Returns:
            If plot=True, returns a tuple of (mint instance, matplotlib figure).
            If plot=False, returns only the mint instance.
        """
        def process_target(
            peak_label: str, row: pd.Series, _targets: pd.DataFrame, ms1: pd.DataFrame
        ) -> tuple[pd.Series, float] | None:
            mz = row.mz_mean
            mz_width = row.mz_width if "mz_width" in row.index else 10

            try:
                _slice = extract_chromatogram_from_ms1(ms1, mz, mz_width)
                if len(_slice) == 0:
                    logging.warning(f"No data points found for {peak_label}")
                    return None

                chrom_data = _slice.groupby("scan_time").sum()

                if chrom_data.values.max() < minimum_intensity:
                    logging.warning(
                        f"Peak intensity for {peak_label} below threshold ({minimum_intensity})"
                    )
                    return None

                max_intensity_idx = chrom_data.values.argmax()
                new_rt = chrom_data.index[max_intensity_idx]

                _targets.loc[peak_label, "rt"] = new_rt
                return (chrom_data, new_rt)

            except Exception as e:
                logging.error(f"Error processing {peak_label}: {str(e)}")
                return None

        def plot_target(
            plot_index: int, n_rows: int, col_wrap: int,
            peak_label: str, mz: float, result: tuple[pd.Series, float]
        ) -> None:
            chrom_data, new_rt = result
            plt.subplot(n_rows, col_wrap, plot_index)
            plt.plot(chrom_data.index, chrom_data.values)
            plt.axvline(new_rt, color="red", linestyle="--")
            plt.title(f"{peak_label}\nm/z={mz:.3f}\nRT={new_rt:.1f}")

        return self._optimize_targets(
            process_target=process_target,
            plot_target=plot_target,
            fns=fns,
            targets=targets,
            peak_labels=peak_labels,
            plot=plot,
            height=height,
            aspect=aspect,
            col_wrap=col_wrap,
            max_plots=100,
            desc="Detecting largest peaks",
        )
