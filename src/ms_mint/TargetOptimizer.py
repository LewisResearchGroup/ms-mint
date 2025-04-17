import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import sys
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path as P
from matplotlib.figure import Figure
from __future__ import annotations

from .io import ms_file_to_df
from .Chromatogram import Chromatogram
from .processing import get_chromatogram_from_ms_file, extract_chromatogram_from_ms1


class TargetOptimizer:
    """Optimizer for MS-MINT target lists.

    This class provides methods to optimize retention time parameters
    in target lists based on actual data from MS files.

    Attributes:
        mint: Mint instance to optimize.
        results: Results of the most recent optimization.
    """

    def __init__(self, mint: Optional["ms_mint.Mint.Mint"] = None) -> None:
        """Initialize a TargetOptimizer instance.

        Args:
            mint: Mint instance to optimize.
        """
        self.mint = mint
        self.reset()

    def reset(self) -> "TargetOptimizer":
        """Reset the optimizer results.

        Returns:
            Self for method chaining.
        """
        self.results: Optional[pd.DataFrame] = None
        return self

    def rt_min_max(
        self,
        fns: Optional[List[Union[str, P]]] = None,
        targets: Optional[pd.DataFrame] = None,
        peak_labels: Optional[List[str]] = None,
        minimum_intensity: float = 1e4,
        plot: bool = False,
        sigma: float = 20,
        filters: Optional[List[Any]] = None,
        post_opt: bool = False,
        post_opt_kwargs: Optional[Dict[str, Any]] = None,
        rel_height: float = 0.9,
        height: int = 3,
        aspect: int = 2,
        col_wrap: int = 3,
        **kwargs,
    ) -> Union[Tuple["ms_mint.Mint.Mint", Figure], "ms_mint.Mint.Mint"]:
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

        if plot:
            n_rows = int(np.ceil(len(peak_labels) / col_wrap))
            fig = plt.figure(figsize=(col_wrap * height * aspect, n_rows * height))

        i = 0
        for peak_label, row in self.mint.tqdm(
            _targets.iterrows(), total=len(targets), desc="Optimizing targets"
        ):
            if peak_label not in peak_labels:
                logging.warning(f"{peak_label} not in {peak_labels}")
                continue

            mz = row.mz_mean
            rt = row.rt

            _slice = extract_chromatogram_from_ms1(ms1, mz).groupby("scan_time").sum()

            chrom = Chromatogram(_slice.index, _slice.values, expected_rt=rt, filters=filters)

            if chrom.x.max() < minimum_intensity:
                logging.warning(
                    f"Peak intensity for {peak_label} below threshold ({minimum_intensity})"
                )
                continue

            chrom.apply_filters()
            chrom.find_peaks(rel_height=rel_height, **kwargs)
            chrom.select_peak_with_gaussian_weight(rt, sigma)

            if post_opt:
                if post_opt_kwargs is None:
                    post_opt_kwargs = {}
                chrom.optimise_peak_times_with_diff(**post_opt_kwargs)

            if chrom.selected_peak_ndxs is None or len(chrom.selected_peak_ndxs) == 0:
                logging.warning(f"No peaks detected for {peak_label}")
                continue

            ndx = chrom.selected_peak_ndxs[0]
            rt_min = chrom.peaks.at[ndx, "rt_min"]
            rt_max = chrom.peaks.at[ndx, "rt_max"]

            _targets.loc[peak_label, ["rt_min", "rt_max"]] = rt_min, rt_max

            if plot:
                i += 1

                if i <= 1000:
                    plt.subplot(n_rows, col_wrap, i)
                    chrom.plot()
                    plt.gca().get_legend().remove()
                    plt.title(f"{peak_label}\nm/z={mz:.3f}")

        self.results = _targets.reset_index()

        if self.mint is not None:
            self.mint.targets = self.results

        if plot:
            plt.tight_layout()
            return self.mint, fig
        else:
            return self.mint

    def detect_largest_peak_rt(
        self,
        fns: Optional[List[Union[str, P]]] = None,
        targets: Optional[pd.DataFrame] = None,
        peak_labels: Optional[List[str]] = None,
        minimum_intensity: float = 1e4,
        plot: bool = False,
        height: int = 3,
        aspect: int = 2,
        col_wrap: int = 3,
        **kwargs,
    ) -> Union[Tuple["ms_mint.Mint.Mint", Figure], "ms_mint.Mint.Mint"]:
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

        if plot:
            n_rows = int(np.ceil(min(len(peak_labels), 100) / col_wrap))
            fig = plt.figure(figsize=(col_wrap * height * aspect, n_rows * height))

        i = 0
        for peak_label, row in self.mint.tqdm(
            _targets.iterrows(), total=len(targets), desc="Detecting largest peaks"
        ):
            if peak_label not in peak_labels:
                logging.warning(f"{peak_label} not in {peak_labels}")
                continue

            mz = row.mz_mean
            mz_width = row.mz_width if "mz_width" in row else 0.01  # Default width if not present

            # Extract chromatogram
            try:
                _slice = extract_chromatogram_from_ms1(
                    ms1, mz, mz_width if "mz_width" in row else None
                )
                if len(_slice) == 0:
                    logging.warning(f"No data points found for {peak_label}")
                    continue

                chrom_data = _slice.groupby("scan_time").sum()

                # Simple approach: find the scan time with maximum intensity
                if chrom_data.values.max() < minimum_intensity:
                    logging.warning(
                        f"Peak intensity for {peak_label} below threshold ({minimum_intensity})"
                    )
                    continue

                # Get the retention time with the maximum intensity
                max_intensity_idx = chrom_data.values.argmax()
                new_rt = chrom_data.index[max_intensity_idx]

                # Update only the RT value
                _targets.loc[peak_label, "rt"] = new_rt

                if plot and i < 100:  # Only plot first 100
                    i += 1
                    plt.subplot(n_rows, col_wrap, i)
                    plt.plot(chrom_data.index, chrom_data.values)
                    plt.axvline(new_rt, color="red", linestyle="--")
                    plt.title(f"{peak_label}\nm/z={mz:.3f}\nRT={new_rt:.1f}")

            except Exception as e:
                logging.error(f"Error processing {peak_label}: {str(e)}")
                continue

        self.results = _targets.reset_index()

        if self.mint is not None:
            self.mint.targets = self.results

        if plot:
            plt.tight_layout()
            return self.mint, fig
        else:
            return self.mint
