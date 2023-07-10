import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import sys

from .io import ms_file_to_df
from .Chromatogram import Chromatogram
from .processing import get_chromatogram_from_ms_file, extract_chromatogram_from_ms1  




class TargetOptimizer:
    """
    Optimizer for target lists.

    :param mint: Mint instance to optimize
    :type mint: ms_mint.Mint.Mint
    """

    def __init__(self, mint=None):
        """
        Optimizer for target lists.

        :param mint: Mint instance to optimize
        :type mint: ms_mint.Mint.Mint
        """
        self.mint = mint
        self.reset()

    def reset(self):
        self.results = None
        return self

    def rt_min_max(
        self,
        fns=None,
        targets=None,
        peak_labels=None,
        minimum_intensity=1e4,
        plot=False,
        sigma=20,
        filters=None,
        post_opt=False,
        post_opt_kwargs=None,
        rel_height=0.9,
        height=3,
        aspect=2,
        col_wrap=3,
        **kwargs,
    ):
        """
        Optimize rt_min and rt_max values based on expected retention times (rt).
        For this optimization all rt values in the target list must be present.

        :param fns: List of filenames to use for optimization
        :type fns: List[str or PosixPath], optional
        :param targets: Target list to optimize
        :type targets: pandas.DataFrame in MINT target list format
        :param peak_labels: Subset of peak_labels to optimize, defaults to None
        :type peak_labels: List[str or PosixPath], optional
        :param minimum_intensity: Minimum intensity required, otherwise skip target, defaults to 1e4
        :type minimum_intensity: float, optional
        :param plot: Whether or not to plot first 100 optimizations, defaults to True
        :type plot: bool, optional
        :param sigma: Sigma value for peak selection, defaults to 20
        :type sigma: float, optional
        :param filters: Filter instances to apply in respective order, defaults to None
        :type filters: ms_mint.filters.Filter, optional
        :param post_opt: Optimize retention times after peak selection, defaults to False
        :type post_opt: bool, optional
        :param post_opt_kwargs: _description_, defaults to 20
        :type post_opt_kwargs: int, optional
        :return: (self, None) or (self, matplotlib.pyplot.Figure) if plot==True
        :rtype: tuple
        """

        if targets is None:
            targets = self.mint.targets.reset_index()

        if fns is None:
            fns = self.mint.ms_files

        if peak_labels is None:
            peak_labels = targets.peak_label.values

        _targets = targets.set_index("peak_label").copy()

        ms1 = pd.concat([ms_file_to_df(fn) for fn in self.mint.tqdm(fns, desc='Reading files')]).sort_values(
            ["scan_time", "mz"]
        )

        if plot:
            n_rows = int(np.ceil(len(peak_labels) / col_wrap))
            fig = plt.figure(figsize=(col_wrap * height * aspect, n_rows * height))

        i = 0
        for peak_label, row in self.mint.tqdm(_targets.iterrows(), total=len(targets), desc='Optimizing targets'):
            if peak_label not in peak_labels:
                logging.warning(f"{peak_label} not in {peak_labels}")
                continue

            mz = row.mz_mean
            rt = row.rt

            _slice = extract_chromatogram_from_ms1(ms1, mz).groupby("scan_time").sum()

            chrom = Chromatogram(
                _slice.index, _slice.values, expected_rt=rt,   filters=filters
            )

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
