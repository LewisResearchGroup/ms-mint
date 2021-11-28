import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import KernelDensity

from ..tools import gaussian
from ..Resampler import Resampler


class RetentionTimeOptimizer:
    def __init__(
        self,
        rt_min=None,
        rt_max=None,
        rt=None,
        rt_margin=0.3,
        how="largest",
        dt=0.01,
        show_figure=False,
        precission=3,
    ):

        self.t_peaks = None
        self.x_peaks = None
        self.peak_width_bottom = None
        self.rt_max = rt_max
        self.rt_min = rt_min
        self.rt_margin = rt_margin
        self.rt = rt
        self.how = how
        self.dt = dt
        self.resampler = Resampler()
        self.show_figure = show_figure
        self.precission = precission
        self.update_values()

    def update_values(self):
        if (self.rt_min is not None) and (self.rt_max is not None):
            if self.rt is None:
                self.rt = np.mean([self.rt_min, self.rt_max])
            if self.rt_margin is None:
                self.rt_margin = (self.rt_max - self.rt_min) / 2

    def find_largest_peak(self, chromatograms):
        props = []
        for chrom in chromatograms:
            prop = self.find_largest_peak_in_chromatogram(chrom)
            if prop is not None:
                props.append(prop)

        if len(props) == 0:
            return (None, None)
        df = pd.DataFrame(props, columns=["rt", "max_intensity", "rt_min", "rt_max"])

        # rt_of_largest_peak_max = df.sort_values('max_intensity', ascending=False).iloc[0]['rt']
        # Filter out far away peaks
        # df = df[(df.rt - rt_of_largest_peak_max).abs()<0.3]

        rt_min = estimate_expectation_value(df.rt_min)
        rt_max = estimate_expectation_value(df.rt_max)

        if self.show_figure:
            plt.vlines(
                [rt_min, rt_max],
                0,
                np.max(df.max_intensity),
                label="Selected RT range",
                color="k",
                ls="--",
            )
            margin = (rt_max - rt_min) * 0.1
            plt.xlim(rt_min - margin, rt_max + margin)
            plt.show()

        return np.round(rt_min, self.precission), np.round(rt_max, self.precission)

    def find_largest_peak_in_chromatogram(self, chrom):
        chrom = chrom.copy()
        chrom_weighted = self.weighted_chrom(chrom)
        chrom_resampled = self.resample(chrom_weighted)
        chrom_smoothed = self.smooth(chrom_resampled)

        self.get_chrom_properties(chrom_smoothed)
        self.find_peaks_and_widths(chrom_smoothed)

        ndx = self.ndx_largest_peak()
        if ndx is None:
            return None
        prop = self.get_peak_properties(ndx)

        if self.show_figure:
            chrom.plot(color="grey", lw=0.5)
            chrom_weighted.plot(ax=plt.gca(), color="darkblue", alpha=0.8)
            chrom_resampled.plot(ax=plt.gca(), color="cyan", lw=0.8)
            chrom_smoothed.plot(ax=plt.gca(), color="orange", lw=1)
            plt.xlim(prop[2] - 0.1, prop[3] + 0.1)
            plt.hlines(-0.1, prop[2], prop[3], lw=2)

        return prop

    def get_chrom_properties(self, chrom):
        self.t_min = chrom.index.min()
        self.t_max = chrom.index.max()
        self.t_span = self.t_max - self.t_min
        self.len_chrom = len(chrom)

    def find_peaks_and_widths(self, chrom, prominence=1000):
        x = chrom.values
        ndx_peaks, _ = find_peaks(x, prominence=(prominence, None))
        t_peaks = chrom.iloc[ndx_peaks].index
        x_peaks = chrom.iloc[ndx_peaks].values
        results_bottom = peak_widths(x, ndx_peaks, rel_height=0.80)
        self.t_peaks = t_peaks
        self.x_peaks = x_peaks
        self.peak_width_bottom = results_bottom

    def get_peak_properties(self, ndx):
        """
        Returns a touple (x_max, t_of_x_max, t_0, t_1)
        """
        return (
            self.t_peaks[ndx],
            self.x_peaks[ndx],
            self.get_peak_start_time(ndx),
            self.get_peak_end_time(ndx),
        )

    def weighted_chrom(self, chrom):
        mean = self.rt
        sigma = self.rt_margin
        weights = gaussian(chrom.index, mean, sigma)
        return chrom * weights

    def ndx_largest_peak(self):
        x_peaks = self.x_peaks
        if len(x_peaks) == 0:
            return None
        return np.argmax(x_peaks)

    def resample(self, chrom):
        return self.resampler.resample(chrom)

    def smooth(self, chrom, n=30):
        return chrom.rolling(n, center=True).mean()

    def x_to_t(self, x):
        return ((self.t_span * np.array(x) / self.len_chrom) + self.t_min).flatten()

    def get_peak_start_time(self, ndx):
        rt_min = self.x_to_t(self.peak_width_bottom[2][ndx])[0]
        return rt_min

    def get_peak_end_time(self, ndx):
        rt_max = self.x_to_t(self.peak_width_bottom[3][ndx])[0]
        return rt_max


def estimate_expectation_value(values, kernel="gaussian"):
    values = np.array(values)
    assert kernel in ["gaussian", "tophat", "epanechnikov"]
    x = np.linspace(np.min(values) - 1, np.max(values) + 1, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(values[:, np.newaxis])
    log_dens = kde.score_samples(x)
    ndx = np.argmax(log_dens)
    return x[ndx][0]
