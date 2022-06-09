import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt

from .tools import find_peaks_in_timeseries, gaussian
from .io import ms_file_to_df
from .filter import Resampler, Smoother, GaussFilter
from .matplotlib_tools import plot_peaks


class Chromatogram:
    def __init__(
        self, scan_times=None, intensities=None, filter=None, expected_rt=None
    ):
        self.t = np.array([0])
        self.x = np.array([0])
        if scan_times is not None:
            self.t = np.append(self.t, scan_times)
        if intensities is not None:
            self.x = np.append(self.x, intensities)
        self.noise_level = None
        if filter is None:
            self.filter = [Resampler(), GaussFilter(), Smoother()]
        else:
            self.filter = filter
        self.peaks = None
        self.selected_peak_ndxs = None
        self.expected_rt = expected_rt
        self.weights = None

    def from_file(self, fn, mz_mean, mz_width=10, expected_rt=None):
        chrom = get_chromatogram_from_ms_file(fn, mz_mean=mz_mean, mz_width=mz_width)
        self.t = np.append(self.t, chrom.index)
        self.x = np.append(self.x, chrom.values)
        if expected_rt is not None:
            self.expected_rt = expected_rt

    def estimate_noise_level(self, window=20):
        data = pd.Series(index=self.t, data=self.x)
        self.noise_level = data.rolling(window, center=True).std().median()

    def apply_filter(self):
        for filt in self.filter:
            self.t, self.x = filt.transform(self.t, self.x)

    def find_peaks(self, prominence=None, rel_height=0.9):
        self.estimate_noise_level()
        if prominence is None:
            prominence = self.noise_level * 3
        self.peaks = find_peaks_in_timeseries(
            self.data.intensity, prominence=prominence, rel_height=rel_height,
        )

    def optimise_peak_times_with_diff(self, rolling_window=20, plot=False):
        peaks = self.peaks
        diff = (
            (self.data - self.data.shift(1))
            .rolling(rolling_window, center=True)
            .mean()
            .fillna(0)
        )
        prominence = 0

        peak_startings = find_peaks_in_timeseries(
            diff.fillna(0).intensity, prominence=prominence, plot=plot
        )
        if plot:
            plt.show()

        peak_endings = find_peaks_in_timeseries(
            -diff.fillna(0).intensity, prominence=prominence, plot=plot
        )
        if plot:
            plt.show()

        for ndx, row in peaks.iterrows():

            new_rt_min = row.rt_min
            new_rt_max = row.rt_max

            candidates_rt_min = peak_startings[peak_startings.rt <= new_rt_min]
            candidates_rt_max = peak_endings[peak_endings.rt >= new_rt_max]

            if len(candidates_rt_min) > 0:
                new_rt_min = candidates_rt_min.tail(1).rt.values[0]

            if len(candidates_rt_max) > 0:
                new_rt_max = candidates_rt_max.head(1).rt.values[0]

            peaks.loc[ndx, ["rt_min", "rt_max"]] = new_rt_min, new_rt_max

    def select_peak_by_rt(self, expected_rt=None):
        peaks = self.peaks
        if expected_rt is None:
            expected_rt = self.expected_rt
        else:
            self.expected_rt = expected_rt
        selected_ndx = (peaks.rt - expected_rt).abs().sort_values().index[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks

    def select_peak_by_highest_intensity(self):
        peaks = self.peaks
        selected_ndx = peaks.sort_values("peak_height", ascending=False).index.values[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks

    def select_peak_with_gaussian_weight(self, expected_rt=None, sigma=50):
        peaks = self.peaks
        if expected_rt is None:
            expected_rt = self.expected_rt
        else:
            self.expected_rt = expected_rt
        if peaks is None or len(peaks) == 0:
            logging.warning("No peaks available to select.")
            return None
        weights = gaussian(peaks.rt, expected_rt, sigma)
        weighted_peaks = weights * peaks.peak_height
        x = np.arange(int(self.t.min()), int(self.t.max()))
        self.weights = max(peaks.peak_height) * gaussian(x, expected_rt, sigma)
        selected_ndx = weighted_peaks.sort_values(ascending=False).index.values[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks

    @property
    def selected_peaks(self):
        self.peaks.loc[self.selected_peak_ndxs]

    @property
    def data(self):
        df = pd.DataFrame(index=self.t, data={"intensity": self.x})
        df.index.name = "scan_time"
        return df

    def plot(self, **kwargs):
        series = self.data
        peaks = self.peaks
        selected_peak_ndxs = self.selected_peak_ndxs
        weights = self.weights
        fig = plot_peaks(
            series,
            peaks,
            highlight=selected_peak_ndxs,
            expected_rt=self.expected_rt,
            weights=weights,
            **kwargs
        )
        return fig


def get_chromatogram_from_ms_file(ms_file, mz_mean, mz_width=20):
    df = ms_file_to_df(ms_file)
    chrom = extract_chromatogram_from_ms1(df, mz_mean, mz_width=mz_width)
    return chrom


def extract_chromatogram_from_ms1(ms1, mz_mean, mz_width=10, unit="minutes"):
    mz_min, mz_max = mz_mean_width_to_mass_range(mz_mean, mz_width)
    chrom = ms1[(ms1["mz"] >= mz_min) & (ms1["mz"] <= mz_max)].copy()
    chrom = chrom.groupby("scan_time", as_index=False).sum()
    return chrom.set_index("scan_time")["intensity"]


def mz_mean_width_to_mass_range(mz_mean, mz_width_ppm=10):
    dmz = mz_mean * 10e-6 * mz_width_ppm
    mz_min = mz_mean - dmz
    mz_max = mz_mean + dmz
    return mz_min, mz_max
