import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt

from pathlib import Path as P

from .tools import find_peaks_in_timeseries, gaussian
from .io import ms_file_to_df
from .filter import Resampler, Smoother


class Chromatogram():
    
    def __init__(self, scan_times=None, intensities=None, filters=None, expected_rt=None):
        self.t = None
        self.x = None
        if scan_times is not None:
            self.t = np.array(scan_times)
        if intensities is not None:
            self.x = intensities
        self.noise_level = None
        if filters is None:
            self.filters = [Resampler(), Smoother()]
        else: self.filters = filters
        self.peaks = None
        self.selected_peak_ndxs = None
        self.expected_rt = expected_rt
        self.weights = None

    def from_file(self, fn, mz_mean, mz_width=10, expected_rt=None):
        chrom = get_chromatogram_from_ms_file(fn, mz_mean=mz_mean, mz_width=mz_width)     
        self.t = chrom.index
        self.x = chrom.values
        if expected_rt is not None:
            self.expected_rt = expected_rt

    def estimate_noise_level(self, window=20):
        data = pd.Series(index=self.t, data=self.x)
        self.noise_level = data.rolling(window, center=True).std().median()
        
    def apply_filter(self):
        for filt in self.filters:
            self.t, self.x = filt.transform(self.t, self.x)
    
    def find_peaks(self, prominence=None):
        self.estimate_noise_level()
        if prominence is None:
            prominence = self.noise_level * 3
        self.peaks = find_peaks_in_timeseries(self.data.intensity, prominence=prominence)
    
    def optimise_peak_times_with_diff(self, rolling_window=20, plot=False):
        peaks = self.peaks
        diff = (self.data - self.data.shift(1)).rolling(rolling_window, center=True).mean().fillna(0)
        prominence = 0
            
        peak_startings = find_peaks_in_timeseries(diff.fillna(0).intensity, prominence=prominence, plot=plot)
        if plot: plt.show()
        peak_endings = find_peaks_in_timeseries(-diff.fillna(0).intensity, prominence=prominence, plot=plot)
        if plot: plt.show()

        for ndx, row in peaks.iterrows():

            new_rt_min = row.rt_min
            new_rt_max = row.rt_max

            candidates_rt_min = peak_startings[peak_startings.rt <= new_rt_min]
            candidates_rt_max = peak_endings[peak_endings.rt >= new_rt_max]

            if len(candidates_rt_min) > 0:
                new_rt_min = candidates_rt_min.tail(1).rt.values[0]
            
            if len(candidates_rt_max) > 0:
                new_rt_max = candidates_rt_max.head(1).rt.values[0]

            peaks.loc[ndx, ['rt_min', 'rt_max']] = new_rt_min, new_rt_max

    def select_peak_by_rt(self, rt):
        peaks = self.peaks
        selected_ndx = (peaks.rt - rt).abs().sort_values().index[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks
        
    def select_peak_by_highest_intensity(self):
        peaks = self.peaks
        selected_ndx = peaks.sort_values('peak_height', ascending=False).index.values[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks
    
    def select_peak_method1(self, rt, sigma):
        peaks = self.peaks
        if peaks is None or len(peaks) == 0:
            logging.warning('No peaks available to select.')
            return None
        weights = gaussian(peaks.rt, rt, sigma)
        weighted_peaks = weights * peaks.peak_height
        x = np.arange( int(self.t.min()), int(self.t.max()) )
        self.weights = max(peaks.peak_height) * gaussian(x, rt, sigma)
        selected_ndx = weighted_peaks.sort_values(ascending=False).index.values[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks        

    @property
    def selected_peaks(self):
        self.peaks.loc[self.selected_peak_ndxs]

    @property
    def data(self):
        df = pd.DataFrame(index=self.t, data={'intensity': self.x})
        df.index.name = 'scan_time'
        return df
    
    def plot(self, **kwargs):
        series = self.data
        peaks = self.peaks
        selected_peak_ndxs = self.selected_peak_ndxs
        weights = self.weights
        _plot_peaks(series, peaks, highlight=selected_peak_ndxs, expected_rt=self.expected_rt, weights=weights, **kwargs)



def get_chromatogram_from_ms_file(ms_file, mz_mean, mz_width=20):
    df = ms_file_to_df(ms_file)
    chrom = extract_chromatogram_from_ms1(df, mz_mean, mz_width=mz_width)
    return chrom


def extract_chromatogram_from_ms1(ms1, mz_mean, mz_width=10, unit="minutes"):
    mz_min, mz_max = mz_mean_width_to_mass_range(mz_mean, mz_width)
    chrom = ms1[(ms1["mz"]>=mz_min) & (ms1["mz"]<=mz_max) ].copy()
    chrom = chrom.groupby("scan_time", as_index=False).sum()
    return chrom.set_index('scan_time')['intensity']


def mz_mean_width_to_mass_range(mz_mean, mz_width_ppm=10):
    dmz = (mz_mean * 10e-6 * mz_width_ppm)
    mz_min = mz_mean - dmz
    mz_max = mz_mean + dmz
    return mz_min, mz_max        


def _plot_peaks(series, peaks, highlight=None, expected_rt=None, weights=None, legend=True):
    if highlight is None: 
        highlight = []
    ax = plt.gca()
    series.intensity.plot(ax=ax, color='black', label='Intensity')
    if peaks is not None:
        series.iloc[peaks.ndxs].plot(label='Peaks', marker='x', y='intensity', lw=0, ax=ax)
        for i, (ndx, (_, rt, rt_span, peak_base_height, peak_height, rt_min, rt_max)) in enumerate(peaks.iterrows()):
            if ndx in highlight:
                plt.axvspan(rt_min, rt_max, color='green', alpha=0.25, label='Selected')
            plt.hlines(peak_base_height, rt_min, rt_max, color='orange', label='Peak width' if i ==0 else None)    
    if expected_rt is not None:
        plt.axvspan(expected_rt, expected_rt+1, color='blue', alpha=1, label='Expected Rt')
    if weights is not None:
        plt.plot(weights, linestyle='--', label='Gaussian weight')
    plt.ylabel('Intensity')
    plt.xlabel('Scan Time [s]')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.ylim((0.1,None))
    if not legend:
        ax.get_legend().remove()

