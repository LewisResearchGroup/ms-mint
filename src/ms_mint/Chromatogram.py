import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Union, List, Any, Tuple, Dict

from .tools import find_peaks_in_timeseries, gaussian, mz_mean_width_to_min_max
from .io import ms_file_to_df
from .filters import Filter, Resampler, Smoother, GaussFilter
from .matplotlib_tools import plot_peaks
from .processing import get_chromatogram_from_ms_file


class Chromatogram:
    """A class for handling chromatogram data extraction and processing.

    This class provides functionality to extract, process, and analyze chromatogram data
    from mass spectrometry files, including peak detection and visualization capabilities.

    Attributes:
        t: Array of scan times.
        x: Array of intensity values.
        noise_level: Estimated noise level of the chromatogram.
        filters: List of filters to be applied to the chromatogram.
        peaks: DataFrame containing detected peaks information.
        selected_peak_ndxs: Indices of selected peaks.
        expected_rt: Expected retention time.
        weights: Weighting values for peak selection.
    """

    def __init__(
        self,
        scan_times: Optional[Union[List[float], np.ndarray]] = None,
        intensities: Optional[Union[List[float], np.ndarray]] = None,
        filters: Optional[List[Filter]] = None,
        expected_rt: Optional[float] = None,
    ) -> None:
        """Initialize a Chromatogram object.

        Args:
            scan_times: Array-like object containing the scan times.
            intensities: Array-like object containing the intensities.
            filters: List of filters to be applied.
            expected_rt: Expected retention time in seconds.
        """
        # Initialize empty arrays for scan_times and intensities
        self.t: np.ndarray = (
            np.array([]) if scan_times is None or 0 in scan_times else np.array([0])
        )
        self.x: np.ndarray = (
            np.array([]) if intensities is None or 0 in scan_times else np.array([0])
        )

        # Update scan_times and intensities if provided
        if scan_times is not None:
            self.t = np.append(self.t, scan_times)
        if intensities is not None:
            self.x = np.append(self.x, intensities)

        # Initialize other attributes
        self.noise_level: Optional[float] = None
        self.filters: List[Filter] = filters or [Resampler(), GaussFilter(), Smoother()]
        self.peaks: Optional[pd.DataFrame] = None
        self.selected_peak_ndxs: Optional[List[int]] = None
        self.expected_rt: Optional[float] = expected_rt
        self.weights: Optional[np.ndarray] = None

    def from_file(
        self, fn: str, mz_mean: float, mz_width: float = 10, expected_rt: Optional[float] = None
    ) -> None:
        """Load chromatogram data from a mass spectrometry file.

        Args:
            fn: Filename of the mass spectrometry file.
            mz_mean: Mean m/z value to extract.
            mz_width: Width of the m/z window to extract.
            expected_rt: Expected retention time in seconds.
        """
        chrom = get_chromatogram_from_ms_file(fn, mz_mean=mz_mean, mz_width=mz_width)
        self.t = np.append(self.t, chrom.index)
        self.x = np.append(self.x, chrom.values)
        if expected_rt is not None:
            self.expected_rt = expected_rt

    def estimate_noise_level(self, window: int = 20) -> None:
        """Estimate the noise level of the chromatogram.

        Uses a rolling window standard deviation approach to estimate the baseline noise.

        Args:
            window: Size of the rolling window for noise estimation.
        """
        data = pd.Series(index=self.t, data=self.x)
        self.noise_level = data.rolling(window, center=True).std().median()

    def apply_filters(self) -> None:
        """Apply all filters in the filter list to the chromatogram data."""
        for filt in self.filters:
            self.t, self.x = filt.transform(self.t, self.x)

    def find_peaks(
        self, prominence: Optional[float] = None, rel_height: float = 0.9, **kwargs
    ) -> None:
        """Find peaks in the chromatogram.

        Args:
            prominence: Minimum prominence of peaks. If None, estimated from noise level.
            rel_height: Relative height for determining peak width.
            **kwargs: Additional keyword arguments to pass to the peak finding function.
        """
        self.estimate_noise_level()
        if prominence is None:
            prominence = self.noise_level * 5
        self.peaks = find_peaks_in_timeseries(
            self.data.intensity, prominence=prominence, rel_height=rel_height, **kwargs
        )

    def optimise_peak_times_with_diff(self, rolling_window: int = 20, plot: bool = False) -> None:
        """Optimize peak start and end times using the derivative.

        Uses the first derivative of the chromatogram to more accurately determine
        peak start and end times.

        Args:
            rolling_window: Window size for rolling mean calculation of the derivative.
            plot: Whether to plot the results of peak detection on the derivative.
        """
        peaks = self.peaks
        diff = (
            (self.data - self.data.shift(1)).rolling(rolling_window, center=True).mean().fillna(0)
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

    def select_peak_by_rt(self, expected_rt: Optional[float] = None) -> pd.DataFrame:
        """Select the peak closest to the expected retention time.

        Args:
            expected_rt: Expected retention time in seconds. If None, uses the stored expected_rt.

        Returns:
            DataFrame containing the selected peak information.
        """
        peaks = self.peaks
        if expected_rt is None:
            expected_rt = self.expected_rt
        else:
            self.expected_rt = expected_rt
        selected_ndx = (peaks.rt - expected_rt).abs().sort_values().index[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks

    def select_peak_by_highest_intensity(self) -> pd.DataFrame:
        """Select the peak with the highest intensity.

        Returns:
            DataFrame containing the selected peak information.
        """
        peaks = self.peaks
        selected_ndx = peaks.sort_values("peak_height", ascending=False).index.values[0]
        self.selected_peak_ndxs = [selected_ndx]
        return self.selected_peaks

    def select_peak_with_gaussian_weight(
        self, expected_rt: Optional[float] = None, sigma: float = 50
    ) -> Optional[pd.DataFrame]:
        """Select peak using Gaussian weighting around expected retention time.

        This method applies a Gaussian weighting centered at the expected retention time
        to favor peaks close to the expected time while still considering peak height.

        Args:
            expected_rt: Expected retention time in seconds. If None, uses the stored expected_rt.
            sigma: Standard deviation of the Gaussian weight function in seconds.

        Returns:
            DataFrame containing the selected peak information, or None if no peaks available.
        """
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
    def selected_peaks(self) -> pd.DataFrame:
        """Get DataFrame of the currently selected peaks.

        Returns:
            DataFrame containing information about the selected peaks.
        """
        return self.peaks.loc[self.selected_peak_ndxs]

    @property
    def data(self) -> pd.DataFrame:
        """Get chromatogram data as a DataFrame.

        Returns:
            DataFrame with scan times as index and intensity as a column.
        """
        df = pd.DataFrame(index=self.t, data={"intensity": self.x})
        df.index.name = "scan_time"
        return df

    def plot(self, label: Optional[str] = None, **kwargs) -> Figure:
        """Plot the chromatogram with detected peaks.

        Args:
            label: Label for the plot.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            Matplotlib Figure object.
        """
        series = self.data
        peaks = self.peaks
        selected_peak_ndxs = self.selected_peak_ndxs
        weights = self.weights
        fig = plot_peaks(
            series,
            peaks,
            label=label,
            highlight=selected_peak_ndxs,
            expected_rt=self.expected_rt,
            weights=weights,
            **kwargs,
        )
        return fig
