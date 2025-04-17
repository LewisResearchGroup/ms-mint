import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Union, Optional, Sequence, Literal


class Filter:
    """Base class for time series filters.

    All filter implementations should inherit from this class and
    implement the transform method.
    """

    def transform(
        self, t: Union[List[float], np.ndarray], x: Union[List[float], np.ndarray]
    ) -> Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]:
        """Transform the time series data.

        Args:
            t: Time points of series
            x: Data points of series

        Returns:
            Tuple of (transformed time points, transformed data points)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


class Resampler(Filter):
    """Filter for time series that resamples the data at a specified frequency.

    Resamples time series data to a regular time grid with the specified
    frequency using pandas time series functionality.
    """

    def __init__(
        self, tau: str = "500ms", input_unit: Literal["seconds", "minutes"] = "seconds"
    ) -> None:
        """Initialize the Resampler filter.

        Args:
            tau: Sampling frequency/period. Default is "500ms".
            input_unit: Time unit of input series (t). Must be either "seconds" or "minutes".
        """
        self.tau = tau
        self.unit = input_unit
        self.name = "resampler"

    def transform(
        self, t: Union[List[float], np.ndarray], x: Union[List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the time series to the specified frequency.

        Args:
            t: Time points of series
            x: Data points of series

        Returns:
            Tuple of (resampled time points, resampled data points)
        """
        ndx = pd.to_timedelta(t, unit=self.unit)
        chrom = pd.Series(index=ndx, data=x)
        # resampled = chrom.resample(self.tau).nearest()
        resampled = chrom.resample(self.tau).fillna("nearest", limit=10)
        new_t = np.array(resampled.index.seconds + (resampled.index.microseconds / 1e6))
        new_x = resampled.values
        return new_t, new_x


class Smoother(Filter):
    """Filter for time series that smoothes data using rolling averages.

    Smoothes the data points by applying one or more rolling averages
    with specified window sizes.
    """

    def __init__(self, windows: Optional[List[int]] = None) -> None:
        """Initialize the Smoother filter.

        Args:
            windows: Window sizes for rolling averages applied to time series.
                Default is [30, 20] if None is provided.
        """
        if windows is None:
            windows = [30, 20]
        self.windows = windows
        self.name = "smoother"

    def transform(
        self, t: Union[List[float], np.ndarray], x: Union[List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rolling average smoothing to the time series.

        Args:
            t: Time points of series
            x: Data points of series

        Returns:
            Tuple of (time points, smoothed data points)
        """
        transformed = pd.Series(index=t, data=x)
        for window in self.windows:
            transformed = transformed.rolling(window, center=True).mean().fillna(0)
        new_t = np.array(transformed.index)
        new_x = transformed.values
        return new_t, new_x


class GaussFilter(Filter):
    """Filter for time series that applies a Gaussian filter.

    Smoothes data using a Gaussian filter with the specified sigma value.
    """

    def __init__(self, sigma: float = 5) -> None:
        """Initialize the Gaussian filter.

        Args:
            sigma: Standard deviation for Gaussian kernel. Default is 5.
        """
        self.sigma = sigma
        self.name = "gauss_filter"

    def transform(
        self, t: Union[List[float], np.ndarray], x: Union[List[float], np.ndarray]
    ) -> Tuple[Union[List[float], np.ndarray], np.ndarray]:
        """Apply Gaussian filtering to the time series.

        Args:
            t: Time points of series
            x: Data points of series

        Returns:
            Tuple of (time points, Gaussian filtered data points)
        """
        new_x = gaussian_filter1d(x, sigma=self.sigma)
        return t, new_x
