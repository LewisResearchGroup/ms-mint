import pandas as pd
from scipy.ndimage import gaussian_filter1d


class Resampler:
    """
    Filter for time series that resamples the data in 
    a certain frequency.
    """
    def __init__(self, tau="500ms", input_unit="seconds"):
        """
        Filter for time series that resamples the data in 
        a certain frequency. The default is 500ms.

        :param tau: Sampling frequency, defaults to "500ms"
        :type tau: str, optional
        :param unit: Time unit of input series (t)
        :type str: "seconds" or "minutes"
        """
        self.tau = tau
        self.unit = input_unit
        self.name = "resampler"

    def transform(self, t, x):
        """
        Transformation method.

        :param t: Time points of series
        :type t: Array or List
        :param x: Data points of series
        :type x: Array or List
        :return: Resampled time series (x, t)
        :rtype: tuple
        """
        ndx = pd.to_timedelta(t, unit=self.unit)
        chrom = pd.Series(index=ndx, data=x)
        #resampled = chrom.resample(self.tau).nearest()
        resampled = chrom.resample(self.tau).fillna('nearest', limit=10)
        new_t = resampled.index.seconds + (resampled.index.microseconds / 1e6)
        new_x = resampled.values
        return new_t, new_x


class Smoother:
    """
    Filter for time series that smoothes the
    x values by running one or more rolling
    averages.
    """
    def __init__(self, windows=[30, 20]):
        """
        Filter for time series that smoothes the
        x values by running one or more rolling
        averages.

        :param windows: Window sizes of rolling averages applied to time series, defaults to [30, 20]
        :type windows: : List[int], optional
        """
        self.windows = windows
        self.name = "smoother"

    def transform(self, t, x):
        """
        Transformation method.

        :param t: Time points of series
        :type t: Array or List
        :param x: Data points of series
        :type x: Array or List
        :return: Resampled time series (x, t)
        :rtype: tuple
        """
        transformed = pd.Series(index=t, data=x)
        for window in self.windows:
            tranformed = transformed.rolling(window, center=True).mean().fillna(0)
        new_t = tranformed.index
        new_x = tranformed.values
        return new_t, new_x


class GaussFilter:
    """
    Filter for time series that applies a Gaussian filter.
    """
    def __init__(self, sigma=5):
        """
        Filter for time series that applies a Gaussian filter.

        :param sigma: Sigma value for Gaussian function, defaults to 5
        :type sigma: int, optional
        """
        self.sigma = sigma

    def transform(self, t, x):
        """
        Transformation method.

        :param t: Time points of series
        :type t: Array or List
        :param x: Data points of series
        :type x: Array or List
        :return: Resampled time series (x, t)
        :rtype: tuple
        """
        new_x = gaussian_filter1d(x, sigma=self.sigma)
        return t, new_x
