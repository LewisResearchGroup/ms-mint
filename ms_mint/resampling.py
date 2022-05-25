"""
Class to resample MS data.
"""

import pandas as pd


class Resampler:
    """Class to resample a chromatogram.

    :param tau: Resampling frequency, defaults to '50ms'
    :type tau: str
    :param unit: Time unit used, defaults to 'minutes'
    :type unit: str, ['minutes', 'seconds']
    :param smooth: Whether or not to apply rolling average.
    :type smooth: bool
    """
    def __init__(self, tau="50ms", unit="minutes", smooth=True):
        self._unit = unit
        self._tau = tau
        self._smooth = smooth

    def _resample_unit_minutes_(self, chrom):
        chrom = chrom.sort_index()
        chrom.index = pd.to_timedelta(chrom.index, unit=self._unit)
        chrom = chrom.resample(self._tau).nearest()
        if self._unit == "seconds":
            chrom.index = chrom.index.seconds + chrom.index.microseconds / 1000000
        if self._unit == "minutes":
            chrom.index = (
                chrom.index.seconds + chrom.index.microseconds / 1000000
            ) / 60
        if self._smooth:
            chrom = chrom.rolling(20, center=True).mean()
            chrom = chrom.rolling(5, center=True).mean()
        return chrom

    def resample(self, chrom):
        """Resample data.

        :param chrom: _description_
        :type chrom: _type_
        :return: _description_
        :rtype: _type_
        """        
        return self._resample_unit_minutes_(chrom)
