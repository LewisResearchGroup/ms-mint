import pandas as pd

class Resampler():
    def __init__(self, tau='50ms', unit='minute'):
        self._unit = unit
        self._tau = tau
    
    def resample_unit_minutes(self, chrom):
        chrom = chrom.sort_index()
        chrom.index = pd.to_timedelta(chrom.index, unit=self._unit)
        chrom = chrom.resample(self._tau).nearest()
        if self._unit == 'second':
            chrom.index = (chrom.index.seconds + chrom.index.microseconds / 1000000)
        if self._unit == 'minute':
            chrom.index = (chrom.index.seconds + chrom.index.microseconds / 1000000) / 60
        chrom = chrom.rolling(20, center=True).mean()
        chrom = chrom.rolling(5, center=True).mean()
        return chrom
    
    def resample(self, chrom):
        if self._unit == 'minute':
            return self.resample_unit_minutes(chrom)
        else:
            raise NotImplementedError