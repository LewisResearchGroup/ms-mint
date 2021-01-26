import pandas as pd

class Resampler():
    def __init__(self, frequeny='50ms', unit='minute'):
        self._unit = unit
        self._frequency = frequeny
    
    def resample_unit_minutes(self, chrom):
        chrom = chrom.sort_index()
        chrom.index = pd.to_timedelta(chrom.index, unit='minute')
        chrom = chrom.resample(self._frequency).nearest()
        chrom.index = (chrom.index.seconds + chrom.index.microseconds / 1000000) / 60
        chrom = chrom.rolling(20, center=True).mean()
        chrom = chrom.rolling(5, center=True).mean()
        return chrom
    
    def resample(self, chrom):
        if self._unit == 'minute':
            return self.resample_unit_minutes(chrom)
        else:
            raise NotImplementedError