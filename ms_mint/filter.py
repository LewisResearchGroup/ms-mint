import pandas as pd


class Resampler:
    def __init__(self, tau="500ms"):
        self.tau = tau
        self.name = "resampler"

    def transform(self, t, x):
        ndx = pd.to_timedelta(t, unit="seconds")
        chrom = pd.Series(index=ndx, data=x)
        resampled = chrom.resample(self.tau).nearest()
        new_t = resampled.index.seconds + (resampled.index.microseconds / 1e6)
        new_x = resampled.values
        return new_t, new_x


class Smoother:
    def __init__(self, windows=[30, 20]):
        self.windows = windows
        self.name = "smoother"

    def transform(self, t, x):
        transformed = pd.Series(index=t, data=x)
        for window in self.windows:
            tranformed = transformed.rolling(window, center=True).mean().fillna(0)
        new_t = tranformed.index
        new_x = tranformed.values
        return new_t, new_x
