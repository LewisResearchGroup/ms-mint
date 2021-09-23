import pandas as pd
from ms_mint.Resampler import Resampler


def test__Resampler_50ms_minutes_dt():
    chrom = pd.Series([0, 10, 5], index=[0, 0.9, 1])
    result = Resampler(smooth=False, tau='50ms', unit='minutes').resample(chrom)
    tau_in_seconds = result.index[1] * 60
    assert tau_in_seconds == 0.05

def test__Resampler_1s_minutes_dt():
    chrom = pd.Series([0, 10, 5], index=[0, 0.9, 1])
    result = Resampler(smooth=False, tau='1s', unit='minutes').resample(chrom)
    tau_in_seconds = result.index[1] * 60
    assert tau_in_seconds == 1

def test__Resampler_1s_seconds_dt():
    chrom = pd.Series([0, 10, 5], index=[0, 0.9, 1])
    result = Resampler(smooth=False, tau='1s', unit='seconds').resample(chrom)
    tau_in_seconds = result.index[1]
    assert tau_in_seconds == 1

def test__Resampler_smooth_1s_seconds_dt():
    chrom = pd.Series([0, 10, 5], index=[0, 0.9, 1])
    result = Resampler(smooth=True, tau='1s', unit='seconds').resample(chrom)
    tau_in_seconds = result.index[1]
    assert tau_in_seconds == 1
