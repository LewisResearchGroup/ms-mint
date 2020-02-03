import pandas as pd
import numpy as np
import os

from pathlib import Path as P

from ms_mint.tools import MINT_ROOT, integrate_peaks_from_filename,\
    STANDARD_PEAKLIST, integrate_peak

TEST_MZXML = os.path.abspath(str(P(MINT_ROOT)/P('../tests/data/test.mzXML')))

print(TEST_MZXML)
def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML),\
        'Test mzXML ({}) not found.'.format(TEST_MZXML)

def test__integrate_peaks_from_filename():
    result = integrate_peaks_from_filename(TEST_MZXML)
    assert isinstance(result, pd.DataFrame)
    assert len(STANDARD_PEAKLIST) == len(result)

def test__integrate_peak():
    mzxml_df = pd.DataFrame({
        'num': [None]*3, 
        'scanType': [None]*3, 
        'centroided': [None]*3, 
        'msLevel': [None]*3, 
        'peaksCount': [None]*3,
        'polarity': [None]*3,
        'retentionTime': [1]*3, 
        'lowMz': [None]*3, 
        'highMz': [None]*3, 
        'basePeakMz': [None]*3, 
        'basePeakIntensity': [None]*3,
        'totIonCurrent': [None]*3, 
        'id': [None]*3, 
        'm/z array': [100, 100.0001, 100.0002], 
        'intensity array': [1, 10, 100]})
    mz_mean = 100
    mz_width = 1
    rt_min = 0
    rt_max = 10
    peak_label = 'Label'
    intensity_threshold = 0
    result = integrate_peak(mzxml_df, mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label)
    expected = 11
    assert result == expected, f'{result} != {expected}'

def fake_ms_data(N=100)
    df = pd.DataFrame({
        'num': [None]*N, 
        'scanType': [None]*N, 
        'centroided': [None]*N, 
        'msLevel': [None]*N, 
        'peaksCount': [None]*N,
        'polarity': [None]*N,
        'retentionTime': [1]*N, 
        'lowMz': [None]*N, 
        'highMz': [None]*N, 
        'basePeakMz': [None]*N, 
        'basePeakIntensity': [None]*N,
        'totIonCurrent': [None]*N, 
        'id': [None]*N, 
        'm/z array': [100]*N, 
        'intensity array': [100]*N})
    return df