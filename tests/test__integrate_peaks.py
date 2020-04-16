import pandas as pd
import numpy as np

from ms_mint.tools import integrate_peak

def fake_ms_data(min_rt=0, max_rt=2, min_mz=0, max_mz=200, drt=.1, dmz=.1):
    retention_times = np.arange(min_rt+drt, max_rt+drt, drt)
    mz_values = np.arange(min_mz+dmz, max_mz+dmz, dmz)
    df = pd.DataFrame(index=mz_values, columns=retention_times)
    df = df.unstack().to_frame().reset_index()
    df.columns = ['retentionTime', 'm/z array', 'intensity array']
    df['intensity array'] = 1
    return df


def test__integrate_peak():
    df = fake_ms_data(0, 10, 0, 200, 1, 1)
    mz_mean = 100
    mz_width = 1
    rt_min = 0
    rt_max = 10
    peak_label = 'Label'
    intensity_threshold = 0
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label).loc[0, 'peak_area']
    expected = 10
    assert result == expected, f'{result} != {expected}'


def test__integrate_peak__rtmax():
    df = fake_ms_data(0, 10, 0, 200, 1, 1)
    mz_mean = 100
    mz_width = 1
    rt_min = 0
    rt_max = 5
    peak_label = 'Label'
    intensity_threshold = 0
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                                intensity_threshold, peak_label).loc[0, 'peak_area']
    expected = 5
    assert result == expected, f'{result} != {expected}'


def test__integrate_peak__rtmin():
    df = fake_ms_data(0, 10, 0, 200, 1, 1)
    mz_mean = 100
    mz_width = 1
    rt_min = 5.1
    rt_max = 10
    peak_label = 'Label'
    intensity_threshold = 0
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                            intensity_threshold, peak_label).loc[0, 'peak_area']
    expected = 5
    assert result == expected, f'{result} != {expected}'

    
def test__integrate_peak__mz():
    df = fake_ms_data(0, 10, 0, 200, 1, 1)
    mz_mean = 100
    rt_min = 0
    rt_max = 10
    peak_label = 'Label'
    intensity_threshold = 0
    results = []
    
    # Expect one per minute (= 10)
    mz_width = 10
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                            intensity_threshold, peak_label).loc[0, 'peak_area']
    results.append(result)
    
    # Expect 3 per minute from mz_mean-1 and mz_mean+1 (= 30)
    mz_width = 10001
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                            intensity_threshold, peak_label).loc[0, 'peak_area']
    results.append(result)
    
    # Expect 1 per minute as dmz depends on mz_width (= 10)
    mz_mean = 1
    mz_width = 10001
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                            intensity_threshold, peak_label).loc[0, 'peak_area']
    results.append(result)
    
    expected = np.array([10, 30, 10])
    assert (np.array(results) == expected).all(), results
    

def test__integrate_peak__intensity_threshold():
    df = fake_ms_data(0, 10, 99, 100, 1, 1)
    mz_mean = 100
    mz_width = 10
    rt_min = 0
    rt_max = 10
    peak_label = 'Label'
    df['intensity array'] = df.retentionTime * 1000
    intensity_threshold = 5000
    result = integrate_peak(df, mz_mean, mz_width, rt_min, rt_max, 
                            intensity_threshold, peak_label).loc[0, 'peak_area']
    expected = 45000
    assert result == expected, f'{result} != {expected}'
