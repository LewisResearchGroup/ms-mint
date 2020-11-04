import os
import pandas as pd 
import numpy as np

from pathlib import Path as P

from ms_mint import tools

TEST_MZXML = os.path.abspath(str(P(tools.MINT_ROOT)/P('../tests/data/test.mzXML')))


def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML),\
        'Test mzXML ({}) not found.'.format(TEST_MZXML)


def test__integrate_peaks():
    ms_data = pd.DataFrame({'retentionTime': [1, 2, 3],
                            'm/z array': [100,200,300], 
                            'intensity array': [2, 3, 7]})

    peaklist = pd.DataFrame(
        {'peak_label': ['A'],
         'mz_mean': [200],
         'mz_width': [10],
         'intensity_threshold': [0],
         'rt_min': [0],
         'rt_max': [10]})

    result = tools.integrate_peaks(ms_data, peaklist)

    expected = pd.DataFrame({'peak_label': {0: 'A'},
                             'mz_mean': {0: 200},
                             'mz_width': {0: 10},
                             'intensity_threshold': {0: 0},
                             'rt_min': {0: 0},
                             'rt_max': {0: 10},
                             'peak_area': {0: 5},
                             'peak_n_datapoints': {0: 2},
                             'peak_max': {0: 3},
                             'peak_min': {0: 2},
                             'peak_median': {0: 2.5},
                             'peak_mean': {0: 2.5},
                             'peak_delta_int': {0: 1},
                             'peak_shape_rt': {0: '1,2'},
                             'peak_shape_int': {0: '100,200'},
                             'peak_mass_diff_25pc': {0: 125.0},
                             'peak_mass_diff_50pc': {0: 150.0},
                             'peak_mass_diff_75pc': {0: 175.0}})

    assert result.equals(expected), result


def test__slice_ms1_df():
    df = pd.DataFrame({'retentionTime': [1, 2, 3],
                       'm/z array': [100,200,300], 
                       'intensity array': [2, 3, 7]})
    result = tools.slice_ms1_df(df, rt_min=1.5, rt_max=2.5, 
        mz_mean=200, mz_width=10, intensity_threshold=0)
    expected = df.iloc[[1]]
    assert result.equals(expected), result


def test__integrate_peaks_from_df():
    df = pd.DataFrame({'retentionTime': [1, 2, 3],
                       'm/z array': [100,200,300], 
                       'intensity array': [2, 3, 7]})

    peaklist = pd.DataFrame(
        {'peak_label': ['A'],
         'mz_mean': [200],
         'mz_width': [10],
         'intensity_threshold': [0],
         'rt_min': [0],
         'rt_max': [10]})

    result = tools.integrate_peaks_from_df(df, peaklist)
    print(result)


def test__slice_ms1_array():
    array = np.array([[  1, 100,   2],
                      [  2, 200,   3],
                      [  3, 300,   7]])
    result = tools.slice_ms1_array(array,  rt_min=1.5, rt_max=2.5, 
        mz_mean=200, mz_width=10, intensity_threshold=0)
    expected = np.array([[  2, 200,   3]])
    assert np.array_equal(result, expected), result


def test__integrate_peaks_from_numpy():
    array = np.array([[  1, 100,   2],
                      [  2, 200,   3],
                      [  3, 300,   7]])

    peaks = [(100, 10, .5, 1.5, 0, 'A'), 
             (200, 10, 1.5, 2.5, 0, 'B')]

    results = tools.integrate_peaks_from_numpy(array, peaks)

    expected = [{'peak_area': 2,
                 'peak_max': 2,
                 'peak_min': 2,
                 'peak_mean': 2.0,
                 'peak_rt_of_max': 1,
                 'peak_delta_int': 0,
                 'peak_n_datapoints': 1,
                 'peak_mass_diff_mean': 100.0,
                 'peak_mass_diff_25pc': 100.0,
                 'peak_mass_diff_75pc': 100.0,
                 'peak_shape_rt': '1',
                 'peak_shape_int': '100',
                 'peak_label': 'A'},
                {'peak_area': 3,
                 'peak_max': 3,
                 'peak_min': 3,
                 'peak_mean': 3.0,
                 'peak_rt_of_max': 2,
                 'peak_delta_int': 0,
                 'peak_n_datapoints': 1,
                 'peak_mass_diff_mean': 200.0,
                 'peak_mass_diff_25pc': 200.0,
                 'peak_mass_diff_75pc': 200.0,
                 'peak_shape_rt': '2',
                 'peak_shape_int': '200',
                 'peak_label': 'B'}]