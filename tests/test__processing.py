import os
import pandas as pd 
import numpy as np

from ms_mint import processing

from paths import TEST_MZXML


def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML),\
        'Test mzXML ({}) not found.'.format(TEST_MZXML)


def test__process_ms1():
    ms_data = pd.DataFrame({'scan_time_min': [1, 2, 3],
                            'mz': [100, 200, 300], 
                            'intensity': [2, 3, 7]})

    peaklist = pd.DataFrame(
        {'peak_label': ['A'],
         'mz_mean': [200],
         'mz_width': [10],
         'intensity_threshold': [0],
         'rt_min': [0],
         'rt_max': [10]})

    result = processing.process_ms1(ms_data, peaklist)

    expect = pd.DataFrame({  'peak_label': {0: 'A'},
                             'mz_mean': {0: 200},
                             'mz_width': {0: 10},
                             'intensity_threshold': {0: 0},
                             'rt_min': {0: 0},
                             'rt_max': {0: 10},
                             'peak_area': {0: 3},
                             'peak_n_datapoints': {0: 1},
                             'peak_max': {0: 3},
                             'peak_rt_of_max': {0: 2},
                             'peak_min': {0: 3},
                             'peak_median': {0: 3.0},
                             'peak_mean': {0: 3.0},
                             'peak_delta_int': {0: 0},
                             'peak_shape_rt': {0: '2'},
                             'peak_shape_int': {0: '3'},
                             'peak_mass_diff_25pc': {0: 0.0},
                             'peak_mass_diff_50pc': {0: 0.0},
                             'peak_mass_diff_75pc': {0: 0.0},
                             'peak_score': {0: None} 
                        })
    
    assert result.equals(expect), result


def test__process_ms1_from_df():
    df = pd.DataFrame({'scan_time_min': [1, 2, 3],
                       'mz': [100,200,300], 
                       'intensity': [2, 3, 7]})
    peaklist = pd.DataFrame(
        {'peak_label': ['A'],
         'mz_mean': [200],
         'mz_width': [10],
         'intensity_threshold': [0],
         'rt_min': [0],
         'rt_max': [10]})
    result = processing.process_ms1_from_df(df, peaklist)
    expect= [['A', 3, 1, 3, 2, 3, 3.0, 3.0, 0, '2', '3', 0.0, 0.0, 0.0, None]]
    print(result)
    print(expect) 
    assert result == expect


def test__slice_ms1_array():
    array = np.array([[  1, 100,   2],
                      [  2, 200,   3],
                      [  3, 300,   7]])
    result = processing.slice_ms1_array(array,  rt_min=1.5, rt_max=2.5, 
        mz_mean=200, mz_width=10, intensity_threshold=0)
    expect = np.array([[  2, 200,   3]])
    print(expect)
    print(result)
    assert np.array_equal(result, expect)


def test__process_ms1_from_numpy():
    array = np.array([[  1, 100,   2],
                      [  2, 200,   3],
                      [  3, 300,   7]])

    peaks = [(100, 10, .5, 1.5, 0, 'A'), 
             (200, 10, 1.5, 2.5, 0, 'B')]

    result = processing.process_ms1_from_numpy(array, peaks)

    expect = [['A', 2, 1, 2, 1, 2, 2.0, 2.0, 0, '1', '2', 0.0, 0.0, 0.0, None],
              ['B', 3, 1, 3, 2, 3, 3.0, 3.0, 0, '2', '3', 0.0, 0.0, 0.0, None]]
    
    print('Expected:')
    [print(i) for i in expect]
    print('Actual')
    [print(i) for i in result]
    assert result == expect