from ms_mint.tools import to_peaks
import pandas as pd

def test__to_peaks():
    peaklist = pd.DataFrame({
                      'mz_mean': [100], 
                      'mz_width': [10],
                      'rt_min': [0.1], 
                      'rt_max': [0.2],
                      'intensity_threshold': [0],
                      'peak_label': ['test']})
    output = to_peaks(peaklist)
    print(output)
    expected = [{'mz_mean': 100, 'mz_width': 10, 'rt_min': 0.1, 'rt_max': 0.2, 'intensity_threshold': 0, 'peak_label': 'test'}]
    assert output == expected, f'Output is \n {output}\n expected is \n {expected}'
