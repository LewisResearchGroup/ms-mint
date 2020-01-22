from ms_mint.tools import to_peaks
import pandas as pd

def test__to_peaks():
    peaklist = pd.DataFrame({
                      'peakMz': [100], 
                      'peakMzWidth[ppm]': [10],
                      'rtmin': [0.1], 
                      'rtmax': [0.2], 
                      'peakLabel': ['test']})
    output = to_peaks(peaklist)
    expected = [{'mz': 100, 'dmz': 10, 'rtmin': 0.1, 'rtmax': 0.2, 'peaklabel': 'test'}]
    assert output == expected, f'Output is \n {output}\n expected is \n {expected}'
