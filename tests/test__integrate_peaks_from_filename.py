import pandas as pd

from ms_mint.processing import process_ms1_file
from ms_mint.standards import MINT_RESULTS_COLUMNS

from paths import TEST_MZML, TEST_MZXML


peaklist = pd.DataFrame({
    'mz_mean': [117.0188], 
    'mz_width': [100],
    'rt_min': [0], 
    'rt_max': [15],
    'rt': [2.3],
    'intensity_threshold': [0],
    'peak_label': ['Succinate-neg'],
    'peaklist_name': ['no-file']})


def check_result(result, peaklist):
    assert list(result.columns) == MINT_RESULTS_COLUMNS, list(result.columns)
    assert (result.peak_label == peaklist.peak_label).all(), result.peak_label
    return True


def test__process_ms1_file_mzxml():
    result = process_ms1_file(TEST_MZXML, peaklist=peaklist)
    assert check_result(result, peaklist)


def test__process_ms1_file_mzml():
    result = process_ms1_file(TEST_MZML, peaklist=peaklist)
    assert check_result(result, peaklist)
