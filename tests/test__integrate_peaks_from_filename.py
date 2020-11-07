from ms_mint.processing import process_ms1_file
from ms_mint.peaklists import read_peaklists
from ms_mint.standards import MINT_RESULTS_COLUMNS

def check_result(result, peaklist):
    assert list(result.columns) == MINT_RESULTS_COLUMNS, list(result.columns)
    assert (result.peak_label == peaklist.peak_label).all(), result.peak_label
    return True

def test__process_ms1_file_mzxml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = process_ms1_file('tests/data/test.mzXML', peaklist=peaklist)
    assert check_result(result, peaklist)

def test__process_ms1_file_mzml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = process_ms1_file('tests/data/test.mzML', peaklist=peaklist)
    assert check_result(result, peaklist)

def test__process_ms1_file_broken_mzml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = process_ms1_file('tests/data/test.mzML', peaklist=peaklist)
    assert check_result(result, peaklist)