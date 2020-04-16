from ms_mint.tools import integrate_peaks_from_filename, read_peaklists, MINT_RESULTS_COLUMNS

def check_result(result, peaklist):
    assert list(result.columns) == MINT_RESULTS_COLUMNS, list(result.columns)
    assert (result.peak_label == peaklist.peak_label).all(), result.peak_label
    return True

def test__integrate_peaks_from_filename_mzxml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = integrate_peaks_from_filename('tests/data/test.mzXML', peaklist=peaklist)
    assert check_result(result, peaklist)

def test__integrate_peaks_from_filename_mzml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = integrate_peaks_from_filename('tests/data/test.mzML', peaklist=peaklist)
    assert check_result(result, peaklist)

def test__integrate_peaks_from_filename_broken_mzml():
    peaklist = read_peaklists('tests/data/peaklist_v1.csv')
    result = integrate_peaks_from_filename('tests/data/test.mzML', peaklist=peaklist)
    assert check_result(result, peaklist)