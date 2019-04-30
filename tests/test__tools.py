import os
from mint.tools import MINT_ROOT, slice_ms1_mzxml,\
    integrate_peaks_from_filename, STANDARD_PEAKLIST
import pandas as pd
from pathlib import Path as P

TEST_MZXML = os.path.abspath(str(P(MINT_ROOT)/P('../static/test.mzXML')))

def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML),\
        'Test mzXML ({}) not found.'.format(TEST_MZXML)

def test__integrate_peaks_from_filename():
    result = integrate_peaks_from_filename(TEST_MZXML)
    assert isinstance(result, pd.DataFrame)
    assert len(STANDARD_PEAKLIST) == len(result)
