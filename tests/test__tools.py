import os

from pathlib import Path as P

from ms_mint.tools import MINT_ROOT

TEST_MZXML = os.path.abspath(str(P(MINT_ROOT)/P('../tests/data/test.mzXML')))

def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML),\
        'Test mzXML ({}) not found.'.format(TEST_MZXML)


