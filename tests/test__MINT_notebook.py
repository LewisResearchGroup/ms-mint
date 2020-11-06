import os
import pandas as pd
from sklearn.metrics import r2_score

from ms_mint.notebook import Mint
from ms_mint.peaklists import check_peaklist
from ms_mint.processing import MINT_RESULTS_COLUMNS

mint = Mint(verbose=True)

class TestClass():
    def test__mint_n_files(self):
        ms_files = ['tests/data/ms_files/fileA.mzXML', 
                    'tests/data/ms_files/fileB.mzxml', 
                    'tests/data/ms_files/fileC.mzML',
                    'tests/data/ms_files/fileD.mzml']
        mint.ms_files = ms_files
        result = mint.n_files
        expect = len(ms_files)
        assert result == expect, f'Expected ({expect}) != result ({result})'

    def test__mint_list(self):
        mint.list_files()
        result = mint.message_box.value
        expect = ('\n'.join(['4 MS-files to process:',
                             'tests/data/ms_files/fileA.mzXML',
                             'tests/data/ms_files/fileB.mzxml',
                             'tests/data/ms_files/fileC.mzML',
                             'tests/data/ms_files/fileD.mzml',
                             '',
                             'Using peak list:',
                             '',
                             'No peaklist defined.']))
        assert expect == result, result

