import pandas as pd

from ms_mint.Mint import Mint
from ms_mint.standards import MINT_RESULTS_COLUMNS

from paths import TEST_MZML, TEST_MZXML, TEST_PEAKLIST_FN, TEST_PEAKLIST_FN_V0

mint = Mint(verbose=True)
mint_b = Mint(verbose=True)


class TestClass():
    
    def run_without_files(self):
        mint.run()
    
    def test__add_experimental_data(self):
        assert mint.n_files == 0, mint.n_files
        mint.ms_files = [TEST_MZXML]
        assert mint.n_files == 1, mint.n_files
        assert mint.files == [TEST_MZXML]

    def test__add_peaklist(self):
        assert mint.n_peaklist_files == 0, mint.n_peaklist_files
        mint.peaklist_files = TEST_PEAKLIST_FN
        assert mint.n_peaklist_files == 1, mint.n_peaklist_files
                
    def test__mint_run_standard(self):
        mint.run()
    
    def test__results_is_dataframe(self):
        results = mint.results
        assert isinstance(results, pd.DataFrame), 'Results is not a DataFrame'
    
    def test__results_lenght(self):
        actual = len(mint.results)
        expected = len(mint.peaklist) * len(mint.files)
        assert  expected == actual, f'Length of results ({actual}) does not equal expected length ({expected})'
    
    def test__results_columns(self):
        expected = MINT_RESULTS_COLUMNS
        actual = mint.results.columns
        assert (expected == actual).all(), actual
    
    def test__crosstab_is_dataframe(self):
        ct = mint.crosstab()
        assert isinstance(ct, pd.DataFrame), f'Crosstab is not a DataFrame ({type(ct)}).'

    def test__mint_run_parallel(self):
        mint.files = [TEST_MZML, TEST_MZXML]
        mint.peaklist_files = TEST_PEAKLIST_FN
        mint.run(nthreads=2)   
      
    def test__mzxml_equals_mzml(self):
        mint.reset()
        mint.files = [TEST_MZML, TEST_MZXML]
        mint.peaklist_files = TEST_PEAKLIST_FN
        mint.run()
        results = []
        for _, grp in mint.results.groupby('ms_file'):
            results.append(grp.peak_area.astype(int))

        print(results)
        assert (results[0] == results[1]).all(), results[0]-results[1]

    def test__peaklist_v0_equals_v1(self):
        mint.reset()
        mint.files = [TEST_MZXML]
        mint.peaklist_files = [TEST_PEAKLIST_FN_V0,
                               TEST_PEAKLIST_FN]
        mint.run()
        results = []
        for _, grp in mint.results.groupby('peaklist_name'):
            results.append(grp.peak_area.values)
        assert (results[0] == results[1]).all(), results
    
    def test__status(self):
        mint.status == 'waiting'
    
    def test__run_returns_none_without_peaklist(self):
        mint.reset()
        mint.files = [TEST_MZXML]
        assert mint.run() is None
    
    def test__run_returns_none_without_ms_files(self):
        mint.reset()
        mint.peaklist_files = TEST_PEAKLIST_FN
        assert mint.run() is None
