import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from ms_mint.Mint import Mint
from ms_mint.tools import check_peaklist

mint = Mint(verbose=True)

class TestClass():
    
    def run_without_files(self):
        mint.run()
    
    def test__add_experimental_data(self):
        assert mint.n_files == 0, mint.n_files
        mint.files = ['tests/data/test.mzXML']
        assert mint.n_files == 1, mint.n_files
        assert mint.files == ['tests/data/test.mzXML']

    def test__add_peaklist(self):
        assert mint.n_peaklist_files == 0, mint.n_peaklist_files
        mint.peaklist_files = 'tests/data/test_peaklist.csv'
        assert mint.peaklist_files == ['tests/data/test_peaklist.csv']
        assert mint.n_peaklist_files == 1, mint.n_peaklist_files
                  
    def test__mint_run_express(self):
        mint.run(mode='express')
        assert mint.rt_projections is None, mint.rt_projections

    def test__export_to_excel(self, tmp_path):
        filename = os.path.join(tmp_path, 'output.xlsx')
        mint.export(filename)
        assert os.path.isfile(filename)

    def test__mint_run_standard(self):
        mint.run(mode='standard')
        assert mint.rt_projections is not None, mint.rt_projections
    
    def test__rt_proj_is_dict(self):
        rt = mint.rt_projections
        assert isinstance(rt, dict), type(rt)
    
    def test__rt_proj_keys(self):
        rt = mint.rt_projections
        expected = mint.peaklist.peak_label.values
        actual = np.array(list(rt.keys()))
        print(f'Expected: {expected}')
        print(f'Actual: {actual}')
        assert len(expected) == len(actual), actual
        assert all([(i in expected) for i in actual])
    
    def test__rt_proj_files(self):
        rt = mint.rt_projections
        key = list(rt.keys())[0]
        actual = list(rt[key].keys())
        expected = ['tests/data/test.mzXML']
        print(f'Expected: {expected}')
        print(f'Actual: {actual}')
        assert len(expected) == len(actual), actual
        assert all([(i in expected) for i in actual])
       
    def test__correct_peakAreas(self):
        df_test = pd.read_csv('tests/data/test_peaklist.csv', dtype={'peak_label': str})
        print(mint.results.dtypes)
        df = pd.merge(df_test, mint.results, on='peak_label', suffixes=('_real', '_calc'))
        R2 = r2_score(df.peak_area_real, df.peak_area_calc)
        assert R2 > 0.999, R2
    
    def test__results_is_dataframe(self):
        results = mint.results
        assert isinstance(results, pd.DataFrame), 'Results is not a DataFrame'
    
    def test__results_lenght(self):
        actual = len(mint.results)
        expected = len(mint.peaklist) * len(mint.files)
        assert  expected == actual, f'Length of results ({actual}) does not equal expected length ({expected})'
    
    def test__restults_columns(self):
        expected = ['peak_label', 'mz_mean', 'mz_width', 'rt_min', 'rt_max',
            'intensity_threshold', 'peaklist', 'peak_area', 'ms_file', 'ms_path',
            'file_size', 'intensity_sum']
        actual = mint.results.columns
        assert (actual == expected).all(), actual
    
    def test__crosstab_is_dataframe(self):
        ct = mint.crosstab
        assert isinstance(ct, pd.DataFrame), f'Crosstab is not a DataFrame ({type(ct)}).'

    def test__mint_run_parallel(self):
        mint.files = ['tests/data/test.mzXML']*2
        peaklist = pd.DataFrame({
                      'mz_mean': [100], 
                      'mz_width': [10],
                      'rt_min': [0.1], 
                      'rt_max': [0.2],
                      'intensity_threshold': [0],
                      'peak_label': ['test'],
                      'peaklist': ['no-file']})
        print(peaklist)
        check_peaklist(peaklist)
        mint.peaklist = peaklist
        mint.run(nthreads=2)
    
    def test__export(self, tmp_path):
        print(mint.export())
        filename = os.path.join(tmp_path, 'output.xlsx')
        mint.export(filename)
        assert os.path.isfile(filename)
        
    def test__mzxml_equals_mzml(self):
        mint.reset()
        mint.files = ['tests/data/test.mzXML' , 'tests/data/test.mzML']
        mint.peaklist_files = 'tests/data/peaklist_v1.csv'
        mint.run()
        results = []
        for _, grp in mint.results.groupby('ms_file'):
            results.append(grp.peak_area)
        assert (results[0] == results[1]).all(), results

    def test__peaklist_v0_equals_v1(self):
        mint.reset()
        mint.files = ['tests/data/test.mzXML']
        mint.peaklist_files = ['tests/data/peaklist_v0.csv', 'tests/data/peaklist_v1.csv']
        mint.run()
        results = []
        for _, grp in mint.results.groupby('peaklist'):
            results.append(grp.peak_area.values)
        assert (results[0] == results[1]).all(), results
    
    def test__status(self):
        mint.status == 'waiting'
    
    def test__run_returns_none_without_peaklist(self):
        mint.reset()
        mint.files = ['tests/data/test.mzXML']
        assert mint.run() is None
    
    def test__run_returns_none_without_ms_files(self):
        mint.reset()
        mint.peaklist_files = 'tests/data/peaklist_v0.csv'
        assert mint.run() is None
        
