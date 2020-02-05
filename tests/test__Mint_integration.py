import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from ms_mint.Mint import Mint
from ms_mint.tools import check_peaklist

app = Mint()

class TestClass():
    
    def run_without_files(self):
        app.run()
    
    def test__add_experimental_data(self):
        assert app.n_files == 0, app.n_files
        app.files = 'tests/data/test.mzXML'
        assert app.n_files == 1, app.n_files
        assert app.files == ['tests/data/test.mzXML']

    def test__add_peaklist(self):
        assert app.n_peaklist_files == 0, app.n_peaklist_files
        app.peaklist_files = 'tests/data/test_peaklist.csv'
        assert app.peaklist_files == ['tests/data/test_peaklist.csv']
        assert app.n_peaklist_files == 1, app.n_peaklist_files

    def test__app_run_express(self):
        app.run(mode='express')
        assert app.rt_projections is None, app.rt_projections

    def test__app_run_standard(self):
        app.run(mode='standard')
        assert app.rt_projections is not None, app.rt_projections
    
    def test__rt_proj_is_dict(self):
        rt = app.rt_projections
        assert isinstance(rt, dict), type(rt)
    
    def test__rt_proj_keys(self):
        rt = app.rt_projections
        expected = app.peaklist.peak_label.values
        actual = np.array(list(rt.keys()))
        print(f'Expected: {expected}')
        print(f'Actual: {actual}')
        assert len(expected) == len(actual), actual
        assert all([(i in expected) for i in actual])
    
    def test__rt_proj_files(self):
        rt = app.rt_projections
        key = list(rt.keys())[0]
        actual = list(rt[key].keys())
        expected = ['tests/data/test.mzXML']
        print(f'Expected: {expected}')
        print(f'Actual: {actual}')
        assert len(expected) == len(actual), actual
        assert all([(i in expected) for i in actual])
       
    def test__correct_peakAreas(self):
        df_test = pd.read_csv('tests/data/test_peaklist.csv', dtype={'peak_label': str})
        print(app.results.dtypes)
        df = pd.merge(df_test, app.results, on='peak_label', suffixes=('_real', '_calc'))
        R2 = r2_score(df.peak_area_real, df.peak_area_calc)
        assert R2 > 0.999, R2
    
    def test__results_is_dataframe(self):
        results = app.results
        assert isinstance(results, pd.DataFrame), 'Results is not a DataFrame'
    
    def test__results_lenght(self):
        actual = len(app.results)
        expected = len(app.peaklist) * len(app.files)
        assert  expected == actual, f'Length of results ({actual}) does not equal expected length ({expected})'
    
    def test__restults_columns(self):
        expected = ['peak_label', 'mz_mean', 'mz_width', 'rt_min', 'rt_max',
            'intensity_threshold', 'peaklist', 'peak_area', 'ms_file', 'ms_path',
            'file_size', 'intensity_sum']
        actual = app.results.columns
        assert (actual == expected).all(), actual
    
    def test__crosstab_is_dataframe(self):
        ct = app.crosstab
        assert isinstance(ct, pd.DataFrame), f'Crosstab is not a DataFrame ({type(ct)}).'

    def test__app_run_parallel(self):
        app.files = ['tests/data/test.mzXML']*2
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
        app.peaklist = peaklist
        app.run(nthreads=2)
    
    def test__export(self, tmp_path):
        print(app.export())
        filename = os.path.join(tmp_path, 'output.xlsx')
        app.export(filename)
        assert os.path.isfile(filename)