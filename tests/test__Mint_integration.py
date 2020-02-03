import pandas as pd

from sklearn.metrics import r2_score

from ms_mint.Mint import Mint
from ms_mint.tools import STANDARD_PEAKFILE, check_peaklist

app = Mint()
app.peaklist_files = STANDARD_PEAKFILE

class TestClass():
    def test__add_experimental_data(self):
        app.files = ['tests/data/test.mzXML']

    def test__add_peaklist(self):
        app.peaklist_files = ['tests/data/test_peaklist.csv']

    def test__app_run(self):
        app.run()

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