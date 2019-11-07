from mint.Mint import Mint
from sklearn.metrics import r2_score
import pandas as pd


app = Mint()

class TestClass():
     
    def test__add_experimental_data(self):
        app.mzxmls = ['tests/data/test.mzXML']

    def test__add_peaklist(self):
        app.peaklists = ['tests/data/test_peaklist.csv']

    def test__app_run(self):
        app.run()

    def test__correct_peakAreas(self):
        df_test = pd.read_csv('tests/data/test_peaklist.csv', dtype={'peakLabel': str})
        df = pd.merge(df_test, app.results, on='peakLabel', suffixes=('_real', '_calc'))
        R2 = r2_score(df.peakArea_real, df.peakArea_calc)
        assert R2 > 0.999
            
    
    
