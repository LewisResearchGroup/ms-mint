from mint.Mint import Mint
from sklearn.metrics import r2_score
import pandas as pd

app = Mint()

def test_peakArea():
    app.mzxml.files = ['static/test.mzXML']
    app.peaklist.files = ['static/test_peaklist.csv']
    df_test = pd.read_csv('static/test_peaklist.csv', dtype={'peakLabel': str})
    app.run()
    df = pd.merge(df_test, app.results, on='peakLabel', suffixes=('_real', '_calc'))
    df['peakAreaError'] = df.peakArea_calc - df.peakArea_real
    R2 = r2_score(df.peakArea_real, df.peakArea_calc)
    assert R2 > 0.999
