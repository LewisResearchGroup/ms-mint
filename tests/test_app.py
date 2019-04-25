from miiit.miiit import *
from sklearn.metrics import r2_score

app = App()

def test_peakArea():
    app.mzxml.files = ['tests/test1.mzXML']
    app.peaklist.files = ['tests/test1_peaklist.csv']
    df_test = pd.read_csv('tests/test1_peaklist.csv', dtype={'peakLabel': str})
    app.run()
    df = pd.merge(df_test, app.results, on='peakLabel', suffixes=('_real', '_calc'))
    x = df.peakArea_real
    y = df.peakArea_calc
    df['peakAreaError'] = df.peakArea_calc - df.peakArea_real
    R2 = r2_score(df.peakArea_real, df.peakArea_calc)
    assert R2 > 0.999
    #assert max_error < 0.001
