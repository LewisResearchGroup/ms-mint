import pandas as pd
from pyteomics import mzxml, mzml

def ms_file_to_df(filename):
    if filename.lower().endswith('.mzxml'):
        return mzxml_to_pandas_df(filename)
    elif  filename.lower().endswith('.mzml'):
        return mzml_to_pandas_df(filename)
    
def mzxml_to_pandas_df(filename):
    '''
    Reads mzXML file and returns a pandas.DataFrame.
    '''
    cols = ['retentionTime', 'm/z array', 'intensity array']
    slices = []
    file = mzxml.MzXML(filename)
    while True:
        try:
            slices.append( pd.DataFrame(file.next()) ) 
        except:
            break
    df = pd.concat(slices)[cols]
    df_to_numeric(df)
    return df
    
def mzml_to_pandas_df(filename):
    '''
    Reads mzML file and returns a pandas.DataFrame.
    '''
    cols = ['retentionTime', 'm/z array', 'intensity array']
    slices = []
    file = mzml.MzML(filename)
    while True:
        try:
            data = file.next()
            data['retentionTime'] = data['scanList']['scan'][0]['scan time'] / 60
            del data['scanList']
            slices.append( pd.DataFrame(data) ) 
        except:
            break
    df = pd.concat(slices)[cols]
    df_to_numeric(df)
    return df

def df_to_numeric(df):
    '''
    Converts dataframe to numeric types if possible.
    '''
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='ignore')