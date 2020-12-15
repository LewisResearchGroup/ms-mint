# ms_mint/io.py

import pandas as pd
import numpy as np
import io
import os
import pymzml

from datetime import date
from pyteomics import mzxml, mzml


def ms_file_to_df(fn):
    if fn.lower().endswith('.mzxml'):
        return mzxml_to_pandas_df(fn)
    elif fn.lower().endswith('.mzml'):
        return mzml_to_df(fn)
    elif fn.lower().endswith('hdf'):
        return pd.read_hdf(fn)
    elif fn.lower().endswith('feather'):
        return pd.read_feather(fn)


def mzxml_to_pandas_df(fn):
    '''
    Reads mzXML file and returns a pandas.DataFrame.
    '''
    cols = ['retentionTime', 'm/z array', 'intensity array']
    slices = []
    file = mzxml.MzXML(fn)
    while True:
        try:
            data = file.next()     
            df = pd.DataFrame({col: np.array(data[col]) for col in cols} )
            slices.append( df )
        except:         
            break  
    df = pd.concat(slices)
    df['retentionTime'] =  df['retentionTime'].astype(np.float32)
    df['m/z array'] = df['m/z array'].astype(np.float32)
    df['intensity array'] = df['intensity array'].astype(int)
    df = df.reset_index(drop=True)
    return df


def mzml_to_pandas_df(fn):
    '''
    Reads mzML file and returns a pandas.DataFrame.
    '''
    cols = ['retentionTime', 'm/z array', 'intensity array']
    slices = []
    file = mzml.MzML(fn)
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
    df['intensity array'] = df['intensity array'].astype(int)
    df = df.reset_index(drop=True)
    return df


def mzml_to_df(fn, assume_time_unit='seconds'):
    run = pymzml.run.Reader(fn)
    data = []
    for spectrum in run:
        # Try to convert time units with build-in method
        # some files have no time unit set. Then convert 
        # to minutes assuming the time unit is as set
        # by assume_time_unit argument.
        try:
            RT = spectrum.scan_time_in_minutes()
        except:
            if assume_time_unit == 'seconds':
                RT = spectrum.scan_time[0] / 60.
            elif assume_time_unit == 'minutes':
                RT = spectrum.scan_time[0]
            
        peaks = spectrum.peaks("centroided")
        data.append((RT,peaks))

    df = pd.DataFrame(data).explode(1)

    df['m/z array'] = df[1].apply(lambda x: x[0])
    df['intensity array'] = df[1].apply(lambda x: x[1]).astype(int)
    df = df.rename(columns={0: 'retentionTime'})
    del df[1]
    df = df.reset_index(drop=True)
    return df


def df_to_numeric(df):
    '''
    Converts dataframe to numeric types if possible.
    '''
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='ignore')


def export_to_excel(mint, fn=None):
    date_string = str(date.today())
    if fn is None:
        file_buffer = io.BytesIO()
        writer = pd.ExcelWriter(file_buffer)
    else:
        writer = pd.ExcelWriter(fn)
    # Write into file
    mint.peaklist.to_excel(writer, 'Peaklist', index=False)
    mint.results.to_excel(writer, 'Results', index=False)
    meta = pd.DataFrame({'MINT_version': [mint.version], 
                         'Date': [date_string]}).T[0]
    meta.to_excel(writer, 'Metadata', index=True, header=False)
    # Close writer and maybe return file buffer
    writer.close()
    if fn is None:
        return file_buffer.seek(0)


def convert_ms_file_to_feather(fn, fn_out=None):
    base, ext = os.path.splitext(fn)
    if fn_out is None:
        fn_out = base+'.feather'
    ms_file_to_df(fn).reset_index(drop=True).to_feather(fn_out)
    return fn_out