# ms_mint/io.py

import pandas as pd
import numpy as np
import io
import pymzml

from pathlib import Path as P
from datetime import date
from pyteomics import mzxml, mzml


def ms_file_to_df(fn):
    fn = str(fn)
    if fn.lower().endswith('.mzxml'):
        df = mzxml_to_df(fn)
    elif fn.lower().endswith('.mzml'):
        df = mzml_to_df(fn)
    elif fn.lower().endswith('hdf'):
        df = pd.read_hdf(fn)
    elif fn.lower().endswith('feather'):
        df = pd.read_feather(fn)
    # Compatibility with old schema
    df = df.rename(columns={
            'retentionTime': 'scan_time_min', 
            'intensity array': 'intensity', 
            'm/z array': 'mz'})
    return df




def mzxml_to_df(fn):
    '''
    Reads mzXML file and returns a pandas.DataFrame.
    '''    
    slices = []
    
    with mzxml.MzXML( fn ) as ms_data:
        data = [x for x in ms_data]
    data = list( extract_mzxml(data) )
                
    df = pd.DataFrame.from_dict( data )\
           .set_index('retentionTime')\
           .apply(pd.Series.explode).reset_index()

    df['retentionTime'] =  df['retentionTime'].astype(np.float64)
    df['m/z array'] = df['m/z array'].astype(np.float64)
    df['intensity array'] = df['intensity array'].astype(int)

    df = df.rename(columns={'num': 'scan_id', 
                            'msLevel': 'ms_level', 
                            'retentionTime': 'scan_time_min', 
                            'm/z array': 'mz', 
                            'intensity array': 'intensity'})
    
    df = df.reset_index(drop=True)
    cols = ['scan_id', 'ms_level', 'polarity', 
            'scan_time_min', 'mz', 'intensity']
    df = df[cols]
    return df


def _extract_mzxml(data):
    cols = ['num', 'msLevel', 'polarity', 
            'retentionTime', 'm/z array', 
            'intensity array']
    return {c: data[c] for c in cols}

extract_mzxml = np.vectorize( _extract_mzxml )


def mzml_to_pandas_df_pyteomics(fn):
    '''
    Reads mzML file and returns a pandas.DataFrame. (deprecated)
    '''
    cols = ['retentionTime', 'm/z array', 'intensity array']
    slices = []
    with  mzml.MzML(fn) as ms_data:
        while True:
            try:
                data = ms_data.next()
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
    
    with pymzml.run.Reader(fn) as ms_data:
        data = [x for x in ms_data]
        
    data = list( extract_mzml(data, assume_time_unit=assume_time_unit) )
    
    df = pd.DataFrame.from_dict( data )\
           .set_index(['scan_id', 'ms_level', 'polarity', 'scan_time_min'])\
           .apply(pd.Series.explode)\
           .reset_index()
    df['mz'] = df['mz'].astype('float64')
    df['intensity'] = df['intensity'].astype('int64')
    return df


def _extract_mzml(data, assume_time_unit):
    try:
        RT = data.scan_time_in_minutes()
    except:
        if assume_time_unit == 'seconds':
            RT = data.scan_time[0] / 60.
        elif assume_time_unit == 'minutes':
            RT = data.scan_time[0]
    peaks = data.peaks("centroided")
        
    return {'scan_id': data.index, 
            'ms_level': data.ms_level, 
            'polarity': '+' if data["positive scan"] else '-', 
            'scan_time_min': RT, 
            'mz': peaks[:,0].astype('float64'),
            'intensity': peaks[:,1].astype('int64')}

extract_mzml = np.vectorize( _extract_mzml )


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
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix('.feather')
    df = ms_file_to_df(fn).reset_index(drop=True)
    df.to_feather(fn_out)
    return fn_out