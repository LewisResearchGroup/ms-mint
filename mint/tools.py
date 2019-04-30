import os
import pandas as pd

from pyteomics import mzxml
from pathlib import Path as P

MINT_ROOT = os.path.dirname(__file__)
STANDARD_PEAKFILE = os.path.abspath(str(P(MINT_ROOT)/P('../static/Standard_Peaklist.csv')))

def read_peaklists(filenames):
    '''
    Extracts peak data from csv file.
    '''
    if isinstance(filenames, str):
        filenames = [filenames]
    peaklist = []
    cols_to_import = ['peakLabel',
                      'peakMz',
                      'peakMzWidth[ppm]',
                      'rtmin',
                      'rtmax']
    for file in filenames:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, usecols=cols_to_import,
                             dtype={'peakLabel': str})
            df['peakListFile'] = file
            peaklist.append(df)
    peaklist = pd.concat(peaklist)
    peaklist.index = range(len(peaklist))
    return peaklist

STANDARD_PEAKLIST = read_peaklists(STANDARD_PEAKFILE)
DEVEL = True

def integrate_peaks_from_filename(mzxml, peaklist=STANDARD_PEAKLIST):
    df = mzxml_to_pandas_df(mzxml)
    peaks = integrate_peaks(df, peaklist)
    peaks['mzxmlFile'] = mzxml
    return peaks 

def integrate_peaks(df, peaklist=STANDARD_PEAKLIST):
    '''
    Takes the output of mzxml_to_pandas_df() and
    batch-calculates peak properties.
    '''
    results = []
    for peak in to_peaks(peaklist):
        result = integrate_peak(df, **peak)
        results.append(result)
    results = pd.concat(results)
    results.index = range(len(results))
    return pd.merge(peaklist, results, right_index=True, left_index=True)

def integrate_peak(mzxml_df, mz, dmz, rtmin, rtmax, peaklabel):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calculates peak properties of one peak specified by
    the input arguements.
    '''
    slizE = slice_ms1_mzxml(mzxml_df, 
                rtmin=rtmin, rtmax=rtmax, mz=mz, dmz=dmz
                )
    peakArea = slizE['intensity array'].sum()
    result = pd.DataFrame({'peakLabel': peaklabel,
                           'rtmin': [rtmin], 
                           'rtmax': [rtmax],
                           'peakMz': [mz],
                           'peakMzWidth[ppm]': [dmz],
                           'peakArea': [peakArea]}
                         )
    return result[['peakArea']]

def peak_rt_projections(df, peaklist):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    batch-calcualtes the projections of peaks onto
    the RT dimension to visualize peak shapes.
    '''
    peaklist.index = range(len(peaklist))
    results = []
    for peak in to_peaks(peaklist):
        result = peak_rt_projection(df, **peak)
        results.append(result)
    return results

def peak_rt_projection(df, mz, dmz, rtmin, rtmax, peaklabel):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calcualtes the projections of one peak, 
    specicied by the input parameters, onto
    the RT dimension to visualize peak shapes.
    '''
    slizE = slice_ms1_mzxml(df, rtmin=rtmin, rtmax=rtmax, mz=mz, dmz=dmz)
    rt_projection = slizE[['retentionTime', 'm/z array', 'intensity array']]\
                    .set_index(['retentionTime', 'm/z array'])\
                    .unstack()\
                    .sum(axis=1)
    return [mz, dmz, rtmin, rtmax, peaklabel, rt_projection]

def to_peaks(peaklist):
    '''
    Takes a dataframe with at least the columns:
    ['peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax', 'peakLabel'].
    Returns a list of dictionaries that define peaks.
    '''
    cols_to_import = ['peakMz', 
                      'peakMzWidth[ppm]',
                      'rtmin', 
                      'rtmax', 
                      'peakLabel']
    tmp = [list(i) for i in list(peaklist[cols_to_import].values)]
    output = [{'mz': el[0],
               'dmz': el[1], 
               'rtmin': el[2],
               'rtmax': el[3], 
               'peaklabel': el[4]} for el in tmp]
    return output

def mzxml_to_pandas_df(filename):
    '''
    Reads mzXML file and returns a pandas.DataFrame.
    '''
    slices = []
    file = mzxml.MzXML(filename)
    while True:
        try:
            slices.append(pd.DataFrame(file.next()))
        except:
            break
    df = pd.concat(slices)
    df_to_numeric(df)
    return df

def df_to_numeric(df):
    '''
    Converts dataframe to numeric types if possible.
    '''
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='ignore')

def slice_ms1_mzxml(df, rtmin, rtmax, mz, dmz):
    '''
    Returns a slize of a metabolomics mzXML file.
    df - pandas.DataFrame that has columns 
            * 'retentionTime'
            * 'm/z array'
            * 'rtmin'
            * 'rtmax'
    rtmin - minimal retention time
    rtmax - maximal retention time
    mz - center of mass (m/z)
    dmz - width of the mass window in ppm
    '''
    df_slice = df.loc[(rtmin <= df.retentionTime) &
                      (df.retentionTime <= rtmax) &
                      (mz-0.0001*dmz <= df['m/z array']) & 
                      (df['m/z array'] <= mz+0.0001*dmz)]
    return df_slice

def check_peaklist(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Cannot find peaklist file ({filename}).')
    try:
        df = pd.read_csv(P(filename))
    except:
        return f'Cannot open peaklist {filename}'
    try:
        df[['peakLabel', 'peakMz', 'peakMzWidth[ppm]','rtmin', 'rtmax']]
    except:
        return f"Not all columns found.\n\
 Please make sure the peaklist file has at least:\
 'peakLabel', 'peakMz', 'peakMzWidth[ppm]','rtmin', 'rtmax'"
    return f'Peaklist file ok ({filename})'

def restructure_rt_projections(data):
    output = {}
    for el in list(data.values())[0]:
        output[el[4]] = {}
    for filename in data.keys():
        for item in data[filename]:
            peaklabel = item[4]
            rt_proj = item[5]
            output[peaklabel][filename] = rt_proj
    return output

