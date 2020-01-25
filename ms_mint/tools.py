import os
import pandas as pd
import numpy as np

from pyteomics import mzxml, mzml
from pathlib import Path as P

from scipy.optimize import curve_fit


MINT_ROOT = os.path.dirname(__file__)
STANDARD_PEAKFILE = os.path.abspath(str(P(MINT_ROOT)/P('../static/Standard_Peaklist.csv')))

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def read_peaklists(filenames):
    '''
    Extracts peak data from csv files that contain peak definitions.
    CSV files must contain columns: 
        - 'peakLabel': str, unique identifier
        - 'peakMz': float, center of mass to be extracted in [Da]
        - 'peakMzWidth[ppm]': float, with of mass window in [ppm]
        - 'rtmin': float, minimum retention time in [min]
        - 'rtmax': float, maximum retention time in [min]
    -----
    Args:
        - filenames: str or PosixPath or list of such with path to csv-file(s)
    Returns:
        pandas.DataFrame in peaklist format
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
    '''
    Peak integration using a filename as input.
    -----
    Args:
        - mzxml: str or PosixPath, path to mzxml filename
        - peaklist: pandas.DataFrame(), DataFrame in peaklist format
    Returns:
        pandas.DataFrame(), DataFrame with integrated peak intensities
    '''
    df = mzxml_to_pandas_df(mzxml)
    peaks = integrate_peaks(df, peaklist)
    peaks['msFile'] = mzxml
    return peaks 


def integrate_peaks(df, peaklist=STANDARD_PEAKLIST):
    '''
    Takes the output of mzxml_to_pandas_df() and
    batch-calculates peak properties.
    '''
    peak_areas = []
    for peak in to_peaks(peaklist):
        peak_area = integrate_peak(df, **peak)
        peak_areas.append(peak_area)
    result = peaklist.copy()
    result['peakArea'] = peak_areas
    return result


def integrate_peak(mzxml_df, mz, dmz, rt_min, rt_max, peak_label):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calculates peak properties of a single peak specified by
    the input arguements.
    '''
    peak_area = slice_ms1_mzxml(mzxml_df, 
                rtmin=rt_min, rtmax=rt_max, mz=mz, dmz=dmz
                )['intensity array'].sum()       
    return peak_area


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


def peak_rt_projection(df, mz, dmz, rt_min, rt_max, peak_label):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calcualtes the projections of one peak, 
    specicied by the input parameters, onto
    the RT dimension to visualize peak shapes.
    '''
    slizE = slice_ms1_mzxml(df, rtmin=rt_min, rtmax=rt_max, mz=mz, dmz=dmz)
    rt_projection = slizE[['retentionTime', 'm/z array', 'intensity array']]\
                    .groupby(['retentionTime', 'm/z array']).sum()\
                    .unstack()\
                    .sum(axis=1)
    return [mz, dmz, rt_min, rt_max, peak_label, rt_projection]


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
               'rt_min': el[2],
               'rt_max': el[3], 
               'peak_label': el[4]} for el in tmp]
    return output


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
    delta_mass = dmz*mz*1e-6
    df_slice = df.loc[(rtmin <= df.retentionTime) &
                      (df.retentionTime <= rtmax) &
                      (mz-delta_mass <= df['m/z array']) & 
                      (df['m/z array'] <= mz+delta_mass)]
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
 'peak_label', 'mz', 'mz_width_ppm','rt_min', 'rt_max'"
    return True


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


def process(args):
    '''
    Pickleable function for (parallel) peak integration.
    Expects a dictionary with keys:
        Mandatory:
        - 'filename': 'path to file to be processed',
        - 'peaklist': 'dataframe containing the peaklist' 
        - 'mode': 'express' or 'standard'
            * 'express' omits calculcation of rt projections
        Optional:
        - 'queue': instance of multiprocessing.Manager().Queue()

    Returns tuple with two elements:
        1) results, dataframe with integration results
        2) rt_projection, dictionary of dictionaries with peak shapes
    '''
    filename = args['filename']
    peaklist = args['peaklist']
    mode = args['mode']
    intensity_threshold = args['intensity_threshold']
    
    if 'queue' in args.keys():
        q = args['queue']
        q.put('filename')
    cols = ['retentionTime', 'm/z array', 'intensity array']
    df = mzxml_to_pandas_df(filename=filename)[cols]
    if intensity_threshold > 0:
        df = df[df['intensity array'] >= intensity_threshold]
    result = integrate_peaks(df, peaklist)
    result['ms_file'] = filename
    result['ms_path'] = os.path.dirname(filename)
    result['file_size'] = os.path.getsize(filename) / 1024 / 1024
    result['intensity_sum'] = df['intensity array'].sum()
    if mode == 'standard':
        rt_projection = {filename: peak_rt_projections(df, peaklist)}
        return result, rt_projection
    elif mode == 'express':
        return result, None


def peaklist_from_masses_and_rt_grid(masses, dt, rt_max=10, mz_ppm=10):
    rt_cuts = np.arange(0, rt_max+dt, dt)
    peaklist = pd.DataFrame(index=rt_cuts, columns=masses).unstack().reset_index()
    del peaklist[0]
    peaklist.columns = ['peakMz', 'rtmin']
    peaklist['rtmax'] = peaklist.rtmin+(1*dt)
    peaklist['peakLabel'] =  peaklist.peakMz.apply(lambda x: '{:.3f}'.format(x)) + '__' + peaklist.rtmin.apply(lambda x: '{:2.2f}'.format(x))
    peaklist['peakMzWidth[ppm]'] = mz_ppm
    return peaklist