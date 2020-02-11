import io
import os
import pandas as pd
import numpy as np

from datetime import date
from .io import ms_file_to_df

MINT_ROOT = os.path.dirname(__file__)
PEAKLIST_COLUMNS = ['peak_label', 'mz_mean', 'mz_width', 
                    'rt_min', 'rt_max', 'intensity_threshold', 'peaklist']

def read_peaklists(filenames):
    '''
    Extracts peak data from csv files that contain peak definitions.
    CSV files must contain columns: 
        - 'peak_label': str, unique identifier
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
    
    NEW_LABELS = {'peakLabel': 'peak_label',
                  'peakMz': 'mz_mean',
                  'peakMzWidth[ppm]': 'mz_width',
                  'rtmin': 'rt_min',
                  'rtmax': 'rt_max'}
    
    if isinstance(filenames, str):
        filenames = [filenames]

    peaklist = []
    for file in filenames:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, dtype={'peakLabel': str})\
                   .rename(columns=NEW_LABELS)
            df['peaklist'] = file
            if 'intensity_threshold' not in df.columns:
                df['intensity_threshold'] = 0
            df['peak_label'] = df['peak_label'].astype(str)
            peaklist.append(df[PEAKLIST_COLUMNS])
    peaklist = pd.concat(peaklist)
    peaklist.index = range(len(peaklist))
    return peaklist


def integrate_peaks_from_filename(filename, peaklist):
    '''
    Peak integration using a filename as input.
    -----
    Args:
        - filename: str or PosixPath, path to mzxml or mzml filename
        - peaklist: pandas.DataFrame(), DataFrame in peaklist format
    Returns:
        pandas.DataFrame(), DataFrame with integrated peak intensities
    '''
    df = ms_file_to_df(filename)
    peaks = integrate_peaks(df, peaklist)
    peaks['ms_file'] = filename
    return peaks 


def integrate_peaks(df, peaklist):
    '''
    Takes the output of mzxml_to_pandas_df() and
    batch-calculates peak properties.
    '''
    peak_areas = []
    for peak in to_peaks(peaklist):
        peak_area = integrate_peak(df, **peak)
        peak_areas.append(peak_area)
    result = peaklist.copy()
    result['peak_area'] = peak_areas
    return result


def integrate_peak(ms_df, mz_mean, mz_width, rt_min, 
                   rt_max, intensity_threshold, peak_label):
    '''
    Takes the output of ms_file_to_df() and 
    calculates peak properties of a single peak specified by
    the input arguements.
    '''
    peak_area = slice_ms1_mzxml(ms_df, 
                rt_min=rt_min, rt_max=rt_max, 
                mz_mean=mz_mean, mz_width=mz_width, 
                intensity_threshold=intensity_threshold
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


def peak_rt_projection(df, mz_mean, mz_width, rt_min, 
                       rt_max, intensity_threshold, peak_label):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calcualtes the projections of one peak, 
    specicied by the input parameters, onto
    the RT dimension to visualize peak shapes.
    '''
    slizE = slice_ms1_mzxml(df, rt_min=rt_min, rt_max=rt_max, 
                            mz_mean=mz_mean, mz_width=mz_width, 
                            intensity_threshold=intensity_threshold)
    rt_projection = slizE[['retentionTime', 'm/z array', 'intensity array']]\
                    .groupby(['retentionTime', 'm/z array']).sum()\
                    .unstack()\
                    .sum(axis=1)
    # return [mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label, rt_projection]
    return [peak_label, rt_projection]


def to_peaks(peaklist):
    '''
    Takes a dataframe with at least the columns:
    ['mz_mean', 'mz_width', 'rt_min', 'rt_max', 'peak_label'].
    Returns a list of dictionaries that define peaks.
    '''
    cols_to_import = ['mz_mean', 
                      'mz_width',
                      'rt_min', 
                      'rt_max', 
                      'intensity_threshold',
                      'peak_label']
                      
    tmp = [list(i) for i in list(peaklist[cols_to_import].values)]
    output = [{'mz_mean': el[0],
               'mz_width': el[1], 
               'rt_min': el[2],
               'rt_max': el[3], 
               'intensity_threshold': el[4],
               'peak_label': el[5],
               } for el in tmp]
    return output


def slice_ms1_mzxml(df, rt_min, rt_max, mz_mean, mz_width, intensity_threshold):
    '''
    Returns a slize of a metabolomics mzXML file.
    - df: pandas.DataFrame that has columns 
            * 'retentionTime'
            * 'm/z array'
            * 'intensity array'
    - rt_min: minimal retention time
    - rt_max: maximal retention time
    - mz_mean: center of mass (m/z)
    - mz_width: width of the mass window in ppm
    - intensity_threshold: threshold for minimum intensity values
    '''
    delta_mass = mz_width*mz_mean*1e-6
    df_slice = df.loc[(rt_min <= df.retentionTime) &
                      (df.retentionTime <= rt_max) &
                      (mz_mean-delta_mass <= df['m/z array']) & 
                      (df['m/z array'] <= mz_mean+delta_mass) &
                      (df['intensity array'] >= intensity_threshold)]
    return df_slice


def check_peaklist(peaklist):
    '''
    Test if 
    1) peaklist has right type, 
    2) all columns are present and 
    3) dtype of column peak_label is string
    '''
    assert isinstance(peaklist, pd.DataFrame)
    peaklist[PEAKLIST_COLUMNS]
    assert peaklist.dtypes['peak_label'] == np.dtype('O')
    assert peaklist.peak_label.value_counts().max() == 1, peaklist.peak_label.value_counts()


def restructure_rt_projections(data):
    output = {}
    for el in list(data.values())[0]:
        output[el[0]] = {}
    for filename in data.keys():
        for item in data[filename]:
            peaklabel = item[0]
            rt_proj = item[1]
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
    
    if 'queue' in args.keys():
        q = args['queue']
        q.put('filename')
    cols = ['retentionTime', 'm/z array', 'intensity array']
    df = ms_file_to_df(filename=filename)[cols]

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


def generate_grid_peaklist(masses, dt, rt_max=10, mz_ppm=10, intensity_threshold=0):
    '''
    Creates a peaklist from a list of masses.
    -----
    Args:
        - masses: iterable of float values
        - dt: float or int, size of peak windows in time dimension [min]
        - rt_max: float, maximum time [min]
        - mz_ppm: width of peak window in m/z dimension
            mass +/- (mz_ppm * mass * 1e-6)
    '''
    rt_cuts = np.arange(0, rt_max+dt, dt)
    peaklist = pd.DataFrame(index=rt_cuts, columns=masses).unstack().reset_index()
    del peaklist[0]
    peaklist.columns = ['mz_mean', 'rt_min']
    peaklist['rt_max'] = peaklist.rt_min+(1*dt)
    peaklist['peak_label'] =  peaklist.mz_mean.apply(lambda x: '{:.3f}'.format(x)) + '__' + peaklist.rt_min.apply(lambda x: '{:2.2f}'.format(x))
    peaklist['mz_width'] = mz_ppm
    peaklist['intensity_threshold'] = intensity_threshold
    peaklist['peaklist'] = 'Generated'
    check_peaklist(peaklist)
    return peaklist


def export_to_excel(mint, filename=None):
    date_string = str(date.today())
    if filename is None:
        file_buffer = io.BytesIO()
        writer = pd.ExcelWriter(file_buffer)
    else:
        writer = pd.ExcelWriter(filename)
    # Write into file
    mint.results.to_excel(writer, 'MINT', index=False)
    mint.peaklist.to_excel(writer, 'Peaklist', index=False)
    mint.crosstab.T.to_excel(writer, 'PeakArea', index=True)
    meta = pd.DataFrame({'MINT_version': [mint.version], 
                         'Date': [date_string]}).T[0]
    meta.to_excel(writer, 'Metadata', index=True, header=False)
    # Close writer and maybe return file buffer
    writer.close()
    if filename is None:
        return file_buffer.seek(0)