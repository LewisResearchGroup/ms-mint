import io
import os
import pandas as pd
import numpy as np

from datetime import date
from .io import ms_file_to_df

MINT_ROOT = os.path.dirname(__file__)
PEAKLIST_COLUMNS = ['peak_label', 'mz_mean', 'mz_width', 
                    'rt_min', 'rt_max', 'intensity_threshold', 'peaklist']

def example_peaklist():
    return pd.read_csv(f'{MINT_ROOT}/../tests/data/example_peaklist.csv')

def example_results():
    return pd.read_csv(f'{MINT_ROOT}/../tests/data/example_results.csv')


RESULTS_COLUMNS = ['peak_label', 'peak_area', 'peak_n_datapoints', 'peak_max', 'peak_min', 'peak_median',
    'peak_mean', 'peak_int_first', 'peak_int_last', 'peak_delta_int',
    'peak_rt_of_max', 'peaklist', 'mz_mean', 'mz_width', 'rt_min', 'rt_max', 
    'intensity_threshold', 'peak_shape_rt', 'peak_shape_int']


MINT_RESULTS_COLUMNS = ['peak_label', 'ms_file', 
    'peak_area', 'peak_n_datapoints', 'peak_max', 'peak_min', 'peak_median',
    'peak_mean', 'peak_int_first', 'peak_int_last', 'peak_delta_int',
    'peak_rt_of_max', 'file_size', 'intensity_sum', 'ms_path', 'peaklist', 
    'mz_mean', 'mz_width', 'rt_min', 'rt_max', 'intensity_threshold',
    'peak_shape_rt', 'peak_shape_int'
    ]


def integrate_peaks(ms_data, peaklist):
    
    def base(peak):        
        slizE = slice_ms1_mzxml(ms_data, **peak)

        shape = slizE[['retentionTime', 'm/z array', 'intensity array']]\
                        .groupby(['retentionTime', 'm/z array']).sum()\
                        .unstack()\
                        .sum(axis=1)
        
        if len(shape) == 0:
            results = peak.copy()
            results.update({'peak_area': 0})
            return results
    
        peak_area = np.int64(shape.sum())
        peak_med = np.float64(shape[shape != 0].median())
        peak_avg = np.float64(shape[shape != 0].mean())
        peak_max = np.float64(shape.max())
        peak_min = np.float64(shape.min())

        results = {}

        float_list_to_comma_sep_str = lambda x: ','.join( [ str(np.round(i, 4)) for i in x ] )

        results['peak_shape_rt'] = float_list_to_comma_sep_str( shape.index )
        results['peak_shape_int'] = float_list_to_comma_sep_str ( shape.values )
        results['peak_area']   = peak_area
        results['peak_max']    = peak_max
        results['peak_min']    = peak_min
        results['peak_median'] = peak_med
        results['peak_mean']   = peak_avg
        results['peak_int_first'] = shape.values[0]
        results['peak_int_last'] = shape.values[-1]
        results['peak_delta_int'] = results['peak_int_last'] - results['peak_int_first']
        results['peak_rt_of_max'] = shape[shape == peak_max].index
        results['peak_n_datapoints'] = len(shape)
        
        if len(results['peak_rt_of_max']) > 0:
            results['peak_rt_of_max'] = np.mean(results['peak_rt_of_max'])
        else:
            results['peak_rt_of_max'] = np.nan
                    
        results.update(peak)

        return results
    
    base = np.vectorize(base)
    results = base(to_peaks(peaklist))
    results = pd.merge(pd.DataFrame(list(results)), peaklist[['peaklist', 'peak_label']], on=['peak_label'])
    
    # Make sure all columns are present
    for col in RESULTS_COLUMNS:
            if not col in results.keys():
                results[col] = np.NaN
                
    return results[RESULTS_COLUMNS]

def integrate_peak(ms_data, mz_mean, mz_width, rt_min, rt_max, 
                   intensity_threshold, peak_label):
    peaklist = pd.DataFrame([dict(mz_mean=mz_mean, 
                                  mz_width=mz_width,
                                  rt_min=rt_min,
                                  rt_max=rt_max, 
                                  intensity_threshold=intensity_threshold, 
                                  peak_label=peak_label,
                                  peaklist='single_peak')], index=[0])
    result = integrate_peaks(ms_data, peaklist)
    return result
    


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
        elif str(file).endswith('.xlsx'):
            df = pd.read_excel(file, dtype={'peakLabel': str})\
                   .rename(columns=NEW_LABELS)
        df['peaklist'] = os.path.basename(file)
        if 'intensity_threshold' not in df.columns:
            df['intensity_threshold'] = 0
        df['peak_label'] = df['peak_label'].astype(str)
        peaklist.append(df[PEAKLIST_COLUMNS])
    peaklist = pd.concat(peaklist)
    peaklist.index = range(len(peaklist))
    return peaklist


def format_peaklist(peaklist):
    for col in ['peak_label']:
        peaklist[col] = peaklist[col].astype(str)
    
    for col in ['mz_mean', 'mz_width', 'rt_min', 
                'rt_max', 'intensity_threshold']:
        peaklist[col] = peaklist[col].astype(float)
    return peaklist
        


def make_peaklabel_unambiguous(peaklist):
    cumcounts = peaklist.groupby('peak_label').cumcount()
    peaklist.loc[cumcounts != 0, 'peak_label'] = peaklist.loc[cumcounts != 0, 'peak_label'] + '_'+ cumcounts[cumcounts!=0].astype(str)
    

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
    results = integrate_peaks(df, peaklist)
    results['ms_file'] = filename
    results['ms_path'] = os.path.dirname(filename)
    results['file_size'] = os.path.getsize(filename) / 1024 / 1024
    results['intensity_sum'] = df['intensity array'].sum()    
    return results[MINT_RESULTS_COLUMNS]


def to_peaks(peaklist):
    '''
    Takes a dataframe with at least the columns:
        ['mz_mean', 'mz_width', 
         'rt_min',  'rt_max', 
         'peak_label', 'intensity_threshold'].
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


def slice_ms1_mzxml(df, rt_min, rt_max, mz_mean, mz_width, intensity_threshold, peak_label=None):
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
    Returns a list of strings indicating identified errors.
    If list is empty peaklist is OK.
    '''
    errors = []
    if not isinstance(peaklist, pd.DataFrame):
        errors.append('Peaklist is not a dataframe.')
    peaklist[PEAKLIST_COLUMNS]
    if not peaklist.dtypes['peak_label'] == np.dtype('O'):
        errors.append('Provided peak labels are not strings.')
    if not peaklist.peak_label.value_counts().max() == 1:
        errors.append('Provided peak labels are not unique.')
    return errors


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
    
    if 'queue' in args.keys():
        q = args['queue']
        q.put('filename')
    cols = ['retentionTime', 'm/z array', 'intensity array']
    try:
        df = ms_file_to_df(filename=filename)[cols]
    except:
        return pd.DataFrame()
    results = integrate_peaks(df, peaklist)
    results['ms_file'] = filename
    results['ms_path'] = os.path.dirname(filename)
    results['file_size'] = os.path.getsize(filename) / 1024 / 1024
    results['intensity_sum'] = df['intensity array'].sum()
    
    return results[MINT_RESULTS_COLUMNS]


def generate_grid_peaklist(masses, dt, rt_max=10, 
                           mz_ppm=10, intensity_threshold=0):
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
    #mint.crosstab().T.to_excel(writer, 'PeakArea', index=True)
    meta = pd.DataFrame({'MINT_version': [mint.version], 
                         'Date': [date_string]}).T[0]
    meta.to_excel(writer, 'Metadata', index=True, header=False)
    # Close writer and maybe return file buffer
    writer.close()
    if filename is None:
        return file_buffer.seek(0)
    

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def diff_peaklist(old_pklist, new_pklist):
    df = dataframe_difference(old_pklist, new_pklist)
    df = df[df['_merge'] == 'right_only']
    return df.drop('_merge', axis=1)


def remove_all_zero_columns(df):
    is_zero = df.max() != 0
    is_zero = is_zero[is_zero].index
    return df[is_zero]


def sort_columns_by_median(df):
    cols = df.median().sort_values(ascending=False).index
    return df[cols]