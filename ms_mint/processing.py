# ms_mint/processing.py

import os
import pandas as pd
import numpy as np

from .io import ms_file_to_df

from .standards import RESULTS_COLUMNS,\
    MINT_RESULTS_COLUMNS


#
#def example_peaklist():
#    return pd.read_csv(f'{MINT_ROOT}/../tests/data/example_peaklist.csv')
#
#def example_results():
#    return pd.read_csv(f'{MINT_ROOT}/../tests/data/example_results.csv')
#


# The controls the order of columns in mint.results

def process_ms1_files_in_parallel(args):
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
    try:
        return process_ms1_file(filename=filename, peaklist=peaklist)
    except:
        return pd.DataFrame()


def process_ms1_file(filename, peaklist):
    '''
    Peak integration using a filename as input.
    -----
    Args:
        - filename: str or PosixPath, path to mzxml or mzml filename
        - peaklist: pandas.DataFrame(), DataFrame in peaklist format
    Returns:
        pandas.DataFrame(), DataFrame with processd peak intensities
    '''
    df = ms_file_to_df(filename)
    results = process_ms1(df, peaklist)
    results['total_intensity'] = df['intensity array'].sum()
    results['ms_file'] = filename
    results['ms_path'] = os.path.dirname(filename)
    results['ms_file_size'] = os.path.getsize(filename) / 1024 / 1024
    return results[MINT_RESULTS_COLUMNS]


def process_ms1(df, peaklist):
    results = process_ms1_from_df(df, peaklist)
    results = pd.DataFrame(results, columns=['peak_label']+RESULTS_COLUMNS)
    results = pd.merge(peaklist, results, on=['peak_label'])
    results = results.reset_index(drop=True)
    # Make sure all columns are present
    #for col in RESULTS_COLUMNS:
    #        if not col in results.keys():
    #            results[col] = np.NaN
    return results


def process_ms1_from_df(df, peaklist):
    peak_cols = ['mz_mean', 'mz_width', 'rt_min', 'rt_max', 
                 'intensity_threshold', 'peak_label']
    array_peaks = peaklist[peak_cols].values
    array_data = df[['retentionTime', 'm/z array', 'intensity array']].values
    result = process_ms1_from_numpy(array_data, array_peaks)
    return result


def process_ms1_from_numpy(array, peaks):
    results = []
    for (mz_mean, mz_width, rt_min, rt_max, 
         intensity_threshold, peak_label) in peaks:
        props = _process_ms1_from_numpy(array, mz_mean=mz_mean, 
                    mz_width=mz_width, rt_min=rt_min, rt_max=rt_max, 
                    intensity_threshold=intensity_threshold, 
                    peak_label=peak_label)
        if props is None:
            continue
        results.append(
            [ props[col] for col in ['peak_label']+RESULTS_COLUMNS ]
        )
    return results


def _process_ms1_from_numpy(array, mz_mean, mz_width, rt_min, rt_max, 
                              intensity_threshold, peak_label=None):
    _slice = slice_ms1_array(array=array, mz_mean=mz_mean, mz_width=mz_width,
                             rt_min=rt_min, rt_max=rt_max, 
                             intensity_threshold=intensity_threshold)
    props = extract_ms1_properties(_slice)
    if props is None:
        return
    if peak_label is not None:
        props['peak_label'] = peak_label
    return props


def extract_ms1_properties(array):
    float_list_to_comma_sep_str = \
        lambda x: ','.join( [ str(np.round(i, 4)) for i in x ] )

    projection = array[:,[0,1]]
    times = array[:,0]
    masses = array[:,1]
    intensities = array[:,2]
    peak_n_datapoints = len(array)
    if peak_n_datapoints == 0:
        return None

    peak_area = intensities.sum()
    peak_mean = intensities.mean()
    peak_max = intensities.max()
    peak_min = intensities.min()
    peak_median = np.median(intensities)

    peak_rt_of_max = times[masses.argmax()]

    peak_delta_int = np.abs(intensities[0] - intensities[-1])

    peak_mass_diff_25pc, peak_mass_diff_50pc, peak_mass_diff_75pc = \
        np.quantile( masses, [.25,.5,.75] )

    peak_shape_rt = float_list_to_comma_sep_str( projection[:,0] )
    peak_shape_int = float_list_to_comma_sep_str ( projection[:,1] )

    return dict(peak_area=peak_area, peak_max=peak_max, 
                peak_min=peak_min, peak_mean=peak_mean, 
                peak_rt_of_max=peak_rt_of_max, peak_median=peak_median,
                peak_delta_int=peak_delta_int, 
                peak_n_datapoints=peak_n_datapoints,
                peak_mass_diff_25pc=peak_mass_diff_25pc, 
                peak_mass_diff_50pc=peak_mass_diff_50pc, 
                peak_mass_diff_75pc=peak_mass_diff_75pc,
                peak_shape_rt=peak_shape_rt, 
                peak_shape_int=peak_shape_int)


def slice_ms1_array(array: np.array, rt_min, rt_max, mz_mean, mz_width, 
                    intensity_threshold):
    delta_mass = mz_width*mz_mean*1e-6
    array = array[(array[:, 0] >= rt_min)]
    array = array[(array[:, 0] <= rt_max)]
    array = array[(np.abs(array[:, 1]-mz_mean) <= delta_mass)]
    array = array[(array[:, 2] >= intensity_threshold)]
    return array


#def gaus(x,a,x0,sigma):
#    return a*np.exp(-(x-x0)**2/(2*sigma**2))
