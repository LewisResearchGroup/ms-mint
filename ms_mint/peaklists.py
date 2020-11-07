# ms_mint/peaklists.py

import os
import pandas as pd
import numpy as np

from .standards import PEAKLIST_COLUMNS, DEPRICATED_LABELS
from .helpers import dataframe_difference


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
    if isinstance(filenames, str):
        filenames = [filenames]

    peaklist = []
    for fn in filenames:
        if fn.endswith('.csv'):
            df = pd.read_csv(fn)
        elif fn.endswith('.xlsx'):
            df = pd.read_excel(fn)
        df['peaklist_name'] = os.path.basename(fn)
        df = standardize_peaklist(df)
        peaklist.append(df)
    peaklist = pd.concat(peaklist)
    return peaklist


def standardize_peaklist(peaklist):
    cols = peaklist.columns
    peaklist = peaklist.rename(columns=DEPRICATED_LABELS)
    if 'intensity_threshold' not in cols:
        peaklist['intensity_threshold'] = 0
    if 'mz_width' not in cols:
        peaklist['mz_width'] = 10
    if 'peaklist_name' not in cols:
        peaklist['peaklist_name'] = 'unknown'
    peaklist['peak_label'] = peaklist['peak_label'].astype(str)
    peaklist.index = range(len(peaklist))
    return peaklist[PEAKLIST_COLUMNS]


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
    print(peaklist)
    if not isinstance(peaklist, pd.DataFrame):
        errors.append('Peaklist is not a dataframe.')
    peaklist[PEAKLIST_COLUMNS]
    if not peaklist.dtypes['peak_label'] == np.dtype('O'):
        errors.append('Provided peak labels are not strings.', 
                       peaklist.dtypes['peak_label'] )
    if not peaklist.peak_label.value_counts().max() == 1:
        errors.append('Provided peak labels are not unique.')
    return errors


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
    peaklist['peak_label'] =  peaklist.mz_mean.apply(lambda x: '{:.3f}'.format(x))\
                              + '__' + peaklist.rt_min.apply(lambda x: '{:2.2f}'.format(x))
    peaklist['mz_width'] = mz_ppm
    peaklist['intensity_threshold'] = intensity_threshold
    peaklist['peaklist_name'] = 'Generated'
    return peaklist


def diff_peaklist(old_pklist, new_pklist):
    df = dataframe_difference(old_pklist, new_pklist)
    df = df[df['_merge'] == 'right_only']
    return df.drop('_merge', axis=1)