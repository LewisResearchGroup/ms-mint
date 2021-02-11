# ms_mint/peaklists.py

import os
import pandas as pd
import numpy as np

from .standards import PEAKLIST_COLUMNS, DEPRECATED_LABELS
from .helpers import dataframe_difference
from .tools import get_mz_mean_from_formulas


def read_peaklists(filenames, ms_mode='negative'):
    '''
    Extracts peak data from csv files that contain peak definitions.
    CSV files must contain columns: 
        - 'peak_label': str, unique identifier
        - 'mz_mean': float, center of mass to be extracted in [Da]
        - 'mz_width': float, with of mass window in [ppm]
        - 'rt_min': float, minimum retention time in [min]
        - 'rt_max': float, maximum retention time in [min]
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
        if len(df) == 0:
            return pd.DataFrame(columns=PEAKLIST_COLUMNS, index=[])
        df['peaklist_name'] = os.path.basename(fn)
        df = standardize_peaklist(df)
        peaklist.append(df)
    peaklist = pd.concat(peaklist)
    return peaklist


def standardize_peaklist(peaklist, ms_mode='neutral'):
    peaklist = peaklist.rename(columns=DEPRECATED_LABELS)
    assert pd.value_counts(peaklist.columns).max() == 1, pd.value_counts( peaklist.columns )
    cols = peaklist.columns
    if 'formula' in peaklist.columns and not 'mz_mean' in peaklist.columns:
        peaklist['mz_mean'] = get_mz_mean_from_formulas(peaklist['formula'], ms_mode)    
    if 'intensity_threshold' not in cols:
        peaklist['intensity_threshold'] = 0
    if 'mz_width' not in cols:
        peaklist['mz_width'] = 10
    if 'peaklist_name' not in cols:
        peaklist['peaklist_name'] = 'unknown'
    for c in ['rt', 'rt_min', 'rt_max']:
        if c not in cols:
            peaklist[c] = None
    del c
    if 'peak_label' not in cols:
        peaklist['peak_label'] = [f'C_{i}' for i in range(len(peaklist)) ]        
    peaklist['intensity_threshold'] = peaklist['intensity_threshold'].fillna(0)
    peaklist['peak_label'] = peaklist['peak_label'].astype(str)
    peaklist.index = range(len(peaklist))
    peaklist = peaklist[peaklist.mz_mean.notna()]
    return peaklist[PEAKLIST_COLUMNS]


def update_retention_time_columns(peaklist):
    for ndx, row in peaklist.iterrows():
        if row['rt'] is not None:
            if ['rt_min'] is None:
                peaklist.loc[ndx, 'rt_min'] = 5 #max( 0, row['rt'] - 0.2 )
            if row['rt_max'] is None:
                peaklist.loc[ndx, 'rt_max'] = row['rt'] + 0.2
        else:
            if (row['rt_min'] is not None) & (row['rt_max'] is not None):
                peaklist.loc[ndx, 'row'] = row[['rt_min', 'rt_max']].mean(axis=1)



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
    if not (peaklist.dtypes['peak_label'] == np.dtype('O')):
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