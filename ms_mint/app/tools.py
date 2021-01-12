import base64
import os
import io

import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

from ms_mint.io import ms_file_to_df
from ms_mint.peaklists import standardize_peaklist, read_peaklists
from ms_mint.io import convert_ms_file_to_feather

from datetime import date



def today():
    return date.today().strftime('%y%m%d')


def parse_ms_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fn_abs = os.path.join(target_dir, filename)
    with open(fn_abs, 'wb') as file:
        file.write(decoded)
    new_fn = convert_ms_file_to_feather(fn_abs)
    print(f'Convert {fn_abs} to {new_fn}')
    if os.path.isfile(new_fn): os.remove(fn_abs)


def parse_pkl_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = standardize_peaklist(df)
    df = df.drop_duplicates()
    return df


def get_dirnames(path):
    dirnames = [ f.name for f in os.scandir(path) if f.is_dir() ]
    return dirnames


def create_chromatograms(ms_files, peaklist, wdir):
    for fn in tqdm(ms_files):
        fn_out = os.path.basename(fn)
        fn_out, _ = os.path.splitext(fn_out) 
        fn_out += '.feather'
        for ndx, row in peaklist.iterrows():
            mz_mean, mz_width = row[['mz_mean', 'mz_width']]
            fn_chro = get_chromatogram_fn(fn, mz_mean, mz_width, wdir)
            if not os.path.isfile(fn_chro):
                create_chromatogram(fn, mz_mean, mz_width, fn_chro)


def create_chromatogram(ms_file, mz_mean, mz_width, fn_out):
    df = ms_file_to_df(ms_file)
    dirname = os.path.dirname(fn_out)
    if not os.path.isdir(dirname): os.makedirs(dirname)
    dmz = mz_mean*1e-6*mz_width
    chrom = df[(df['m/z array']-mz_mean).abs()<=dmz]
    chrom['retentionTime'] = chrom['retentionTime'].round(3)
    chrom = chrom.groupby('retentionTime').max().reset_index()
    chrom[['retentionTime', 'intensity array']].to_feather(fn_out)
    return chrom


def get_chromatogram(ms_file, mz_mean, mz_width, wdir):
    fn = get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir)
    if not os.path.isfile(fn):
        chrom = create_chromatogram(ms_file, mz_mean, mz_width, fn)
    else:
        chrom = pd.read_feather(fn)
    return chrom


def get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir):
    ms_file = os.path.basename(ms_file)
    base, _ = os.path.splitext(ms_file)
    fn = os.path.join(wdir, 'chromato', f'{mz_mean}-{mz_width}'.replace('.', '_'), base, '.feather')
    return fn


def get_peaklist_fn(wdir):
    return os.path.join(wdir, 'peaklist', 'peaklist.csv')


def get_peaklist(wdir):
    fn = get_peaklist_fn( wdir )
    if os.path.isfile( fn ):
        return read_peaklists( fn ).set_index('peak_label')
    else: return None


def update_peaklist(wdir, peak_label, rt_min=None, rt_max=None):
    peaklist = get_peaklist(wdir)

    if isinstance(peak_label, str):
        if not np.isnan(rt_min):
            peaklist.loc[peak_label, 'rt_min'] = rt_min
        if not np.isnan(rt_max):
            peaklist.loc[peak_label, 'rt_max'] = rt_max

    if isinstance(peak_label, int):
        peaklist = peaklist.reset_index()
        if not np.isnan(rt_min):
            peaklist.loc[peak_label, 'rt_min'] = rt_min
        if not np.isnan(rt_max):
            peaklist.loc[peak_label, 'rt_max'] = rt_max
        peaklist = peaklist.set_index('peak_label')
    peaklist.to_csv(get_peaklist_fn(wdir))


def get_results_fn(wdir):
    return os.path.join(wdir, 'results', 'results.csv')


def get_results( wdir ):
    fn = get_results_fn( wdir )
    df = pd.read_csv( fn )
    df['MS-file'] = df['ms_file']
    return df
    

def get_metadata(wdir):
    fn = get_metadata_fn( wdir )
    if os.path.isfile(fn):
        return pd.read_csv( fn )
    else: return None


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, 'metadata','metadata.csv')
    dirname = os.path.dirname(fn)
    if not os.path.isdir(dirname): os.makedirs(dirname)
    return fn


def get_ms_fns(wdir):
    fns = glob(os.path.join(wdir, 'ms_files', '*.feather'))
    return fns

    
def Basename(fn):
    fn = os.path.basename(fn)
    fn, _ = os.path.splitext(fn)
    return fn


def get_complete_results( wdir ):
    meta = get_metadata( wdir )
    resu = get_results( wdir )
    resu['MS-file'] = [ Basename(fn) for fn in resu['MS-file']]
    df = pd.merge(meta, resu, on='MS-file')
    df['log(peak_max+1)'] = df.peak_max.apply(np.log1p)
    return df


def gen_tabulator_columns(col_names=None):
    if col_names is None: col_names = []
    col_names = list(col_names)
    if 'MS-file' in col_names: col_names.remove('MS-file')
    if 'Color' in col_names: col_names.remove('Color')
    if 'index' in col_names: col_names.remove('index')

    columns = [
        { "formatter":"rowSelection", "titleFormatter":"rowSelection", 
          "hozAlign":"center", "headerSort": False, "width":"1px", 'frozen': True},
        { "title": "MS-file", "field": "MS-file", "headerFilter":True, 
          'headerSort': True, "editor": "input", "headerFilter":True, 
          'sorter': 'string', 'frozen': True},
        { 'title': 'Color', 'field': 'Color', "headerFilter":False,  "formatter":"color", 
          'width': '3px', "headerSort": False},
    ]
    for col in col_names:
        content = { 'title': col, 'field': col, "headerFilter":True, 'width': '12px' }
        columns.append(content)

    #columns[-1]['width'] = None
    #columns[-1]['widthGrowth'] = 5

    return columns