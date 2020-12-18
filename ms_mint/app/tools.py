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
    return read_peaklists(get_peaklist_fn(wdir)).set_index('peak_label')


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


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, 'metadata','metadata.csv')
    dirname = os.path.dirname(fn)
    if not os.path.isdir(dirname): os.makedirs(dirname)
    return fn

def get_ms_fns(wdir):
    fns = glob(os.path.join(wdir, 'ms_files', '*.feather'))
    return fns