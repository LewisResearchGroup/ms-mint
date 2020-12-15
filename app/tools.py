import base64
import os
import io
from glob import glob
import pandas as pd
from tqdm import tqdm

from ms_mint.io import mzml_to_pandas_df, mzxml_to_pandas_df, ms_file_to_df
from ms_mint.peaklists import standardize_peaklist


def parse_ms_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fn_abs = os.path.join(target_dir, filename)
    with open(fn_abs, 'wb') as file:
        file.write(decoded)


def parse_pkl_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = standardize_peaklist(df)
    return df


def get_dirnames(path):
    dirnames = [ f.name for f in os.scandir(path) if f.is_dir() ]
    return dirnames


def create_chromatograms(ms_files, peaklist, wdir):
    for fn in tqdm(ms_files):
        df = ms_file_to_df(fn)
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