import base64
import os
import io
import shutil
import subprocess
import platform
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from pathlib import Path as P

import wget
import urllib3, ftplib
from urllib.parse import urlparse
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import matplotlib as mpl
import matplotlib.cm as cm

import ms_mint
from ms_mint.io import ms_file_to_df
from ms_mint.peaklists import standardize_peaklist, read_peaklists
from ms_mint.io import convert_ms_file_to_feather
from ms_mint.standards import PEAKLIST_COLUMNS

from datetime import date

from .filelock import FileLock

def lock(fn):
    return FileLock(f'{fn}.lock', timeout=1)


def today():
    return date.today().strftime('%y%m%d')


def get_versions():
    string = ''
    try:
        string += subprocess.getoutput('conda env export --no-build')
    except:
        pass
    return string


def get_issue_text():
    return f'''
    %0A%0A%0A%0A%0A%0A%0A%0A%0A
    MINT version: {ms_mint.__version__}%0A
    OS: {platform.platform()}%0A
    Versions: ---
    '''

def parse_ms_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fn_abs = os.path.join(target_dir, filename)
    with lock(fn_abs):
        with open(fn_abs, 'wb') as file:
            file.write(decoded)
    new_fn = convert_ms_file_to_feather(fn_abs)
    if os.path.isfile(new_fn): os.remove(fn_abs)


def parse_pkl_files(contents, filename, date, target_dir, ms_mode=None):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = standardize_peaklist(df, ms_mode=ms_mode)
    df = df.drop_duplicates()
    return df


def get_dirnames(path):
    dirnames = [ f.name for f in os.scandir(path) if f.is_dir() ]
    return dirnames


def workspace_path(tmpdir, ws_name):
    return os.path.join(tmpdir, 'workspaces', ws_name)


def maybe_migrate_workspaces(tmpdir):
    dir_names = get_dirnames(tmpdir)
    ws_path = get_workspaces_path(tmpdir)
    if not os.path.isdir(ws_path) and len(dir_names)>0:
        print('Migrating to new directory scheme')
        os.makedirs(ws_path)
        for dir_name in dir_names:
            old_dir = os.path.join( tmpdir, dir_name)
            new_dir = workspace_path(tmpdir, dir_name)
            shutil.move(old_dir, new_dir)
            print('Moving', old_dir, 'to', new_dir)


def workspace_exists(tmpdir, ws_name):
    path = workspace_path(tmpdir, ws_name)
    return os.path.isdir(path)


def get_active_workspace(tmpdir):
    '''Returns name of last activated workspace,
       if workspace still exists. Otherwise,
       return None.
    '''
    fn_ws_info = os.path.join( tmpdir, '.active-workspace')
    if not os.path.isfile(fn_ws_info):
        return None
    with open(fn_ws_info, 'r') as file:
        ws_name = file.read()
    if ws_name in get_workspaces(tmpdir):
        return ws_name
    else:
        return None


def save_activated_workspace(tmpdir, ws_name):
    fn_ws_info = os.path.join( tmpdir, '.active-workspace')
    with open(fn_ws_info, 'w') as file:
            file.write(ws_name)


def create_workspace(tmpdir, ws_name):
    path = workspace_path(tmpdir, ws_name)
    assert not os.path.isdir(path)
    os.makedirs(path)
    os.makedirs(os.path.join(path, 'ms_files'))
    os.makedirs(os.path.join(path, 'peaklist'))
    os.makedirs(os.path.join(path, 'results'))
    os.makedirs(os.path.join(path, 'figures'))
    os.makedirs(os.path.join(path, 'chromato'))

def get_workspaces_path(tmpdir):
    # Defines the path to the workspaces
    # relative to `tmpdir`
    return os.path.join(tmpdir, 'workspaces')


def get_workspaces(tmpdir):
    ws_path = get_workspaces_path(tmpdir)
    ws_names = get_dirnames( ws_path )
    ws_names = [ws for ws in ws_names if not ws.startswith('.')]
    ws_names.sort()
    return ws_names


class Chromatograms():
    def __init__(self, wdir, peaklist, ms_files, progress_callback=None):
        self.wdir = wdir
        self.peaklist = peaklist
        self.ms_files = ms_files
        self.n_peaks = len(peaklist)
        self.n_files = len(ms_files)
        self.progress_callback = progress_callback

    def create_all(self):
        for fn in tqdm( self.ms_files ):
            self.create_all_for_ms_file(fn)
        return self

    def create_all_for_ms_file(self, ms_file: str):
        fn = ms_file
        df = ms_file_to_df(fn)
        for ndx, row in self.peaklist.iterrows():
            mz_mean, mz_width = row[['mz_mean', 'mz_width']]
            fn_chro = get_chromatogram_fn(fn, mz_mean, mz_width, self.wdir)
            if os.path.isfile(fn_chro): continue
            dirname = os.path.dirname(fn_chro)
            if not os.path.isdir(dirname): os.makedirs(dirname)
            dmz = mz_mean*1e-6*mz_width
            chrom = df[(df['mz']-mz_mean).abs()<=dmz]
            chrom['scan_time_min'] = chrom['scan_time_min'].round(3)
            chrom = chrom.groupby('scan_time_min').max().reset_index()
            chrom[['scan_time_min', 'intensity']].to_feather(fn_chro)

    def get_single(self, mz_mean, mz_width, ms_file):
        return get_chromatogram(ms_file, mz_mean, mz_width, self.wdir)      
    

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


def create_chromatogram(ms_file, mz_mean, mz_width, fn_out, verbose=False):
    if verbose: print('Creating chromatogram')
    df = ms_file_to_df(ms_file)
    if verbose: print('...file read')
    dirname = os.path.dirname(fn_out)
    if not os.path.isdir(dirname): os.makedirs(dirname)
    dmz = mz_mean*1e-6*mz_width
    chrom = df[(df['mz']-mz_mean).abs()<=dmz]
    chrom['scan_time_min'] = chrom['scan_time_min'].round(3)
    chrom = chrom.groupby('scan_time_min').max().reset_index()
    with lock(fn_out):
        chrom[['scan_time_min', 'intensity']].to_feather(fn_out)
    if verbose:print('...done creating chromatogram.')
    return chrom


def get_chromatogram(ms_file, mz_mean, mz_width, wdir):
    fn = get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir)
    if not os.path.isfile(fn):
        chrom = create_chromatogram(ms_file, mz_mean, mz_width, fn)
    else:
        try:
            chrom = pd.read_feather(fn)
        except:
            os.remove(fn)
            logging.warning(f'Cound not read {fn}.')
            return None

    chrom = chrom.rename(columns={
            'retentionTime': 'scan_time_min', 
            'intensity array': 'intensity', 
            'm/z array': 'mz'})

    return chrom


def get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir):
    ms_file = os.path.basename(ms_file)
    base, _ = os.path.splitext(ms_file)
    fn = os.path.join(wdir, 'chromato', f'{mz_mean}-{mz_width}'.replace('.', '_'), base)+'.feather'
    return fn


def get_peaklist_fn(wdir):
    return os.path.join(wdir, 'peaklist', 'peaklist.csv')


def get_peaklist(wdir):
    fn = get_peaklist_fn( wdir )
    if os.path.isfile( fn ):
        return read_peaklists( fn ).set_index('peak_label')
    else: 
        return pd.DataFrame(columns=PEAKLIST_COLUMNS)


def update_peaklist(wdir, peak_label, rt_min=None, rt_max=None, rt=None):
    peaklist = get_peaklist(wdir)

    if isinstance(peak_label, str):
        if rt_min is not None and not np.isnan(rt_min):
            peaklist.loc[peak_label, 'rt_min'] = rt_min
        if rt_max is not None and not np.isnan(rt_max):
            peaklist.loc[peak_label, 'rt_max'] = rt_max
        if rt is not None and not np.isnan(rt):
            peaklist.loc[peak_label, 'rt'] = rt

    if isinstance(peak_label, int):
        peaklist = peaklist.reset_index()
        if rt_min is not None and not np.isnan(rt_min):
            peaklist.loc[peak_label, 'rt_min'] = rt_min
        if rt_max is not None and not np.isnan(rt_max):
            peaklist.loc[peak_label, 'rt_max'] = rt_max
        if rt is not None and not np.isnan(rt):
            peaklist.loc[peak_label, 'rt'] = rt
        peaklist = peaklist.set_index('peak_label')

    fn = get_peaklist_fn(wdir)
    with lock(fn):
        peaklist.to_csv(fn)


def get_results_fn(wdir):
    return os.path.join(wdir, 'results', 'results.csv')


def get_results( wdir ):
    fn = get_results_fn( wdir )
    df = pd.read_csv( fn )
    df['MS-file'] = [filename_to_label(fn) for fn in df['ms_file']]
    return df
    

def get_metadata(wdir):
    fn = get_metadata_fn( wdir )
    fn_path = os.path.dirname(fn)
    ms_files = get_ms_fns( wdir, abs_path=False )
    ms_files = [filename_to_label(fn) for fn in ms_files]
    df = None
    if not os.path.isdir( fn_path ):
        os.makedirs( fn_path )
    if os.path.isfile(fn):
        df = pd.read_csv( fn )
        if 'MS-file' not in df.columns:
            df = None

    if df is None or len(df) == 0:
        df = init_metadata( ms_files )

    if 'Color' not in df.columns:
        df['Color'] = None
    
    df = df[df['MS-file'] != '']

    df = df.groupby('MS-file').first().reindex(ms_files).reset_index()

    if 'PeakOpt' not in df.columns:
        df['PeakOpt'] = False

    else: df['PeakOpt'] = df['PeakOpt'].astype(bool)

    if 'InAnalysis' not in df.columns:
        df['InAnalysis'] = True
    else: 
        df['InAnalysis'] = df['InAnalysis'].astype(bool)

    if 'index' in df.columns: del df['index']

    df['Column'] = df['Column'].apply(format_columns)

    df['Type'] = df['Type'].fillna('Not set')

    df.reset_index(inplace=True)

    return df


def init_metadata( ms_files ):
    ms_files = list(ms_files)
    ms_files = [filename_to_label(fn) for fn in ms_files]
    df = pd.DataFrame({'MS-file': ms_files})
    df['InAnalysis'] = True
    df['Label'] = ''
    df['Color'] = None
    df['Type'] = 'Biological Sample'
    df['RunOrder'] = ''
    df['Batch'] = ''
    df['Row'] = ''
    df['Column'] = ''
    df['PeakOpt'] = ''
    return df


def write_metadata( meta, wdir ):
    fn = get_metadata_fn( wdir )
    with lock(fn):
        meta.to_csv( fn, index=False)


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, 'metadata', 'metadata.csv')
    return fn


def get_ms_dirname( wdir):
    return os.path.join(wdir, 'ms_files')


def get_ms_fns(wdir, abs_path=True):
    path = get_ms_dirname( wdir )
    fns = glob(os.path.join(path, '**', '*.*'), recursive=True)
    fns = [fn for fn in fns if is_ms_file(fn)]
    if not abs_path:
        fns = [os.path.basename(fn) for fn in fns]
    return fns


def is_ms_file(fn: str):
    if    fn.lower().endswith('.mzxml') \
       or fn.lower().endswith('.mzml') \
       or fn.lower().endswith('.feather'):
        return True
    return False


def Basename(fn):
    fn = os.path.basename(fn)
    fn, _ = os.path.splitext(fn)
    return fn


def format_columns(x):
    try:
        if (x is None) or (x == '') or np.isnan(x): return None
    except:
        print(type(x))
        print(x)
        assert False
    return f'{int(x):02.0f}'


def get_complete_results( wdir, include_labels=None, exclude_labels=None, 
        file_types=None, include_excluded=False ):
    meta = get_metadata( wdir )
    resu = get_results( wdir )

    if not include_excluded: meta = meta[meta['InAnalysis']]
    df = pd.merge(meta, resu, on=['MS-file'])
    if include_labels is not None and len(include_labels) > 0:
        df = df[df.peak_label.isin(include_labels)]
    if exclude_labels is not None and len(exclude_labels) > 0:
        df = df[~df.peak_label.isin(exclude_labels)]
    if file_types is not None and file_types != []:
        df = df[df.Type.isin(file_types)]
    df['log(peak_max+1)'] = df.peak_max.apply(np.log1p)
    if 'index' in df.columns: df = df.drop('index', axis=1)
    return df


def gen_tabulator_columns(col_names=None, add_ms_file_col=False, add_color_col=False, 
                          add_peakopt_col=False, add_ms_file_active_col=False,
                          col_width='12px', editor='input'):

    if col_names is None: col_names = []
    col_names = list(col_names)

    standard_columns = ['MS-file', 'InAnalysis', 'Color', 'index', 'PeakOpt', ]

    for col in standard_columns:
        if col in col_names: col_names.remove(col)
    

    columns = [
            { "formatter": "rowSelection", "titleFormatter":"rowSelection",           
              "titleFormatterParams": {
                  "rowRange": "active" # only toggle the values of the active filtered rows
              },
              "hozAlign":"center", "headerSort": False, "width":"1px", 'frozen': True}]

    if add_ms_file_col:
        columns.append(
            { "title": "MS-file", 
              "field": "MS-file", 
              "headerFilter":True, 
              'headerSort': True, 
              "editor": "input", 
              'sorter': 'string', 
              'frozen': True
            })
    
    if add_color_col:
        columns.append(
            { 'title': 'Color', 
              'field': 'Color', 
              "headerFilter":False,  
              "editor": "input", 
              "formatter": "color", 
              'width': '3px', 
              "headerSort": False
            })
    
    if add_peakopt_col:
        columns.append(
            { 'title': 'PeakOpt', 
              'field': 'PeakOpt', 
              "headerFilter":False,  
              "formatter": "tickCross", 
              'width': '6px', 
              "headerSort": True,
              "hozAlign": "center",
              "editor": True
            })

    if add_ms_file_active_col:
        columns.append(
            { 'title': 'InAnalysis', 
              'field': 'InAnalysis', 
              "headerFilter":True,  
              "formatter": "tickCross", 
              'width': '6px', 
              "headerSort": True,
              "hozAlign": "center",
              "editor": True
            })

    for col in col_names:
        content = { 'title': col, 
                    'field': col, 
                    "headerFilter":True, 
                    'width': col_width, 
                    'editor': editor 
                  }

        columns.append(content)
    return columns


def parse_table_content(content, filename):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith('.csv'):
        df = pd.read_csv( io.StringIO(decoded.decode('utf-8')) )
    elif filename.lower().endswith('.xlsx'):
        df = pd.read_excel( io.BytesIO(decoded) )
    return df


def fig_to_src(dpi=100):
    out_img = io.BytesIO()

    plt.savefig(out_img, format='jpeg', bbox_inches='tight', dpi=dpi)

    plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def merge_metadata(old, new):
    old = old.set_index('MS-file')

    new = new.groupby('MS-file').first().replace('null', None)

    for col in new.columns:
        if col == '' or col.startswith('Unnamed'):
            continue
        if not col in old.columns: old[col] = None
        for ndx in new.index:
            value = new.loc[ndx, col]
            if value is None:
                continue
            if ndx in old.index:
                old.loc[ndx, col] = value
    return old.reset_index()


def file_colors( wdir ):
    meta = get_metadata( wdir )
    colors = {}
    for ndx, (fn, co) in meta[['MS-file', 'Color']].iterrows():
        if not (isinstance(co, str)): co = None
        colors[fn] = co
    return colors


def get_figure_fn(kind, wdir, label, format):
    path = os.path.join( wdir, 'figures', kind)
    clean_label = clean_string(label)
    fn = f'{kind}__{clean_label}.{format}'
    fn = os.path.join( path, fn)
    return path, fn


def clean_string(fn: str):
    for x in ['"', "'", '(', ')', '[', ']', ' ', '\\', '/', '{', '}']:
        fn = fn.replace(x, '_')
    return fn


def savefig(kind=None, wdir=None, label=None, format='png', dpi=150):
    path, fn = get_figure_fn(kind=kind, wdir=wdir, label=label, format=format)
    maybe_create(path)
    try:
        with lock(fn):
            plt.savefig(fn, dpi=dpi, bbox_inches='tight')
    except:
        print(f'Could not save figure {fn}, maybe no figure was created: {label}')
    return fn


def maybe_create(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def png_fn_to_src(fn):
    encoded_image = base64.b64encode(open(fn, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


def get_ms_fns_for_peakopt(wdir):
    '''Extract the filenames for peak optimization from
       the metadata table and recreate the complete filename.'''
    df = get_metadata( wdir )
    fns = df[df.PeakOpt.astype(bool) == True]['MS-file']
    ms_files = get_ms_fns( wdir )
    mapping = {filename_to_label(fn): fn for fn in ms_files}
    fns = [mapping[fn] for fn in fns]
    return fns


def float_to_color(x, vmin=0, vmax=2, cmap=None):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(x)


def write_peaklist(peaklist, wdir):
    fn = get_peaklist_fn( wdir )
    if 'peak_label' in peaklist.columns:
        peaklist = peaklist.set_index('peak_label')
    with lock(fn):
        peaklist.to_csv(fn)


def filename_to_label(fn: str):
    if is_ms_file(fn):
        fn = os.path.splitext(fn)[0]
    return os.path.basename(fn)


    
def import_from_url(url, target_dir, fsc=None):
    filenames = get_filenames_from_url(url)
    filenames = [fn for fn in filenames if is_ms_file(fn)]
    if len(filenames) == 0:
        return None
    fns = []
    n_files = len(filenames)
    for i, fn in enumerate( tqdm( filenames )):
        _url = url+'/'+fn
        logging.info('Downloading', _url)
        if fsc is not None: fsc.set('progress', int(100*(1+i)/n_files))
        wget.download(_url, out=target_dir)
    return fns


def get_filenames_from_url(url):
    if url.startswith('ftp'):
        return get_filenames_from_ftp_directory(url)
    if '://' in url:
            url = url.split('://')[1]    
    with urllib3.PoolManager() as http:
        r = http.request('GET', url)
    soup = BeautifulSoup(r.data, 'html')
    files = [A['href'] for A in soup.find_all('a', href=True)]
    return files


def get_filenames_from_ftp_directory(url):
    url_parts = urlparse(url)
    domain = url_parts.netloc
    path = url_parts.path
    ftp = ftplib.FTP(domain)
    ftp.login()
    ftp.cwd(path)
    filenames = ftp.nlst()
    ftp.quit()
    return filenames


def import_from_local_path(path, target_dir, fsc=None):
    fns = glob(os.path.join(path, '**', '*.*'), recursive=True)
    fns = [fn for fn in fns if is_ms_file(fn)]
    fns_out = []
    n_files = len(fns)
    for i, fn in enumerate( tqdm(fns) ):
        if fsc is not None: fsc.set('progress', int(100*(1+i)/n_files))
        fn_out = P(target_dir)/P(fn).with_suffix('.feather').name
        if P(fn_out).is_file(): continue
        fns_out.append(fn_out)
        try:
            convert_ms_file_to_feather(fn, fn_out)
        except:
            logging.warning(f'Could not convert {fn}')
    return fns_out
    
            