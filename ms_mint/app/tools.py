import base64
import os
import io
import shutil
import subprocess
import platform

import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import ms_mint
from ms_mint.io import ms_file_to_df
from ms_mint.peaklists import standardize_peaklist, read_peaklists
from ms_mint.io import convert_ms_file_to_feather

from datetime import date

from filelock import FileLock

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
    Versions:
    {get_versions()}
    '''

def parse_ms_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fn_abs = os.path.join(target_dir, filename)
    with lock(fn_abs):
        with open(fn_abs, 'wb') as file:
            file.write(decoded)
    new_fn = convert_ms_file_to_feather(fn_abs)
    print(f'Convert {fn_abs} to {new_fn}')
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


def get_actived_workspace(tmpdir):
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
    return get_dirnames( ws_path )


class Chromatograms():
    def __init__(self, wdir, peaklist, ms_files, progress_callback=None):
        self.wdir = wdir
        self.peaklist = peaklist
        self.ms_files = ms_files
        self.n_peaks = len(peaklist)
        self.n_files = len(ms_files)
        print(self.n_peaks, self.n_files)
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
            chrom = df[(df['m/z array']-mz_mean).abs()<=dmz]
            chrom['retentionTime'] = chrom['retentionTime'].round(3)
            chrom = chrom.groupby('retentionTime').max().reset_index()
            chrom[['retentionTime', 'intensity array']].to_feather(fn_chro)

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
    chrom = df[(df['m/z array']-mz_mean).abs()<=dmz]
    chrom['retentionTime'] = chrom['retentionTime'].round(3)
    chrom = chrom.groupby('retentionTime').max().reset_index()
    chrom[['retentionTime', 'intensity array']].to_feather(fn_out)
    if verbose:print('...done creating chromatogram.')
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
    fn = os.path.join(wdir, 'chromato', f'{mz_mean}-{mz_width}'.replace('.', '_'), base)+'.feather'
    return fn


def get_peaklist_fn(wdir):
    return os.path.join(wdir, 'peaklist', 'peaklist.csv')


def get_peaklist(wdir):
    fn = get_peaklist_fn( wdir )
    if os.path.isfile( fn ):
        return read_peaklists( fn ).set_index('peak_label')
    else: return None


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
    df['MS-file'] = df['ms_file']
    return df
    

def get_metadata(wdir):
    fn = get_metadata_fn( wdir )
    fn_path = os.path.dirname(fn)
    ms_files = get_ms_fns( wdir )
    ms_files = [ os.path.basename(fn) for fn in ms_files ]
    df = None
    if not os.path.isdir( fn_path ):
        os.makedirs( fn_path )
    if os.path.isfile(fn):
        df = pd.read_csv( fn )
        if 'MS-file' not in df.columns:
            df = None
    if df is None or len(df) == 0:
        df = init_metadata( ms_files )
    assert 'MS-file' in df.columns, df
    df = df.set_index('MS-file').reindex(ms_files).reset_index()
    df['MS-file'] = df['MS-file'].apply(os.path.basename)
    if 'PeakOpt' not in df.columns:
        df['PeakOpt'] = False
    else: df['PeakOpt'] = df['PeakOpt'].astype(bool)
    return df


def init_metadata( ms_files ):
    print('Init metadata')
    ms_files = list(ms_files)
    df = pd.DataFrame({'MS-file': ms_files})
    df['Label'] = ''
    df['Type'] = 'Biological Sample'
    df['Run Order'] = ''
    df['Batch'] = ''
    df['Row'] = ''
    df['Column'] = ''
    df['PeakOpt'] = ''
    return df


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, 'metadata', 'metadata.csv')
    return fn


def get_ms_dirname( wdir):
    return os.path.join(wdir, 'ms_files')


def get_ms_fns(wdir):
    path = get_ms_dirname( wdir )
    fns = glob(os.path.join(path, '*.*'))
    return fns

    
def Basename(fn):
    fn = os.path.basename(fn)
    fn, _ = os.path.splitext(fn)
    return fn


def get_complete_results( wdir ):
    meta = get_metadata( wdir )
    resu = get_results( wdir )
    assert 'MS-file' in resu.columns, resu
    assert 'MS-file' in meta.columns, meta
    print(len(resu), len(meta))
    resu['MS-file'] = [ Basename(fn) for fn in resu['MS-file']]
    meta['MS-file'] = [ Basename(fn) for fn in meta['MS-file']]
    df = pd.merge(meta, resu, on='MS-file')
    df['log(peak_max+1)'] = df.peak_max.apply(np.log1p)
    return df


def gen_tabulator_columns(col_names=None, add_ms_file_col=False, add_color_col=False, 
                          add_peakopt_col=False,
                          col_width='12px', editor='input'):

    if col_names is None: col_names = []
    col_names = list(col_names)

    standard_columns = ['MS-file', 'Color', 'index', 'PeakOpt']
    for col in standard_columns:
        if col in col_names: col_names.remove(col)
    

    columns = [
            { "formatter": "rowSelection", "titleFormatter":"rowSelection", 
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
    print('Decoding content from', filename)
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
    old_columns = old.columns.to_list()
    new_columns = new.columns.to_list()
    old_columns = [col for col in old_columns if (col=='MS-file') or (col not in new_columns)]
    return pd.merge(old[old_columns], new, on='MS-file', how='left')


def get_figure_fn(kind, wdir, label, format):
    path = os.path.join( wdir, 'figures', kind)
    clean_label = clean_string(label)
    fn = f'{kind}__{clean_label}.{format}'
    fn = os.path.join( path, fn)
    return path, fn


def clean_string(fn: str):
    for x in ['"', "'", '(', ')', '[', ']', ' ']:
        fn = fn.replace(x, '_')
    return fn


def savefig(kind=None, wdir=None, label=None, format='png', dpi=150):
    path, fn = get_figure_fn(kind=kind, wdir=wdir, label=label, format=format)
    maybe_create(path)
    try:
        plt.savefig(fn, dpi=dpi, bbox_inches='tight')
    except:
        print(f'Could not save figure {fn}, maybe no figure was created.')
    return fn


def maybe_create(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def png_fn_to_src(fn):
    encoded_image = base64.b64encode(open(fn, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


def get_ms_fns_for_peakopt(wdir):
    df = get_metadata( wdir )
    assert 'MS-file' in df.columns, df
    fns = df[df.PeakOpt.astype(bool) == True]['MS-file']
    print(df.PeakOpt.sum(), len(fns))
    return [ os.path.join( get_ms_dirname(wdir), fn) for fn in fns]


def float_to_color(x, vmin=0, vmax=2, cmap=None):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(x)


def write_peaklist(peaklist, wdir):
    fn = get_peaklist_fn( wdir )
    if 'peak_label' in peaklist.columns:
        peaklist = peaklist.set_index('peak_label')
    print('Write peaklist:', fn)
    print(peaklist)
    with lock(fn):
        peaklist.to_csv(fn)
