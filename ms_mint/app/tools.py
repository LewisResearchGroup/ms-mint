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

import ms_mint
from ms_mint.io import ms_file_to_df
from ms_mint.peaklists import standardize_peaklist, read_peaklists
from ms_mint.io import convert_ms_file_to_feather

from datetime import date


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
    ms_files = get_ms_fns( wdir )
    ms_files = pd.DataFrame([{'MS-file': Basename(fn) } for fn in ms_files])    
    if os.path.isfile(fn):
        df = pd.read_csv( fn ).reset_index()
    else:
        df = ms_files
        df['Label'] = ''
        df['Type'] = 'Biological Sample'
        df['Run Order'] = ''
        df['Batch'] = ''
        df['Row'] = ''
        df['Column'] = ''
    return df


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, 'metadata', 'metadata.csv')
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


def gen_tabulator_columns(col_names=None, add_ms_file_col=True, add_color_col=True, col_width='12px', editor='input'):
    if col_names is None: col_names = []
    col_names = list(col_names)
    
    if 'MS-file' in col_names: col_names.remove('MS-file')
    if 'Color' in col_names: col_names.remove('Color')
    if 'index' in col_names: col_names.remove('index')

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
    plt.savefig(out_img, format='png', bbox_inches='tight', dpi=dpi)
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
    clean_label = label
    fn = f'{kind}__{clean_label}.{format}'
    fn = os.path.join( path, fn)
    return path, fn


def savefig(kind, wdir, label, format='svg'):
    path, fn = get_figure_fn(kind=kind, wdir=wdir, label=label, format=format)
    maybe_create(path)
    plt.savefig(fn)


def maybe_create(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)