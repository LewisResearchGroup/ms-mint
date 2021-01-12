
import io
import os
import re
import base64

import shutil
import tempfile
import subprocess
import platform

from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import seaborn as sns    

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.snippets import send_file
from dash_extensions.enrich import FileSystemCache

from flask_caching import Cache

import plotly.graph_objects as go

from ms_mint.Mint import Mint
from ms_mint.peaklists import read_peaklists
from ms_mint.io import convert_ms_file_to_feather
from ms_mint.vis.plotly.plotly_tools import plot_heatmap
from ms_mint.peak_optimization.RetentionTimeOptimizer import RetentionTimeOptimizer as RTOpt

import ms_mint

from .tools import parse_ms_files, get_dirnames, parse_pkl_files, get_chromatogram,\
    get_metadata_fn, get_results_fn, update_peaklist, today, get_ms_fns, get_peaklist,\
    Basename, get_metadata, get_results, get_complete_results, gen_tabulator_columns

from .ms_files import ms_layout
from .workspaces import ws_layout
from .peaklist import pkl_layout
from .heatmap import heat_layout, heat_layout_empty, heat_layout_non_empty
from .peak_optimization import pko_layout, pko_layout_no_data
from .metadata import meta_layout
from .quality_control import qc_layout, qc_layout_no_data

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

def get_versions():
    string = ''
    try:
        string += subprocess.getoutput('conda env export --no-build')
    except:
        pass
    return string


ISSUE_TEXT = f'''
%0A%0A%0A%0A%0A%0A%0A%0A%0A
MINT version: {ms_mint.__version__}%0A
OS: {platform.platform()}%0A
Versions:
{get_versions()}
'''

pd.options.display.max_colwidth= 1000


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, 'MINT')
    tmpdir = os.getenv('MINT_DATA_DIR', default=tmpdir)
    cachedir = os.path.join(tmpdir, '.cache')
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    return tmpdir, cachedir

TMPDIR, CACHEDIR = make_dirs()

fsc = FileSystemCache(CACHEDIR)

app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        "https://codepen.io/chriddyp/pen/bWLwgP.css"],
    requests_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/'),
    routes_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/')
    )

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/MINT-cache'
})

app.title = 'MINT'

app.config['suppress_callback_exceptions'] = True

layout = html.Div([
    dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=False),
    html.Button('Run MINT', id='run-mint'),
    html.Button('Download', id='res-download'),
    html.Button('Delete results', id='res-delete'),
    html.A(href='https://soerendip.github.io/ms-mint/gui/', 
         children=[html.Button('Documentation', id='B_help', style={'float': 'right'})],
         target="_blank"),
    html.A(href=f'https://github.com/soerendip/ms-mint/issues/new?body={ISSUE_TEXT}', 
         children=[html.Button('Issues', id='B_issues', style={'float': 'right'})],
         target="_blank"),   
    Download(id='res-download-data'),
    html.Div(id='run-mint-output'),
    dcc.Markdown(id='res-delete-output'),
    dbc.Progress(id="progress-bar", value=100, style={'margin-bottom': '20px'}),
    html.Div(id='tmpdir', children=TMPDIR, style={'visibility': 'hidden'}),
    html.Div(id='wdir', children=TMPDIR),
    dcc.Tabs(id='tab', value='workspaces', children=[
        dcc.Tab(label='Workspace', value='workspaces'),
        dcc.Tab(label='MS-files', value='msfiles'),
        dcc.Tab(label='Metadata', value='metadata'),
        dcc.Tab(label='Peaklist', value='peaklist'),
        dcc.Tab(label='Peak Optimization', value='pko'),
        dcc.Tab(label='Quality Control', value='qc'),
        dcc.Tab(label='Heatmap', value='heatmap'),
        #   dcc.Tab(label='PCA', value='pca'),
    ]),
    html.Div(id='tab-content', style={'margin': '5%'})
], style={'margin':'2%'})

app.layout = layout

@app.callback(Output('tab-content', 'children'),
              Input('tab', 'value'),
              State('wdir', 'children'))
def render_content(tab, wdir):
    if tab == 'msfiles':
        return ms_layout
    elif tab == 'workspaces':
        return ws_layout
    elif tab == 'peaklist':
        return pkl_layout
    elif tab == 'heatmap':
        return heat_layout
    elif tab == 'pko':
        pkl = get_peaklist(wdir)
        if pkl is not None and len(pkl) > 0:
            return pko_layout
        else: return pko_layout_no_data
    elif tab == 'metadata':
        return meta_layout
    elif tab == 'qc':
        fn =  get_results_fn(wdir)
        if os.path.isfile( fn ):
            return qc_layout
        else:
            return qc_layout_no_data


@app.callback(
Output('run-mint-output', 'children'),
Input('run-mint', 'n_clicks'),
State('wdir', 'children')
)
def run_mint(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate

    def set_progress(x):
        fsc.set('progress', x)

    mint = Mint(verbose=False, progress_callback=set_progress)
    mint.peaklist_files = os.path.join(wdir, 'peaklist', 'peaklist.csv')
    mint.ms_files = glob( os.path.join(wdir, 'ms_files', '*.*'))
    mint.run()
    mint.export( os.path.join(wdir, 'results', 'results.csv'))


@app.callback(
Output('peak-labels', 'options'),
Input('tab', 'value'),
State('wdir', 'children')
)
def peak_labels(tab, wdir):
    if tab not in ['qc']:
        raise PreventUpdate
    peaklist = get_peaklist( wdir ).reset_index()
    peak_labels = [{'value': i, 'label': i} for i in peaklist.peak_label]
    return peak_labels



# MS-FILES ############################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


@app.callback(
Output('ms-upload-output', 'children'),
Input('ms-upload', 'contents'),
Input('ms-convert-output', 'children'),
State('ms-upload', 'filename'),
State('ms-upload', 'last_modified'),
State('wdir', 'children'))
def ms_upload(list_of_contents, converted, list_of_names, list_of_dates, wdir):
    target_dir = os.path.join(wdir, 'ms_files')
    if list_of_contents is not None:
        n_total = len(list_of_contents)
        n_uploaded = 0
        for i, (c, n, d) in tqdm( enumerate( zip(list_of_contents, list_of_names, list_of_dates) ), total=n_total):
            fsc.set('progress', int( 100*(i+1)/n_total ))
            if n.lower().endswith('mzxml') or n.lower().endswith('mzml'):
                try:
                    parse_ms_files(c, n, d, target_dir)
                    n_uploaded += 1
                except:
                    pass
        return html.P(f'{n_uploaded} files uploaded.')
    

@app.callback(
Output('ms-table', 'data'),
Input('ms-upload-output', 'children'),
Input('wdir', 'children'), 
Input('ms-delete-output', 'children'),
State('ms-table', 'derived_virtual_selected_rows'),
State('ms-table', 'derived_virtual_indices')
)
def ms_table(value, wdir, files_deleted, ndxs_selected, ndxs_filtered):   
    target_dir = os.path.join(wdir, 'ms_files')
    ms_files = glob(os.path.join(target_dir, '*.*'), recursive=True)
    data =  pd.DataFrame([{'MS-file': os.path.basename(fn) } for fn in ms_files])
    return data.to_dict('records')


@app.callback(
Output('ms-convert-output', 'children'),
Input('ms-convert', 'n_clicks'),
State('wdir', 'children')
)
def ms_convert(n_clicks, wdir):
    target_dir = os.path.join(wdir, 'ms_files')
    if n_clicks is None:
        raise PreventUpdate
    fns = glob(os.path.join(target_dir, '*.*'))
    fns = [fn for fn in fns if not fn.endswith('.feather')]
    n_total = len(fns)
    for i, fn in enumerate( fns ):
        fsc.set('progress', int(100*(i+1)/n_total))
        new_fn = convert_ms_file_to_feather(fn)
        if os.path.isfile(new_fn): os.remove(fn)
    return 'Files converted to feather format.'


@app.callback(
Output('ms-delete-output', 'children'),
Input('ms-delete', 'n_clicks'),
[State('ms-table', 'selected_rows'),
 State('ms-table', 'data'),
 State('wdir', 'children')]
)
def ms_delete(n_clicks, ndxs, data, wdir):
    if n_clicks is None:
        raise PreventUpdate
    target_dir = os.path.join(wdir, 'ms_files')
    print(data)
    print(len(data))
    print(ndxs)

    for ndx in ndxs:
        fn = data[ndx]['MS-file']
        print('Delete file', fn)
        fn = os.path.join(target_dir, fn)
        os.remove(fn)
    return 'Files deleted'

# METADATA ############################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

@app.callback(
Output('meta-table', 'data'),
Output('meta-table', 'columns'),
Output('meta-column', 'options'),
Input('meta-upload', 'contents'),
Input('meta-apply-output', 'children'),
State('wdir', 'children')
)
def meta_upload(contents, message, wdir):
    print('METADATA')
    target_dir = os.path.join(wdir, 'ms_files')
    print(target_dir)
    ms_files = glob(os.path.join(target_dir, '*.*'), recursive=True)
    data = pd.DataFrame([{'MS-file': Basename(fn) } for fn in ms_files])
    print(data)
    fn = get_metadata_fn(wdir)
    if os.path.isfile(fn):
        metadata = pd.read_csv(fn)
        metadata['MS-file'] = [Basename(fn) for fn in metadata['MS-file'] ]
        data = pd.merge(data, metadata, on='MS-file', how='left')
    else:
        data['Label'] = ''
        data['Type'] = 'Biological Sample'
        data['Run Order'] = ''
        data['Batch'] = ''
        data['Row'] = ''
        data['Column'] = ''
    columns = [{'label':i, 'value':i} for i in data.columns]
    data = data.reset_index()
    print(data)
    print('Done')
    return data.to_dict('records'), gen_tabulator_columns(data.columns), columns


@app.callback(
Output('meta-apply-output', 'children'),
Input('meta-apply', 'n_clicks'),
Input('meta-table', 'dataSorted'),
State('meta-table', 'multiRowsClicked'),
State('meta-table', 'data'),
State('meta-table', 'dataFiltered'),
State('meta-action', 'value'),
State('meta-column', 'value'),
State('meta-input', 'value'),
State('wdir', 'children'),
)
def meta_save(n_clicks, tmp, selected_rows, data, 
        data_filtered, action, column, value, wdir):
    if n_clicks is None:
        raise PreventUpdate

    fn = get_metadata_fn( wdir )
    df = pd.DataFrame(data).set_index('index')

    if action == 'Set':
        filtered_rows = [r for r in data_filtered['rows'] if r is not None]
        filtered_ndx = [r['index'] for r in filtered_rows]
        ndxs = [r['index'] for r in selected_rows if r['index'] in filtered_ndx]
        if len(ndxs) == 0 or column is None:
            return 'No rows selected.'
        df.loc[ndxs, column] = value
    elif action == 'Create column': df[value] = ''
    elif action == 'Delete column': del df[column]

    df.to_csv(fn, index=False)
    return 'Data saved.'


# WORKSPACES ##########################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

@app.callback(
Output('ws-create-output', 'children'),
Input('ws-create', 'n_clicks'),
[State('ws-name', 'value'),
State('tmpdir', 'children')]
)
def create_workspace(n_clicks, ws_name, tmpdir):

    ws_names = get_dirnames(tmpdir)

    if ws_name is None or ws_name == '':
        raise PreventUpdate

    if not re.match('^[\w_-]+$', ws_name):
        return 'Name can only contain: a-z, A-Z, 0-9, -,  _'

    if ws_name not in ws_names:
        os.makedirs(os.path.join(tmpdir, ws_name))
        os.makedirs(os.path.join(tmpdir, ws_name, 'ms_files'))
        os.makedirs(os.path.join(tmpdir, ws_name, 'peaklist'))
        os.makedirs(os.path.join(tmpdir, ws_name, 'results'))
        os.makedirs(os.path.join(tmpdir, ws_name, 'figures'))
        os.makedirs(os.path.join(tmpdir, ws_name, 'chromato'))
        return f'Created workspace "{ws_name}"'

    return 'Nothing'

@app.callback(
Output('ws-table', 'data'),
Input('ws-create-output', 'children'),
Input('tab', 'value'),
Input('ws-delete-output', 'children'),
State('tmpdir', 'children')
)
def ws_table(value, tab, delete, tmpdir):
    ws_names = get_dirnames(tmpdir)
    ws_names =  [{'Workspace': ws_name} for ws_name in ws_names 
                    if not ws_name.startswith('.')]
    return ws_names

@app.callback(
[Output('ws-activate-output', 'children'),
 Output('wdir', 'children')],
Input('ws-activate', 'n_clicks'),
[State('ws-table', 'derived_virtual_selected_rows'),
 State('ws-table', 'data'),
 State('tmpdir', 'children')]
)
def ws_activate(n_clicks, ndx, data, tmpdir):
    fn_ws_info = os.path.join( tmpdir, '.active-workspace')
    if n_clicks is None and os.path.isfile(fn_ws_info):
        with open(fn_ws_info, 'r') as file:
            ws_name = file.read()
    else:
        if ndx is None or len(ndx) != 1:
            raise PreventUpdate
        ndx = ndx[0]
        ws_name = data[ndx]['Workspace']
    
    wdir = os.path.join(tmpdir, ws_name)
    message = f'Workspace __{ws_name}__ activated.'
    
    with open(fn_ws_info, 'w') as file:
        file.write(ws_name)
    
    fsc.set('progress', 0)

    return message, wdir


@app.callback(
 Output('progress-bar', 'value'),
 Input('progress-interval', 'n_intervals'),
)
def set_progress(n):
    return fsc.get('progress')


@app.callback(
Output("ws-delete-popup", "is_open"),
Input("ws-delete", "n_clicks"), 
Input("ws-delete-cancel", "n_clicks"),
Input('ws-delete-confirmed', 'n_clicks'),
State("ws-delete-popup", "is_open")
)
def ws_delete(n1, n2, n3, is_open):
    if n1 is None:
        raise PreventUpdate
    if n1 or n2 or n3:
        return not is_open
    return is_open


@app.callback(
Output('ws-delete-output', 'children'),
Input('ws-delete-confirmed', 'n_clicks'),
[State('ws-table', 'derived_virtual_selected_rows'),
 State('ws-table', 'data'),
 State('tmpdir', 'children')]
)
def ws_delete_confirmed(n_clicks, ndxs, data, tmpdir):
    if n_clicks is None or len(ndxs) == 0:
        raise PreventUpdate
    for ndx in ndxs:
        ws_name = data[ndx]['Workspace']
        dirname = os.path.join(tmpdir, ws_name)
        shutil.rmtree(dirname)
    message = f'Worskpace {ws_name} deleted.'
    return message



# PEAKLIST ############################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


@app.callback(
Output('pkl-table', 'data'),
Input('pkl-upload', 'contents'),
[State('pkl-upload', 'filename'),
 State('pkl-upload', 'last_modified'),
 State('wdir', 'children')]
)
def pkl_upload(list_of_contents, list_of_names, list_of_dates, wdir):
    target_dir = os.path.join(wdir, 'peaklist')
    fn = os.path.join( target_dir, 'peaklist.csv')
    if list_of_contents is not None:
        dfs = [parse_pkl_files(c, n, d, target_dir) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates) ]
        data = dfs[0].to_dict('records')    
        return data
    elif os.path.isfile(fn):
        return read_peaklists(fn).to_dict('records')   


@app.callback(
Output('pkl-save-output', 'children'),
[Input('pkl-save', 'n_clicks'),
 Input('pkl-table', 'data')],
State('wdir', 'children'))
def plk_save(n_clicks, data, wdir):
    target_dir = os.path.join(wdir, 'peaklist')
    df = pd.DataFrame(data)
    fn = os.path.join( target_dir, 'peaklist.csv')
    df.to_csv(fn)
    return 'Peaklist saved.'




# PEAK OPTIMIZATION ###################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


@app.callback(
Output('pko-dropdown', 'options'),
Input('tab', 'value'),
State('wdir', 'children')
)
def pko_controls(tab, wdir):
    if tab != 'pko':
        raise PreventUpdate
    peaklist = get_peaklist( wdir )
    if peaklist is None:
        raise PreventUpdate
    options = [{'label':label, 'value': i} for i, label in enumerate(peaklist.index)]
    return options


@app.callback(
Output('pko-figure', 'figure'),
[Input('pko-dropdown', 'value'),
 Input('pko-set-rt-output', 'children')],
State('wdir', 'children'),
State('pko-figure', 'figure')
)
def pko_figure(peak_label, n_clicks, wdir, fig):
    if peak_label is None:
        raise PreventUpdate
    
    peaklist = get_peaklist( wdir ).reset_index()
    ms_files = get_ms_fns( wdir )
    mz_mean, mz_width, rt, rt_min, rt_max, label = \
        peaklist.loc[peak_label, ['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max', 'peak_label']]

    if (rt_min is None) or np.isnan(rt_min): rt_min = rt-0.2
    if (rt_max is None) or np.isnan(rt_max): rt_max = rt+0.2
    if (rt is None) or np.isnan(rt): rt = np.mean([rt_min, rt_max])

    if True or fig is None:
        fig = go.Figure()
        fig.layout.hovermode = 'closest'
        fig.layout.xaxis.range=[rt_min, rt_max]

        fig.update_layout( 
            xaxis=dict( 
                rangeslider=dict( 
                    visible=True
                )
            )
        )
        fig.update_layout(title=label)

    #else:
    #    fig = go.Figure(fig)
    #    fig['data'] = []

    fig.add_vline(rt)
    fig.add_vrect(x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1)

    n_files = len(ms_files)
    for i, fn in tqdm(enumerate(ms_files)):
        fsc.set('progress', int(100*(i+1)/n_files))
        name = os.path.basename(fn)
        name, _ = os.path.splitext(name)
        chrom = get_chromatogram(fn, mz_mean, mz_width, wdir)
        fig.add_trace(
            go.Scatter(x=chrom['retentionTime'], 
                       y=chrom['intensity array'], 
                       name=name)
        )
        fig.update_layout(showlegend=False)
    return fig


@app.callback(
Output('pko-find-closest-peak-output', 'children'),
Input('pko-find-closest-peak', 'n_clicks'),
State('wdir', 'children'))
def pko_find_closest_peak(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    fn_pkl = os.path.join( wdir, 'peaklist', 'peaklist.csv') 
    
    def set_progress(x):
        fsc.set('progress', x)

    mint = Mint(verbose=True, progress_callback=set_progress)
    mint.peaklist_files = fn_pkl
    mint.ms_files = glob( os.path.join( wdir, 'ms_files', '*.*'))
    rtopt = RTOpt(mint)
    new_peaklist = rtopt.fit_transform(margin=1, how='closest')
    new_peaklist.to_csv(fn_pkl)


@app.callback(
Output('pko-set-rt-output', 'children'),
Input('pko-set-rt', 'n_clicks'),
[State('pko-dropdown', 'value'),
 State('pko-figure', 'figure'),
 State('wdir', 'children')]
)
def pko_set_rt(n_clicks, peak_label, fig, wdir):
    if n_clicks is None:
        raise PreventUpdate
    rt_min, rt_max = fig['layout']['xaxis']['range']
    rt_min, rt_max = np.round(rt_min, 4), np.round(rt_max, 4)
    update_peaklist(wdir, peak_label, rt_min, rt_max)
    return f'Set RT span to ({rt_min},{rt_max})'


@app.callback(
Output('pko-dropdown', 'value'),
Input('pko-prev', 'n_clicks'),
Input('pko-next', 'n_clicks'),
State('pko-dropdown', 'value'),
State('pko-dropdown', 'options')
)
def pko_prev_next(n_prev, n_next, value, options):
    if n_prev is None and n_next is None:
        raise PreventUpdate
    prop_id = dash.callback_context.triggered[0]['prop_id']
    if value is None:
        return 0
    if prop_id.startswith('pko-prev'):
        return (value - 1) % len(options)
    if prop_id.startswith('pko-next'):
        return (value + 1) % len(options)


def fig_to_src(dpi=100):
    out_img = io.BytesIO()   
    plt.savefig(out_img, format='png', bbox_inches='tight', dpi=dpi)
    plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


TIMEOUT = 60
@app.callback(
Output('pko-peak-preview-output', 'children'),
Input('pko-peak-preview', 'n_clicks'),
State('wdir', 'children')
)
@cache.memoize(timeout=TIMEOUT)
def peak_preview(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    sns.set_context('paper')
    ms_files = get_ms_fns(wdir)
    peaklist = get_peaklist(wdir)
    n_total = len(peaklist)
    images = []
    for i, (ndx, row) in tqdm( enumerate(peaklist.iterrows()), total=n_total ):
        fsc.set('progress', int(100*(i+1)/n_total))
        mz_mean, mz_width, rt, rt_min, rt_max = row[['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']]

        if np.isnan(rt_min) or rt_min is None: rt_min = rt-0.5
        if np.isnan(rt_max) or rt_max is None: rt_max = rt+0.5

        plt.figure(figsize=(3.5,2.5), dpi=50)
        for fn in ms_files:
            try:        
                fn_chro = get_chromatogram(fn, mz_mean, mz_width, wdir)
                fn_chro = fn_chro[(rt_min < fn_chro['retentionTime']) &
                                  (fn_chro['retentionTime'] < rt_max)]
                plt.plot(fn_chro['retentionTime'], fn_chro['intensity array'], lw=1, color='k')
            except:
                pass
        plt.title(ndx)
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))        
        src = fig_to_src()
        _id = f'peak-preview-{i}'
        images.append(html.Img(id=_id, src=src))
        images.append( dbc.Tooltip(ndx, target=_id, style={'font-size=': 'large'}) )
    images.append( dcc.Markdown('---') )
    return images


# QUALITY CONTROL #####################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


@app.callback(
Output('qc-figures', 'children'),
Input('qc-update', 'n_clicks'),
State('tab', 'value'),
State('qc-groupby', 'value'),
State('qc-graphs', 'value'),
State('qc-select', 'value'),
State('file-types', 'value'),
State('peak-labels', 'value'),
State('wdir', 'children')
)
def qc_figures(n_clicks, tab, groupby, kinds, options, file_types, peak_labels, wdir):
    if n_clicks is None:
        raise PreventUpdate

    df = get_complete_results( wdir )

    if file_types is not None and len(file_types) > 0:
        df = df[df.Type.isin(file_types)]
    if peak_labels is not None and len(peak_labels) > 0:
        df = df[df.peak_label.isin(peak_labels)]

    sns.set_context('paper')
    
    by_col = 'Label'
    by_col = 'Batch'
    quant_col = 'peak_max'
    quant_col = 'log(peak_max+1)'

    if by_col is not None:
        df = df.sort_values(['peak_label', by_col])

    if options is None:
        options = []

    figures = []
    n_total = len(df.peak_label.drop_duplicates())
    for i, (peak_label, grp) in tqdm( enumerate(df.groupby('peak_label')), total=n_total ):

        if not 'Dense' in options: figures.append(dcc.Markdown(f'#### `{peak_label}`', style={'float': 'center'}))
        fsc.set('progress', int(100*(i+1)/n_total))

        # Sorting to ensure similar legends
        if by_col is not None:
            grp = grp.sort_values(by_col).reset_index(drop=True)


        if len(grp) < 1:
            continue

        if 'hist' in kinds: 
            fig = sns.displot(data=grp, x=quant_col, height=3, hue=groupby, aspect=1)
            plt.title(peak_label)
            src = fig_to_src(dpi=150)
            figures.append( html.Img(src=src, style={'width': '300px'}) )

        if 'density' in kinds:
            fig = sns.displot(data=grp, x=quant_col, hue=groupby, kind='kde', common_norm=False, height=3,  aspect=1)
            plt.title(peak_label)
            src = fig_to_src(dpi=150)
            figures.append( html.Img(src=src, style={'width': '300px'}) )

        if 'boxplot' in kinds:
            n_groups = len( grp[groupby].drop_duplicates() )
            aspect = max(1, n_groups/10)
            fig = sns.catplot(data=grp, y=quant_col, x=groupby, height=3, kind='box', aspect=aspect, color='w')
            if quant_col in ['peak_max', 'peak_area']:
                plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.title(peak_label)
            plt.xticks(rotation=90)
            src = fig_to_src(dpi=150)
            figures.append( html.Img(src=src, style={'width': '300px'}) )

        if not 'Dense' in options: figures.append(dcc.Markdown('---'))

        #if i == 3: break

    return figures


@app.callback(
Output('file-types', 'options'),
Output('file-types', 'value'),
Input('tab', 'value'),
State('wdir', 'children')
)
def file_types(tab, wdir):
    if not tab in ['qc', 'heatmap']:
        raise PreventUpdate
    meta = get_metadata( wdir )
    if meta is None:
        raise PreventUpdate
    file_types = meta['Type'].drop_duplicates()
    options = [{'value': i, 'label': i} for i in file_types]
    return options, file_types


# HEATMAP #############################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


@app.callback(
Output('heatmap-controls', 'children'),
[Input('tab', 'value'),
 Input('res-delete-output', 'children')],
State('wdir', 'children')
)
def heat_controls(tab, delete, wdir):
    if tab != 'heatmap':
        raise PreventUpdate
    fn = get_results_fn(wdir)
    if os.path.isfile(fn):
        return heat_layout_non_empty
    else: 
        return heat_layout_empty


@app.callback(
Output('heatmap-figure', 'figure'),
Input('heatmap-update', 'n_clicks'),
State('file-types', 'value'),
State('heatmap-ms-order', 'value'),
State('heatmap-options', 'value'),
State('wdir', 'children')
)
def heat_heatmap(n_clicks, file_types, ms_order, options, wdir):
    mint = Mint()

    df = get_complete_results(wdir)
    if file_types is not None and file_types != []:
        df = df[df.Type.isin(file_types)]

    mint.results = df

    var_name = 'peak_max'
    data = mint.crosstab(var_name)
    data.index = [ Basename(i) for i in data.index ]

    if ms_order is not None:
        df = df.sort_values(ms_order)
        ms_files = df['MS-file'].drop_duplicates()
        data = data.loc[ms_files]

    data.fillna(0, inplace=True)

    name = var_name
    if 'log1p' in options:
        data = data.apply(np.log1p)
        name = f'log( {var_name}+1 )'

    fig = plot_heatmap(data, 
        normed_by_cols='normed_by_cols' in options, 
        transposed='transposed' in options, 
        clustered='clustered' in options,
        add_dendrogram='add_dendrogram' in options,
        correlation='correlation' in options, 
        call_show='call_show' in options,
        name=name)

    return fig



@app.callback(
Output('res-delete-output', 'children'),
Input('res-delete', 'n_clicks'),
State('wdir', 'children')
)
def heat_delete(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    os.remove(get_results_fn(wdir))
    return 'Results file deleted.'


@app.callback([
Output('res-download-data', 'data'),
Input('res-download', 'n_clicks'),
State('wdir', 'children')
])
def update_link(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    fn = get_results_fn(wdir)
    workspace = os.path.basename( wdir )
    return [send_file(fn, filename=f'{today()}-MINT-results_{workspace}.csv')]


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, dev_tools_hot_reload_interval=5000,
    dev_tools_hot_reload_max_retry=30)
