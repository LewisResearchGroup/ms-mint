import os
import re

import shutil
import tempfile

from glob import glob

import pandas as pd
import numpy as np

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable
from dash_extensions.enrich import Output, Dash, Trigger, FileSystemCache

import plotly.graph_objects as go
import plotly.express as px

from ms_mint.Mint import Mint
from ms_mint.peaklists import read_peaklists
from ms_mint.io import convert_ms_file_to_feather

from .tools import parse_ms_files, get_dirnames, parse_pkl_files, get_chromatogram, create_chromatograms,\
    get_metadata_fn, get_results_fn, update_peaklist, get_peaklist, get_peaklist_fn

from .ms_files import ms_layout
from .workspaces import ws_layout
from .peaklist import pkl_layout
from .results import res_layout, res_layout_empty, res_layout_non_empty
from .peak_optimization import pko_layout


from tqdm import tqdm

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, 'MINT')
    cachedir = os.path.join(tmpdir, '.cache')
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    return tmpdir, cachedir

TMPDIR, CACHEDIR = make_dirs()
fsc = FileSystemCache(CACHEDIR)

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP, 
    "https://codepen.io/chriddyp/pen/bWLwgP.css"
                                                ])
app.title = 'MINT'

app.config['suppress_callback_exceptions'] = True

layout = html.Div([
    dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=False),
    html.Button('Run MINT', id='run-mint'),
    html.Div(id='run-mint-output'),
    dbc.Progress(id="progress-bar", value=100, style={'margin-bottom': '20px'}),
    html.Div(id='tmpdir', children=TMPDIR, style={'visibility': 'hidden'}),
    html.Div(id='wdir', children=TMPDIR),
    dcc.Tabs(id='tab', value='workspaces', children=[
        dcc.Tab(label='Workspace', value='workspaces'),
        dcc.Tab(label='MS-files', value='msfiles'),
        dcc.Tab(label='Peaklist', value='peaklist'),
        dcc.Tab(label='Peak Optimization', value='pko'),
        dcc.Tab(label='Results', value='results'),
    ]),
    html.Div(id='tab-content', style={'margin': '5%'})
], style={'margin':'2%'})

app.layout = layout

@app.callback(Output('tab-content', 'children'),
              Input('tab', 'value'))
def render_content(tab):
    if tab == 'msfiles':
        return ms_layout
    elif tab == 'workspaces':
        return ws_layout
    elif tab == 'peaklist':
        return pkl_layout
    elif tab == 'results':
        return res_layout
    elif tab == 'pko':
        return pko_layout


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

    mint = Mint(verbose=True, progress_callback=set_progress)
    mint.peaklist_files = os.path.join(wdir, 'peaklist', 'peaklist.csv')
    mint.ms_files = glob( os.path.join(wdir, 'ms_files', '*.*'))
    mint.run()
    mint.export( os.path.join(wdir, 'results', 'results.csv'))

# MS-FILES
@app.callback(
Output('ms-upload-output', 'children'),
Input('ms-upload', 'contents'),
State('ms-upload', 'filename'),
State('ms-upload', 'last_modified'),
State('wdir', 'children'))
def ms_upload(list_of_contents, list_of_names, list_of_dates, wdir):
    target_dir = os.path.join(wdir, 'ms_files')
    if list_of_contents is not None:
        n_total = len(list_of_contents)
        n_uploaded = 0 
        for c, n, d in tqdm( zip(list_of_contents, list_of_names, list_of_dates), total=n_total ):
            try:
                parse_ms_files(c, n, d, target_dir)
                n_uploaded += 1
            except:
                pass
        return html.P(f'{n_uploaded} files uploaded.')
    

@app.callback(
Output('ms-table', 'data'),
[Input('ms-upload-output', 'children'),
 Input('wdir', 'children'), 
 Input('ms-delete-output', 'children'),
 Input('ms-set-labels', 'n_clicks')],
[State('ms-input', 'value'),
 State('ms-table', 'derived_virtual_selected_rows'),
 State('ms-table', 'derived_virtual_indices')
]
)
def ms_table(value, wdir, files_deleted, set_labels, ms_input, 
    ndxs_selected, ndxs_filtered):   

    target_dir = os.path.join(wdir, 'ms_files')
    ms_files = glob(os.path.join(target_dir, '*.*'), recursive=True)
    data =  pd.DataFrame([{'MS-file': os.path.basename(fn) } for fn in ms_files])

    fn = get_metadata_fn(wdir)
    if os.path.isfile(fn):
        metadata = pd.read_csv(fn)
        data = pd.merge(data, metadata, on='MS-file', how='left')

    prop_id = dash.callback_context.triggered[0]['prop_id']

    if prop_id.startswith('ms-set-labels'):
        if ndxs_selected is []:
            ndxs = ms-set-labels
        else: ndxs = ndxs_filtered
        data.loc[ndxs, 'Label'] = ms_input
    return data.to_dict('records')


@app.callback(
Output('ms-save-output', 'children'),
Input('ms-table', 'data'),
State('wdir', 'children')
)
def ms_save_meta(data, wdir):
    if data is None:
        raise PreventUpdate
    fn = get_metadata_fn(wdir)
    if len(data) > 0:
        pd.DataFrame(data).to_csv(fn, index=False)
        return 'Metadata saved.'
    return ''


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
    print(fns)
    for fn in fns: 
        print('Convert to feather:', fn)
        new_fn = convert_ms_file_to_feather(fn)
        if os.path.isfile(new_fn): os.remove(fn)
    return 'Files converted to feather format.'

@app.callback(
Output('ms-delete-output', 'children'),
Input('ms-delete', 'n_clicks'),
[State('ms-table', 'derived_virtual_selected_rows'),
 State('ms-table', 'data'),
 State('wdir', 'children')]
)
def ms_delete(n_clicks, ndxs, data, wdir):
    if n_clicks is None:
        raise PreventUpdate
    target_dir = os.path.join(wdir, 'ms_files')
    for ndx in ndxs:
        fn = data[ndx]['MS-file']
        fn = os.path.join(target_dir, fn)
        os.remove(fn)
    return 'Files deleted'


# WORKSPACES
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
        os.makedirs(os.path.join(tmpdir, ws_name, 'chromatograms'))
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
        if len(ndx) != 1:
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



# PEAKLIST
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



# PEAK OPTIMIZATION
@app.callback(
Output('pko-dropdown', 'options'),
Input('tab', 'value'),
State('wdir', 'children')
)
def pko_controls(tab, wdir):
    if tab != 'pko':
        raise PreventUpdate
    peaklist = pd.read_csv( os.path.join(wdir, 'peaklist', 'peaklist.csv') )
    ms_files = glob( os.path.join( wdir, 'ms_files', '*.*') )
    options = [{'label':i, 'value': i} for i in peaklist.peak_label]
    #create_chromatograms(ms_files, peaklist, wdir)
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
    
    peaklist = pd.read_csv( os.path.join(wdir, 'peaklist', 'peaklist.csv') )
    peaklist = peaklist.set_index('peak_label')
    ms_files = glob( os.path.join( wdir, 'ms_files', '*.*') )
    mz_mean, mz_width, rt, rt_min, rt_max = \
        peaklist.loc[peak_label, ['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']]


    if (rt_min is None) or np.isnan(rt_min): rt_min = rt-0.2
    if (rt_max is None) or np.isnan(rt_max): rt_max = rt+0.2
    if (rt is None) or np.isnan(rt): rt = np.mean([rt_min, rt_max])

    #print(peak_label, mz_mean, mz_width, rt, rt_min, rt_max )

    if True or fig is None:
        fig = go.Figure()
        fig.layout.hovermode = 'closest'
        rangeslider=dict( 
            visible=True
        ),
        
        fig.layout.xaxis.range=[rt_min, rt_max]

        fig.update_layout( 
            xaxis=dict( 
                rangeslider=dict( 
                    visible=True
                )
            )
        )
        fig.update_layout(title=peak_label)

    #else:
    #    fig = go.Figure(fig)
    #    fig['data'] = []

    fig.add_vline(rt)
    fig.add_vrect(x0=rt_min, x1=rt_max, line_width=0, fillcolor="blue", opacity=0.1)

    n_files = len(ms_files)
    for i, fn in tqdm(enumerate(ms_files)):
        fsc.set('progress', int(100*i/n_files))
        name = os.path.basename(fn)
        name, _ = os.path.splitext(name)
        chrom = get_chromatogram(fn, mz_mean, mz_width, wdir)
        fig.add_trace(
            go.Scatter(x=chrom.retentionTime, y=chrom['intensity array'], name=name)
        )


    return fig


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
    return f'Set RT span for __{peak_label}__ to ({rt_min},{rt_max})'



# RESULTS
@app.callback(
Output('res-controls', 'children'),
[Input('tab', 'value'),
 Input('res-delete-output', 'children')],
State('wdir', 'children')
)
def res_controls(tab, delete, wdir):
    if tab != 'results':
        raise PreventUpdate
    fn = get_results_fn(wdir)
    if os.path.isfile(fn):
        return res_layout_non_empty
    else: 
        return res_layout_empty

@app.callback(
Output('res-delete-output', 'children'),
Input('res-delete', 'n_clicks'),
State('wdir', 'children')
)
def res_delete(n_clicks, wdir):
    os.remove(get_results_fn(wdir))
    return 'Results file deleted.'


from ms_mint.vis.plotly.plotly_tools import plot_peak_shapes, plot_peak_shapes_3d, plot_heatmap

@app.callback(
Output('res-heatmap-figure', 'figure'),
Input('res-heatmap', 'n_clicks'),
[State('res-heatmap-options', 'value'),
 State('wdir', 'children')]
)
def res_heatmap(n_clicks, options, wdir):
    fn = get_results_fn(wdir)
    mint = Mint()
    mint.load(fn)

    data = mint.crosstab('peak_max')



    data.fillna(0, inplace=True)
    fig = plot_heatmap(data, 
        normed_by_cols='normed_by_cols' in options, 
        transposed='transposed' in options, 
        clustered='clustered' in options,
        add_dendrogram='add_dendrogram' in options,
        correlation='correlation' in options, 
        call_show='call_show' in options,
        name='peak_max')
    return fig




if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, dev_tools_hot_reload_interval=5000,
    dev_tools_hot_reload_max_retry=30)
