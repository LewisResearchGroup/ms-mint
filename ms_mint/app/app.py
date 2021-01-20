
import os

import tempfile
import platform

from glob import glob

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.snippets import send_file
from dash_extensions.enrich import FileSystemCache

from flask_caching import Cache

from ms_mint.Mint import Mint

import ms_mint

from . import tools as T

from . import ms_files
from . import workspaces
from . import metadata
from . import peaklist
from . import peak_optimization
from . import quality_control
from . import heatmap

from dash_uploader import configure_upload

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

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}



pd.options.display.max_colwidth = 1000


components = {
    'workspaces':   {'label': 'Workspace',          'callbacks_func': workspaces.callbacks,         'layout_func': workspaces.layout},
    'msfiles':      {'label': 'MS-files',           'callbacks_func': ms_files.callbacks,           'layout_func': ms_files.layout},
    'metadata':     {'label': 'Metadata',           'callbacks_func': metadata.callbacks,           'layout_func': metadata.layout},
    'peaklist':     {'label': 'Peaklist',           'callbacks_func': peaklist.callbacks,           'layout_func': peaklist.layout},
    'pko':          {'label': 'Peak Optimization',  'callbacks_func': peak_optimization.callbacks,  'layout_func': peak_optimization.layout},
    'qc':           {'label': 'Quality Control',    'callbacks_func': quality_control.callbacks,    'layout_func': quality_control.layout},
    'heatmap':      {'label': 'Heatmap',            'callbacks_func': heatmap.callbacks,            'layout_func': heatmap.layout},
}



app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        "https://codepen.io/chriddyp/pen/bWLwgP.css"],
    requests_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/'),
    routes_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/')
    )

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

UPLOAD_FOLDER_ROOT = "/tmp"
configure_upload(app, UPLOAD_FOLDER_ROOT)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/MINT-cache'
})

app.title = 'MINT'

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=False),
    html.Button('Run MINT', id='run-mint'),
    html.Button('Download results', id='res-download'),
    html.Button('Delete results', id='res-delete'),
    html.A(href='https://soerendip.github.io/ms-mint/gui/', 
         children=[html.Button('Documentation', id='B_help', style={'float': 'right'})],
         target="_blank"),
    html.A(href=f'https://github.com/soerendip/ms-mint/issues/new?body={T.get_issue_text()}', 
         children=[html.Button('Issues', id='B_issues', style={'float': 'right'})],
         target="_blank"),   
    Download(id='res-download-data'),
    html.Div(id='run-mint-output'),
    html.Div(id='active-workspace'),
    dcc.Markdown(id='res-delete-output'),
    dbc.Progress(id="progress-bar", value=100, style={'margin-bottom': '20px'}),
    html.Div(id='tmpdir', children=TMPDIR, style={'visibility': 'hidden'}),
    html.P('Current directory:  ', style={'display': 'inline-block', 'margin-right': '5px'}),
    html.Div(id='wdir', children=TMPDIR, style={'display': 'inline-block'}),
    dcc.Tabs(id='tab', value='workspaces', children=[dcc.Tab(value=key, label=components[key]['label']) for key in components.keys()]),
    html.Div(id='tab-content', style={'margin': '5%'})
], style={'margin':'2%'})


@app.callback(Output('tab-content', 'children'),
              Input('tab', 'value'),
              State('wdir', 'children'))
def render_content(tab, wdir):
    return components[tab]['layout_func']()

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
    peaklist = T.get_peaklist( wdir ).reset_index()
    peak_labels = [{'value': i, 'label': i} for i in peaklist.peak_label]
    return peak_labels



@app.callback(
Output('res-delete-output', 'children'),
Input('res-delete', 'n_clicks'),
State('wdir', 'children')
)
def heat_delete(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    os.remove( T.get_results_fn(wdir) )
    return 'Results file deleted.'


@app.callback([
Output('res-download-data', 'data'),
Input('res-download', 'n_clicks'),
State('wdir', 'children')
])
def update_link(n_clicks, wdir):
    if n_clicks is None:
        raise PreventUpdate
    fn = T.get_results_fn(wdir)
    workspace = os.path.basename( wdir )
    return [send_file(fn, filename=f'{T.today()}-MINT-results_{workspace}.csv')]

for component in components.values():
    component['callbacks_func'](app=app, fsc=fsc, cache=cache)

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, 
        dev_tools_hot_reload_interval=5000,
        dev_tools_hot_reload_max_retry=30)
