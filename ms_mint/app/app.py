
import os

import tempfile

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
from dash_extensions.enrich import FileSystemCache

from flask_caching import Cache

from . import tools as T

from . import workspaces
from . import ms_files
from . import metadata
from . import peaklist
from . import peak_optimization
from . import quality_control
from . import heatmap
from . import run_mint
from . import add_metab

import dash_uploader as du
from tempfile import gettempdir

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
    'add_metab':    {'label': 'Add Metabolites',    'callbacks_func': add_metab.callbacks,          'layout_func': add_metab.layout},
    'pko':          {'label': 'Peak Optimization',  'callbacks_func': peak_optimization.callbacks,  'layout_func': peak_optimization.layout},
    'run':          {'label': 'Run MINT',           'callbacks_func': run_mint.callbacks,           'layout_func': run_mint.layout},
    'qc':           {'label': 'Statistics',         'callbacks_func': quality_control.callbacks,    'layout_func': quality_control.layout},
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


UPLOAD_FOLDER_ROOT = gettempdir()
du.configure_upload(app, UPLOAD_FOLDER_ROOT)


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/MINT-cache'
})

app.title = 'MINT'

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    html.Img(src=app.get_asset_url('logo.png'), style={'height': '30px'}),
    dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=False),
    html.A(href='https://soerendip.github.io/ms-mint/gui/', 
         children=[html.Button('Documentation', id='B_help', style={'float': 'right'})],
         target="_blank"),
    html.A(href=f'https://github.com/soerendip/ms-mint/issues/new?body={T.get_issue_text()}', 
         children=[html.Button('Issues', id='B_issues', style={'float': 'right'})],
         target="_blank"),   
    Download(id='res-download-data'),
    dbc.Progress(id="progress-bar", value=100, style={'margin-bottom': '20px', 'width': '100%'}),
    dcc.Markdown(id='res-delete-output'),
    dcc.Markdown(id='run-mint-output', style={'float': 'center', }),
    html.Div(id='tmpdir', children=TMPDIR, style={'visibility': 'hidden'}),
    html.P('Current Workspace: ', style={'display': 'inline-block', 'margin-right': '5px'}),
    html.Div(id='active-workspace', style={'display': 'inline-block'}),
    html.Div(id='wdir', children=TMPDIR, style={'display': 'inline-block', 'visibility': 'visible', 'float': 'right'}),
    html.Div(id='pko-creating-chromatograms'),
    dcc.Tabs(id='tab', value='workspaces',  #vertical=True, style={'display': 'inline-block'},
        children=[
            dcc.Tab(value=key, 
                    label=components[key]['label'],
                    )
            for key in components.keys()]
    ),
    html.Div(id='pko-image-store', style={'visibility': 'hidden', 'height': '0px'}),
    html.Div(id='tab-content', style={'margin': '5%'})
], style={'margin':'2%'})


for component in components.values():
    component['callbacks_func'](app=app, fsc=fsc, cache=cache)


@app.callback(
    Output('tab-content', 'children'),
    Input('tab', 'value'),
    State('wdir', 'children')
)
def render_content(tab, wdir):
    return components[tab]['layout_func']()


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


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, 
        dev_tools_hot_reload_interval=5000,
        dev_tools_hot_reload_max_retry=30)
