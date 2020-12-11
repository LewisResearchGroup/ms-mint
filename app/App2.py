import os

from glob import glob

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable

from ms_mint.Mint import Mint
from tools import parse_ms_files, get_dirnames
from ms_files import ms_layout
from workspaces import ws_layout
import shutil
import tempfile

def make_tmpdir():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, 'MINT')
    os.makedirs(tmpdir, exist_ok=True)
    return tmpdir

TMPDIR = make_tmpdir()

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP, 
    "https://codepen.io/chriddyp/pen/bWLwgP.css"
                                                ])
app.title = 'MINT'

app.config['suppress_callback_exceptions'] = True

layout = html.Div([
    dcc.Interval(id="interval", n_intervals=0, interval=500, disabled=False),
    dbc.Progress(id="progress-bar", value=100, style={'margin-bottom': '20px'}),
    html.Div(id='tmpdir', children=TMPDIR),
    html.Div(id='wdir', children=''),
    dcc.Tabs(id='tab', value='msfiles', children=[
        dcc.Tab(label='Workspace', value='workspaces'),
        dcc.Tab(label='MS-files', value='msfiles'),
        dcc.Tab(label='Peaklist', value='peaklist'),
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
        return html.Div([
            html.H3('Tab content 2')
        ])

@app.callback(
Output('ms-upload-output', 'children'),
Input('ms-upload', 'contents'),
State('ms-upload', 'filename'),
State('ms-upload', 'last_modified'))
def ms_upload(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_ms_files(c, n, d, TMPDIR) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates) ]
        return html.H1('Done')
    
@app.callback(
Output('ms-table', 'data'),
Input('ms-upload-output', 'children')
)
def ms_table(value):
    ms_files = glob(os.path.join(TMPDIR, '*.mz*ML'), recursive=True)
    ms_files =  [{'MS-files': os.path.basename(fn) } for fn in ms_files]
    return ms_files



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

    if ws_name not in ws_names:
        os.makedirs(os.join(tmpdir, ws_name))
        return f'Created workspace "{ws_name}"'

    return 'Nothing'

@app.callback(
Output('ws-table', 'data'),
Input('ws-create-output', 'children'),
State('tmpdir', 'children')
)
def ws_table(value, tmpdir):
    ws_names = get_dirnames(tmpdir)
    ws_names =  [{'Workspace': ws_name} for ws_name in ws_names]
    return ws_names




if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
