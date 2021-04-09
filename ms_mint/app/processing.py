import os

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.snippets import send_file

from dash.dependencies import Input, Output, State

from ms_mint.Mint import Mint

from . import tools as T

_label = 'Processing'

_layout = html.Div([
    html.H3('Run MINT'),
    html.Button('Run MINT',         id='run-mint'),
    html.Button('Download results', id='res-download'),
    html.Button('Delete results',   id='res-delete', style={'float': 'right'}),
])

_outputs = html.Div(id='run-outputs', children=[
    html.Div(id={'index': 'run-mint-output', 'type': 'output'}, style={'visibility': 'hidden'}),
])

def layout():
    return _layout


def callbacks(app, fsc, cache):

    @app.callback(
        Output('res-delete-output', 'children'),
        Input('res-delete', 'n_clicks'),
        State('wdir', 'children')
    )
    def heat_delete(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate
        os.remove( T.get_results_fn(wdir) )
        return dbc.Alert('Results file deleted.')


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
        return [send_file(fn, filename=f'{T.today()}-{workspace}_MINT-results.csv')]


    @app.callback(
        Output({'index': 'run-mint-output', 'type': 'output'}, 'children'),
        Input('run-mint', 'n_clicks'),
        State('wdir', 'children')
    )
    def run_mint(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate

        def set_progress(x):
            fsc.set('progress', x)

        mint = Mint(verbose=True, progress_callback=set_progress)
        mint.peaklist_files = T.get_peaklist_fn( wdir )
        mint.ms_files = T.get_ms_fns( wdir )
        mint.run()
        mint.export( T.get_results_fn(wdir) )
        return dbc.Alert('Finished running MINT', color='success')
