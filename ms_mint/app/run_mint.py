import os
from glob import glob 

import dash_html_components as html
#import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash_extensions.snippets import send_file

from dash.dependencies import Input, Output, State

from ms_mint.Mint import Mint

from . import tools as T


_layout = html.Div([
    html.H3('Run MINT'),
    html.Button('Run MINT',         id='run-mint'),
    html.Button('Download results', id='res-download'),
    html.Button('Delete results',   id='res-delete', style={'float': 'right'}),
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
        return '''`Success`'''
