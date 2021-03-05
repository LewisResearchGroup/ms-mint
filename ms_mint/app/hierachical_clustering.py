import os
import numpy as np

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ms_mint.notebook import Mint
from ms_mint.vis.plotly.plotly_tools import plot_heatmap

from . import tools as T


_label='Hierachical Clustering'

_options = ['Transposed']

options = [{'value': x, 'label': x} for x in _options]

_layout = html.Div([
    html.H3(_label),

    dcc.Input(id='hc-figsize-x', placeholder='Figure size x', value=8, type="number"),
    dcc.Input(id='hc-figsize-y', placeholder='Figure size x', value=8, type="number"),
    dcc.Dropdown(id='hc-options', options=options, value=[]),

    html.Button('Update', id='hc-update'),

    dcc.Loading( 
        html.Div(id='hc-figures',
         style={'margin': 'auto', 
                'text-align': 'center',
                'maxWidth': '100%',
                'minHeight': '300px'}) )

])

_ouptuts = html.Div([])

def layout():
    return _layout


def callbacks(app, fsc, cache):
    
    @app.callback(
        Output('hc-figures', 'children'),
        Input('hc-update', 'n_clicks'),
        State('hc-figsize-x', 'value'),
        State('hc-figsize-y', 'value'),
        State('hc-options', 'value'),
        State('wdir', 'children')
    )
    def create_figure(n_clicks, fig_size_x, fig_size_y, options, wdir):
        if n_clicks is None: raise PreventUpdate

        mint = Mint()

        if fig_size_x is None: fig_size_x = 8
        if fig_size_y is None: fig_size_y = 8

        fig_size_x = min(float(fig_size_x), 100)
        fig_size_y = min(float(fig_size_y), 100)

        print(fig_size_x, fig_size_y)

        mint.results = T.get_results( wdir )

        mint.plot_clustering(figsize=(fig_size_x, fig_size_y), 
            transpose='Transposed' in options)

        src = T.fig_to_src()
        print('HC figure created.')
        return html.Img(src=src, style={'maxWidth': '80%'})