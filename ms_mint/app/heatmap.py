import numpy as np

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ms_mint.Mint import Mint
from ms_mint.vis.plotly.plotly_tools import plot_heatmap

from . import tools as T

_label = 'Heatmap'


heatmap_options = [
    { 'label': 'Normalized by biomarker', 'value': 'normed_by_cols'},
    { 'label': 'Cluster', 'value': 'clustered'},
    { 'label': 'Dendrogram', 'value': 'add_dendrogram'},
    { 'label': 'Transposed', 'value': 'transposed'},
    { 'label': 'Correlation', 'value': 'correlation'},
    { 'label': 'Show in new tab', 'value': 'call_show'},
    { 'label': 'log1p', 'value': 'log1p'},
]


_layout = html.Div([
    html.H3('Heatmap'),
    html.Button('Update', id='heatmap-update'),
    dcc.Dropdown(id='heatmap-options', value=['normed_by_cols'],
        options=heatmap_options, multi=True),
    dcc.Loading( 
        dcc.Graph(id='heatmap-figure', 
                  style={'marginTop': '50px'}) ),
])


def layout():
    return _layout


def callbacks(app, fsc, cache):

    @app.callback(
        Output('heatmap-controls', 'children'),
        Input('ana-secondary-tab', 'value'),
        State('wdir', 'children')
    )
    def heat_controls(tab, wdir):
        if tab != _label:
            raise PreventUpdate
        return _layout
        

    @app.callback(
    Output('heatmap-figure', 'figure'),
    Input('heatmap-update', 'n_clicks'),
    State('ana-file-types', 'value'),
    State('ana-peak-labels-include', 'value'),
    State('ana-peak-labels-exclude', 'value'),
    State('ana-ms-order', 'value'),
    State('heatmap-options', 'value'),
    State('wdir', 'children')
    )
    def heat_heatmap(n_clicks, file_types, include_labels, exclude_labels, 
            ms_order, options, wdir):
        mint = Mint()

        df = T.get_complete_results( wdir, include_labels=include_labels, 
                exclude_labels=exclude_labels, file_types=file_types )

        if len(df) == 0: return 'No results yet. First run MINT.'

        mint.results = df

        var_name = 'peak_max'
        data = mint.crosstab(var_name)
        data.index = [ T.Basename(i) for i in data.index ]

        if ms_order is not None and len(ms_order)>0:
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



