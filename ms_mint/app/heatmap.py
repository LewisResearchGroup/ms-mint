import os
import numpy as np

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ms_mint.Mint import Mint
from ms_mint.vis.plotly.plotly_tools import plot_heatmap

from . import tools as T

heat_layout = html.Div([
    html.Div(id='res-upload-output'),
    html.Div(id='heatmap-controls'),
    html.Div(id='heatmap-output')
])


heatmap_options = [
    { 'label': 'Normalized by biomarker', 'value': 'normed_by_cols'},
    { 'label': 'Cluster', 'value': 'clustered'},
    { 'label': 'Dendrogram', 'value': 'add_dendrogram'},
    { 'label': 'Transposed', 'value': 'transposed'},
    { 'label': 'Correlation', 'value': 'correlation'},
    { 'label': 'Show in new tab', 'value': 'call_show'},
    { 'label': 'log1p', 'value': 'log1p'},
    ]


heat_layout_empty = html.Div([
    dcc.Upload(
            id='res-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
])


_layout = html.Div([
    html.H3('Heatmap'),
    html.Button('Update', id='heatmap-update'),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Dropdown(id='heatmap-options', value=['normed_by_cols', 'clustered'],
        options=heatmap_options, multi=True),
    dcc.Dropdown(id='heatmap-ms-order', options=[], placeholder='MS-file sorting', multi=True),
    dcc.Loading( 
        dcc.Graph(id='heatmap-figure', 
                  style={'margin-top': '50px'}) ),
])


def layout():
    return _layout


def callbacks(app, fsc, cache):

    @app.callback(
    Output('heatmap-controls', 'children'),
    [Input('tab', 'value'),
    Input('res-delete-output', 'children')],
    State('wdir', 'children')
    )
    def heat_controls(tab, delete, wdir):
        if tab != 'heatmap':
            raise PreventUpdate
        fn = T.get_results_fn(wdir)
        if os.path.isfile(fn):
            return _layout
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

        df = T.get_complete_results(wdir)
        if len(df) == 0: return 'No results yet. First run MINT.'

        if file_types is not None and file_types != []:
            df = df[df.Type.isin(file_types)]
        
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

    @app.callback(
        Output('heatmap-ms-order', 'options'),
        Input('tab', 'value'),
        State('wdir', 'children')
    )
    def ms_order_options(tab, wdir):
        if not tab == 'heatmap': raise PreventUpdate
        cols = T.get_metadata(wdir).dropna(how='all', axis=1).columns.to_list()
        if 'index' in cols: cols.remove('index')
        options = [{'value':i, 'label': i} for i in cols]
        return options

