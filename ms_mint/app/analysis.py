
import dash_html_components as html
#import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from . import heatmap
from . import pca
from . import quality_control
from . import tools as T

components = {
    'heatmap':      {'label': 'Heatmap',            'callbacks_func': heatmap.callbacks,            'layout_func': heatmap.layout},
    'qc':           {'label': 'Statistics',         'callbacks_func': quality_control.callbacks,    'layout_func': quality_control.layout},
    'pca':          {'label': 'Decomposition',      'callbacks_func': pca.callbacks,                'layout_func': pca.layout}
}


groupby_options = [{'label': 'Batch', 'value': 'Batch'},
                   {'label': 'Label', 'value': 'Label'},
                   {'label': 'Type',  'value': 'Type'},
                   {'label': 'Color', 'value': 'Color'}]


_layout = html.Div([
    dcc.Tabs(id='secondary-tab', value='heatmap', vertical=False, 
        children=[
            dcc.Tab(value=key, 
                    label=components[key]['label'],
                    )
            for key in components.keys()]
    ),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Dropdown(id='peak-labels-include', options=[], placeholder='Include peak_labels', multi=True),
    dcc.Dropdown(id='peak-labels-exclude', options=[], placeholder='Exclude peak_labels', multi=True),    
    dcc.Dropdown(id='ms-order', options=[], placeholder='MS-file sorting', multi=True),
    dcc.Dropdown(id='qc-groupby', options=groupby_options, value=None, placeholder='Group by column'),
    html.Div(id='secondary-tab-content')
])


def layout():
    return _layout


def callbacks(app, fsc, cache):

    for component in components.values():
        func = component['callbacks_func']
        if func is not None:
            func(app=app, fsc=fsc, cache=cache)

    @app.callback(
        Output('secondary-tab-content', 'children'),
        Input('secondary-tab', 'value'),
        State('wdir', 'children')
    )
    def render_content(tab, wdir):
        func = components[tab]['layout_func']
        if func is not None:
            return func()
        else:
            raise PreventUpdate


    @app.callback(
    Output('file-types', 'options'),
    Output('file-types', 'value'),
    Input('tab', 'value'),
    State('wdir', 'children')
    )
    def file_types(tab, wdir):
        if not tab in ['qc', 'analysis']:
            raise PreventUpdate
        meta = T.get_metadata( wdir )
        if meta is None:
            raise PreventUpdate
        file_types = meta['Type'].drop_duplicates()
        options = [{'value': i, 'label': i} for i in file_types]
        return options, file_types


    @app.callback(
        Output('ms-order', 'options'),
        Input('secondary-tab', 'value'),
        State('wdir', 'children')
    )
    def ms_order_options(tab, wdir):
        if not tab == 'heatmap': raise PreventUpdate
        cols = T.get_metadata(wdir).dropna(how='all', axis=1).columns.to_list()
        if 'index' in cols: cols.remove('index')
        options = [{'value':i, 'label': i} for i in cols]
        return options


    @app.callback(
        Output('peak-labels-include', 'options'),
        Output('peak-labels-exclude', 'options'),
        Input('tab', 'value'),
        State('wdir', 'children')
    )
    def peak_labels(tab, wdir):
        if tab not in ['analysis']:
            raise PreventUpdate
        peaklist = T.get_peaklist( wdir ).reset_index()
        options = [{'value': i, 'label': i} for i in peaklist.peak_label]
        return options, options