import os

import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
from dash_table import DataTable
from dash.dependencies import Input, Output, State

from . import tools as T

from ms_mint.standards import PEAKLIST_COLUMNS
from ms_mint.peaklists import read_peaklists


columns = [{"name": i, "id": i, 
            "selectable": True}  for i in PEAKLIST_COLUMNS]

pkl_table = DataTable(
                id='pkl-table',
                columns=columns,
                data=None,
                sort_action="native",
                sort_mode="single",
                row_selectable=False,
                row_deletable=True,
                editable=True,
                column_selectable=False,
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 30,
                filter_action='native',                
                style_table={'overflowX': 'scroll'},
                style_as_list_view=True,
                style_cell={'padding-left': '5px', 
                            'padding-right': '5px'},
                style_header={'backgroundColor': 'white',
                              'fontWeight': 'bold'},
                export_format='csv',
                export_headers='display',
                merge_duplicate_headers=True
            ) 


_layout = html.Div([
    html.H3('Peaklist'),

    dcc.Upload(
            id='pkl-upload',
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
            multiple=True
        ),
    dcc.Dropdown('pkl-ms-mode', options=[
        {'value': 'positive', 'label': 'Add proton mass to formula (positive mode)'},
        {'value': 'negative', 'label': 'Subtract proton mass from formula (negative mode)'}], value=None),
    html.Div(id='pkl-upload-output'),
    html.Div(id='pkl-save-output'),
    html.Button('Save peaklist', id='pkl-save'),
    html.Button('Delete selected peaks', id='pkl-delete', style={'float': 'right'}),
    pkl_table,
])


def layout():
    return _layout


def callbacks(app, fsc=None, cache=None):
    @app.callback(
    Output('pkl-table', 'data'),
    Input('pkl-upload', 'contents'),
    Input('pkl-ms-mode', 'value'),
    State('pkl-upload', 'filename'),
    State('pkl-upload', 'last_modified'),
    State('wdir', 'children')
    )
    def pkl_upload(list_of_contents, ms_mode, list_of_names, list_of_dates, wdir):
        target_dir = os.path.join(wdir, 'peaklist')
        fn = os.path.join( target_dir, 'peaklist.csv')
        if list_of_contents is not None:
            dfs = [T.parse_pkl_files(c, n, d, target_dir, ms_mode=ms_mode) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates) ]
            data = dfs[0].to_dict('records')    
            return data
        elif os.path.isfile(fn):
            return read_peaklists(fn).to_dict('records')   

    @app.callback(
        Output('pkl-save-output', 'children'),
        Input('pkl-save', 'n_clicks'),
        Input('pkl-table', 'data'),
        State('wdir', 'children')
    )
    def plk_save(n_clicks, data, wdir):
        target_dir = os.path.join(wdir, 'peaklist')
        df = pd.DataFrame(data)
        fn = os.path.join( target_dir, 'peaklist.csv')
        df = df.sort_values('peak_label')
        df.to_csv(fn)
        return 'Peaklist saved.'

  

