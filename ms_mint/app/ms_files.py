import os
from glob import glob
import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

from dash_table import DataTable
from dash.dependencies import Input, Output, State

from ms_mint.io import convert_ms_file_to_feather

from . import tools as T


ms_table = html.Div(id='ms-table-container', 
    style={'min-height':  100, 'margin': '5%'},
    children=[
        DataTable(id='ms-table',
            columns=[ {"name": i, "id": i, "selectable": True, 'editable': i != 'MS-file'}  
                                for i in ['MS-file']],
                    data=None,
                    row_selectable='multi',
                    row_deletable=False,
                    style_cell={'textAlign': 'left'},
                    sort_action='native',
                    filter_action='native',                
                    )
])

_layout = html.Div([
    html.H3('MS-file'),
    dcc.Markdown('''At the moment the upload is limited to ~10 files at a time.
    '''),
    dcc.Upload(
            id='ms-upload',
            children=html.Div([
                html.A('Drag and Drop or click to select files')
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
    html.Button('Convert to Feather', id='ms-convert'),
    html.Button('Delete selected files', id='ms-delete', style={'float': 'right'}),
    dcc.Loading( html.Div(id='ms-upload-output') ),
    html.Div(id='ms-convert-output'),
    html.Div(id='ms-delete-output'),
    html.Div(id='ms-save-output'),
    dcc.Loading( ms_table )
])


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
    Output('ms-upload-output', 'children'),
    Input('ms-upload', 'contents'),
    Input('ms-convert-output', 'children'),
    State('ms-upload', 'filename'),
    State('ms-upload', 'last_modified'),
    State('wdir', 'children'))
    def ms_upload(list_of_contents, converted, list_of_names, list_of_dates, wdir):
        target_dir = os.path.join(wdir, 'ms_files')
        if list_of_contents is not None:
            n_total = len(list_of_contents)
            n_uploaded = 0
            for i, (c, n, d) in enumerate( zip(list_of_contents, list_of_names, list_of_dates) ):
                fsc.set('progress', int( 100*(i+1)/n_total ))
                if n.lower().endswith('mzxml') or n.lower().endswith('mzml'):
                    try:
                        T.parse_ms_files(c, n, d, target_dir)
                        n_uploaded += 1
                    except:
                        pass
            return html.P(f'{n_uploaded} files uploaded.')
        

    @app.callback(
    Output('ms-table', 'data'),
    Input('ms-upload-output', 'children'),
    Input('wdir', 'children'), 
    Input('ms-delete-output', 'children'),
    State('ms-table', 'derived_virtual_selected_rows'),
    State('ms-table', 'derived_virtual_indices')
    )
    def ms_table(value, wdir, files_deleted, ndxs_selected, ndxs_filtered):   
        target_dir = os.path.join(wdir, 'ms_files')
        ms_files = glob(os.path.join(target_dir, '*.*'), recursive=True)
        data =  pd.DataFrame([{'MS-file': os.path.basename(fn) } for fn in ms_files])
        return data.to_dict('records')


    @app.callback(
    Output('ms-convert-output', 'children'),
    Input('ms-convert', 'n_clicks'),
    State('wdir', 'children')
    )
    def ms_convert(n_clicks, wdir):
        target_dir = os.path.join(wdir, 'ms_files')
        if n_clicks is None:
            raise PreventUpdate
        fns = glob(os.path.join(target_dir, '*.*'))
        fns = [fn for fn in fns if not fn.endswith('.feather')]
        n_total = len(fns)
        for i, fn in enumerate( fns ):
            fsc.set('progress', int(100*(i+1)/n_total))
            new_fn = convert_ms_file_to_feather(fn)
            if os.path.isfile(new_fn): os.remove(fn)
        return 'Files converted to feather format.'


    @app.callback(
    Output('ms-delete-output', 'children'),
    Input('ms-delete', 'n_clicks'),
    [State('ms-table', 'selected_rows'),
    State('ms-table', 'data'),
    State('wdir', 'children')]
    )
    def ms_delete(n_clicks, ndxs, data, wdir):
        if n_clicks is None:
            raise PreventUpdate
        target_dir = os.path.join(wdir, 'ms_files')
        for ndx in ndxs:
            fn = data[ndx]['MS-file']
            fn = os.path.join(target_dir, fn)
            os.remove(fn)
        return 'Files deleted'