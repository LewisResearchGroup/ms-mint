import os
import shutil
import uuid

from glob import glob

import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State

from ms_mint.io import convert_ms_file_to_feather

from dash_tabulator import DashTabulator
from tqdm import tqdm

import dash_uploader as du



from . import tools as T


options = {
           "selectable": True,
           "headerFilterLiveFilterDelay":3000,
           "layout": "fitDataFill",
           "height": "900px",
           }

clearFilterButtonType = {"css": "btn btn-outline-dark", "text":"Clear Filters"}


# If color column is not present, for some strange reason, the header filter disappears.
columns = [
        { "formatter": "rowSelection", "titleFormatter":"rowSelection", 
          "hozAlign":"center", "headerSort": False, "width":"1px", 'frozen': True},
        { "title": "MS-file", "field": "MS-file", "headerFilter":True, 
          'headerSort': True, "editor": "input", "width": "100%",
          'sorter': 'string', 'frozen': True},
        { 'title': '', 'field': '', "headerFilter":False,  "formatter":"color", 
          'width': '3px', "headerSort": False},
    ]


ms_table = html.Div(id='ms-table-container', 
    style={'min-height':  100, 'marginTop': '10%'},
    children=[
        DashTabulator(id='ms-table',
            columns=columns, 
            options=options,
            clearFilterButtonType=clearFilterButtonType
        )
])


_layout = html.Div([
    html.H3('Upload MS-files'),
    dcc.Upload(
            id='ms-upload',
            children=html.Div([
                html.A('''Click here or drag and drop to upload mzML/mzXML files here. 
                You can upload up to 10 files at a time.''', style={'margin': 'auto', 'padding': 'auto'})
            ]),
            style={
                'width': '100%',
                'height': '120px',
                'lineHeight': '120px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-bottom': '15px',
            },
            # Allow multiple files to be uploaded
            multiple=True),
    html.Div( du.Upload(id='ms-upload-zip', filetypes=['tar', 'zip'], 
                        upload_id=uuid.uuid1(),
                        text='Click here to drag and drop ZIP/TAR compressed archives'),
              style={
                    'textAlign': 'center',
                    'width': '100%',
                    'padding': '0px',
                    'display': 'inline-block',
                }),
    html.Div(id='ms-upload-zip-filename'),    
    html.Div(id='ms-upload-zip-output'),

    dcc.Markdown('---', style={'margin-top': '10px'}),
    dcc.Markdown('##### Actions'),
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
                if n.lower().endswith('mzxml') or n.lower().endswith('mzml') or n.lower().endswith('zip'):
                    try:
                        T.parse_ms_files(c, n, d, target_dir)
                        n_uploaded += 1
                    except:
                        print(f'Could not parse file {n}')
                if n.lower().endswith('zip'):
                    print('Zip file uploaded', target_dir, n)
                    fn = os.path.join( target_dir, n)
                    shutil.unpack_archive(fn, target_dir)
                    os.remove(fn)
            return html.P(f'{n_uploaded} files uploaded.')


    @app.callback(
    Output('ms-table', 'data'),
    Input('ms-upload-output', 'children'),
    Input('wdir', 'children'), 
    Input('ms-delete-output', 'children'),
    Input('ms-upload-zip-output', 'children')
    )
    def ms_table(value, wdir, files_deleted, zip_extracted): 
        target_dir = os.path.join(wdir, 'ms_files')
        ms_files = glob(os.path.join(target_dir, '*.*'), recursive=True)
        data =  pd.DataFrame([{'MS-file': os.path.basename(fn) } for fn in ms_files])
        print(data)
        return data.to_dict('records')


    @app.callback(
    Output('ms-convert-output', 'children'),
    Input('ms-convert', 'n_clicks'),
    State('ms-table', 'multiRowsClicked'),
    State('wdir', 'children')
    )
    def ms_convert(n_clicks, rows, wdir):
        target_dir = os.path.join(wdir, 'ms_files')
        print(rows)
        if n_clicks is None:
            raise PreventUpdate
        fns = [row['MS-file'] for row in rows]
        fns = [fn for fn in fns if not fn.endswith('.feather')]
        fns = [os.path.join(target_dir, fn) for fn in fns]
        n_total = len(fns)
        for i, fn in enumerate( fns ):
            fsc.set('progress', int(100*(i+1)/n_total))
            new_fn = convert_ms_file_to_feather(fn)
            if os.path.isfile(new_fn): os.remove(fn)
        return 'Files converted to feather format.'


    @app.callback(
        Output('ms-delete-output', 'children'),
        Input('ms-delete', 'n_clicks'),
        State('ms-table', 'multiRowsClicked'),
        State('wdir', 'children')
    )
    def ms_delete(n_clicks, rows, wdir):
        if n_clicks is None:
            raise PreventUpdate
        target_dir = os.path.join(wdir, 'ms_files')
        for row in rows:
            fn = row['MS-file']
            fn = os.path.join(target_dir, fn)
            os.remove(fn)
        return f'{len(rows)} files deleted'


    @du.callback(
    output=Output('ms-upload-zip-filename', 'children'),
    id='ms-upload-zip',
    )
    def get_zip_filename(filenames):
        return filenames[0]

    @app.callback(
        Output('ms-upload-zip-output', 'children'),
        Input('ms-upload-zip-filename', 'children'),
        State('wdir', 'children')
    )
    def get_a_list(fn, wdir):
        if fn is None:
            raise PreventUpdate
        ms_dir = T.get_ms_dirname( wdir )
        upload_path = os.path.dirname( fn )
        shutil.unpack_archive(fn, extract_dir=upload_path)
        search_pattern = os.path.join( upload_path, '**', '*.*')
        fns = glob(search_pattern, recursive=True)
        n_total = len(fns)
        print(fns)
        for i, fn in tqdm( enumerate(fns), total=n_total ):
            fsc.set('progress', int( 100 * (i+1) / n_total))
            if fn.lower().endswith('mzxml') or fn.lower().endswith('mzml'):
                try:
                    shutil.move(fn, ms_dir)
                    print(f'Moved {fn} to {ms_dir}')
                except:
                    pass
        for remainings in glob(os.path.join(upload_path, '*')):
            print('Cleaning up:', remainings)
            if os.path.isfile(remainings): os.remove(remainings)
            elif os.path.isdir(remainings): shutil.rmtree(remainings)
        return 'Done'
