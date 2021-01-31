import pandas as pd

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions.javascript import Namespace
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash_tabulator import DashTabulator

from . import tools as T


ns = Namespace("myNamespace", "tabulator")

options = {
           "groupBy": "Label", 
           "selectable": True,
           "headerFilterLiveFilterDelay":3000,
           #"dataSorted" : ns("dataSorted"),
           "layout": "fitDataFill",
           "height": "900px",
           }

downloadButtonType = {"css": "btn btn-primary", "text":"Export", "type":"csv", "filename":"Metadata"}
clearFilterButtonType = {"css": "btn btn-outline-dark", "text":"Clear Filters"}

meta_table = html.Div(id='meta-table-container', 
    style={'min-height':  100, 'margin': '0%'},
    children=[
        DashTabulator(id='meta-table',
            columns=T.gen_tabulator_columns(add_ms_file_col=True, add_color_col=True, add_peakopt_col=True), 
            options=options,
            downloadButtonType=downloadButtonType,
            clearFilterButtonType=clearFilterButtonType
        )
])

options = [
    {'label': 'Batch', 'value': 'Batch'},
    {'label': 'Label', 'value': 'Label'},
    {'label': 'Color', 'value': 'Color'},
    {'label': 'Type', 'value': 'Type'},
    {'label': 'Concentration', 'value': 'Concentration'},
    ]

_layout = html.Div([
    html.H3('Metadata'),
    dcc.Upload(
            id='meta-upload',
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
            multiple=True,
        ),
    dcc.Markdown('---'),
    dcc.Markdown('##### Actions'),
    dbc.Row([
        dcc.Dropdown(
            id='meta-action', options=[
                {'label': 'Set', 'value': 'Set'},
                {'label': 'Create column', 'value': 'Create column'},
                {'label': 'Delete column', 'value': 'Delete column'}
            ],
            value='Set', style={'width': '150px'}),
        dcc.Dropdown(
            id='meta-column', options=options, 
            value=None, style={'width': '150px'}),
        dcc.Input(id='meta-input'),
        html.Button('Apply', id='meta-apply'),
    ], style={'margin-left': '5px'}),

    html.Div(id='meta-apply-output'),
    html.Div(id='meta-table-saved-on-edit-output'),
    dcc.Markdown('---'),    
    dcc.Loading( meta_table ),
])


def layout():
    return _layout


def callbacks(app, fsc, cache):
        
    @app.callback(
    Output('meta-table', 'data'),
    Output('meta-table', 'columns'),
    Output('meta-column', 'options'),
    Input('meta-upload', 'contents'),
    Input('meta-upload', 'filename'),
    Input('meta-apply-output', 'children'),
    State('wdir', 'children')
    )
    def meta_upload(contents, filename, message, wdir):
        metadata = T.get_metadata( wdir )
        if (contents is not None) and (len(contents) > 0):
            contents = T.parse_table_content(contents[0], filename[0])
            metadata = T.merge_metadata(metadata, contents)
        columns = [{'label':col, 'value':col} for col in metadata.columns if col != 'index']
        if 'index' not in metadata.columns: metadata = metadata.reset_index()
        return metadata.to_dict('records'), T.gen_tabulator_columns(metadata.columns,
            add_ms_file_col=True, add_color_col=True, add_peakopt_col=True), columns


    @app.callback(
    Output('meta-apply-output', 'children'),
    Input('meta-apply', 'n_clicks'),
    #Input('meta-table', 'cellEdited'),
    State('meta-table', 'multiRowsClicked'),
    State('meta-table', 'data'),
    State('meta-table', 'dataFiltered'),
    State('meta-action', 'value'),
    State('meta-column', 'value'),
    State('meta-input', 'value'),
    State('wdir', 'children'),
    )
    def meta_save(n_clicks, selected_rows, data, 
                  data_filtered, action, column, value, wdir):
        if data is None or len(data) == 0:
            raise PreventUpdate
        prop_id = dash.callback_context.triggered[0]['prop_id']
        fn = T.get_metadata_fn( wdir )
        df = pd.DataFrame(data)
        if 'index' in df.columns:
            df = df.set_index('index')
        else:
            df = df.reset_index()
        if prop_id == 'meta-apply.n_clicks':
            if action == 'Set':
                filtered_rows = [r for r in data_filtered['rows'] if r is not None]
                filtered_ndx = [r['index'] for r in filtered_rows]
                if selected_rows == []:
                    # If nothing is selected apply to all visible rows
                    ndxs = filtered_ndx
                else:
                    # If something is selected only apply to selected rows
                    ndxs = [r['index'] for r in selected_rows if r['index'] in filtered_ndx]
                print('ndxs:', ndxs)
                if len(ndxs) == 0 or column is None:
                    return 'No rows selected.'
                df.loc[ndxs, column] = value
            elif action == 'Create column': df[value] = ''
            elif action == 'Delete column': del df[column]
        with T.lock(fn):
            df.to_csv(fn, index=False)
        if prop_id == 'meta-table.cellEdited':
            raise PreventUpdate
        return 'Data saved.'

    
    @app.callback(
        Output('meta-table-saved-on-edit-output', 'children'),
        Input('meta-table', 'cellEdited'),
        State('meta-table', 'data'),
        State('wdir', 'children'),
    )
    def save_table_on_edit(cell_edited, data, wdir):
        if data is None or cell_edited is None:
            raise PreventUpdate
        fn = T.get_metadata_fn( wdir )
        df = pd.DataFrame(data)
        with T.lock(fn):
            df.to_csv(fn, index=False)
    