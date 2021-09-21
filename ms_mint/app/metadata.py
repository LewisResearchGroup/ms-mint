import pandas as pd

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_extensions.javascript import Namespace
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash_tabulator import DashTabulator

from . import tools as T


ns = Namespace("myNamespace", "tabulator")

tabulator_options = {
           "groupBy": "Label", 
           "selectable": True,
           "headerFilterLiveFilterDelay":3000,
           "layout": "fitDataFill",
           "height": "900px",
           }

downloadButtonType = {"css": "btn btn-primary", "text":"Export", "type":"csv", "filename":"Metadata"}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text":"Clear Filters"}

meta_table = html.Div(id='meta-table-container', 
    style={'minHeight':  100, 'margin': '0%'},
    children=[
        DashTabulator(id='meta-table',
            columns=T.gen_tabulator_columns(
                        add_ms_file_col=True, add_color_col=True, 
                        add_peakopt_col=True, add_ms_file_active_col=True
            ), 
            options=tabulator_options,
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

_label = 'Metadata'

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
                {'label': 'Create column', 'value': 'create_column'},
                {'label': 'Delete column', 'value': 'delete_column'},
                #{'label': 'Delete selected files', 'value': 'delete_ms_files'},
            ],
            value='Set', style={'width': '150px'}),
        dcc.Dropdown(
            id='meta-column', options=options, 
            value=None, style={'width': '150px'}),
        dcc.Input(id='meta-input'),
        dcc.Dropdown(
            id='meta-input-bool', options=[{'value': 'True', 'label': 'True'},
                                           {'value': 'False', 'label': 'False'}],
            value=None),

        html.Button('Apply', id='meta-apply'),
    ], style={'marginLeft': '5px'}),

    html.Div(id='meta-table-saved-on-edit-output'),
    dcc.Markdown('---'),    
    dcc.Loading( meta_table ),
])


_outputs = html.Div(id='meta-outputs', 
    children=[
        html.Div(id={'index': 'meta-apply-output', 'type': 'output'})
    ]
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
        
    @app.callback(
    Output('meta-table', 'data'),
    Output('meta-table', 'columns'),
    Output('meta-column', 'options'),
    Input('meta-upload', 'contents'),
    Input('meta-upload', 'filename'),
    Input({'index': 'meta-apply-output', 'type': 'output'}, 'children'),
    State('wdir', 'children')
    )
    def meta_upload(contents, filename, message, wdir):
        metadata = T.get_metadata( wdir )
        if (contents is not None) and (len(contents) > 0):
            contents = T.parse_table_content(contents[0], filename[0])
            metadata = T.merge_metadata(metadata, contents)
        columns = metadata.columns.to_list()
        columns = [{'label':col, 'value':col} for col in columns if col != 'index']
        if 'index' not in metadata.columns: metadata = metadata.reset_index()
        return (metadata.to_dict('records'), 
                T.gen_tabulator_columns(metadata.columns,
                    add_ms_file_col=True, 
                    add_color_col=True, 
                    add_peakopt_col=True, 
                    add_ms_file_active_col=True), 
                columns)

    @app.callback(
        Output('meta-table', 'downloadButtonType'),
        Input('tab', 'value'),
        State('active-workspace', 'children')
    )
    def update_table_export_fn(tab, ws_name):
        fn = f'{T.today()}-{ws_name}_MINT-metadata'
        downloadButtonType = {
            "css": "btn btn-primary", 
            "text":"Export", 
            "type":"csv", 
            "filename": fn }
        return downloadButtonType


    @app.callback(
    Output({'index': 'meta-apply-output', 'type': 'output'}, 'children'),
    Input('meta-apply', 'n_clicks'),
    State('meta-table', 'data'),
    State('meta-table', 'multiRowsClicked'),
    State('meta-table', 'dataFiltered'),
    State('meta-action', 'value'),
    State('meta-column', 'value'),
    State('meta-input', 'value'),
    State('meta-input-bool', 'value'),
    State('wdir', 'children'),
    )
    def meta_save(n_clicks, data, selected_rows,
                  data_filtered, action, column, value, 
                  value_bool, wdir):
        
        if action == 'Set' and column == 'PeakOpt':
            value = value_bool

        if data is None or len(data) == 0:
            raise PreventUpdate

        df = pd.DataFrame(data)

        if 'index' in df.columns:
            df = df.set_index('index')
        else:
            df = df.reset_index()

        prop_id = dash.callback_context.triggered[0]['prop_id']
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
                if len(ndxs) == 0 or column is None:
                    return dbc.Alert('No rows selected.', color='danger')
                df.loc[ndxs, column] = value
            elif action == 'create_column': df[value] = ''
            elif action == 'delete_column': del df[column]

        T.write_metadata(df, wdir )

        if prop_id == 'meta-table.cellEdited':
            raise PreventUpdate
        return dbc.Alert('Metadata saved.', color='info')

    
    @app.callback(
        Output('meta-table-saved-on-edit-output', 'children'),
        Input('meta-table', 'cellEdited'),
        State('meta-table', 'data'),
        State('wdir', 'children'),
    )
    def save_table_on_edit(cell_edited, data, wdir):
        if data is None or cell_edited is None:
            raise PreventUpdate
        df = pd.DataFrame(data)
        T.write_metadata( df , wdir)

    
    @app.callback(
        Output('meta-input', 'style'),
        Output('meta-input-bool', 'style'),
        Input('meta-action', 'value'),
        Input('meta-column', 'value')
    )
    def set_something(action, column): 
        visible = {'visibility': 'visible', 'width': '150px', 'margin': 0}
        hidden = {'visibility': 'hidden', 'width': '0px', 'margin':0}
        if (action == 'Set') & (column == 'PeakOpt'):
            return hidden, visible
        return visible, hidden
        