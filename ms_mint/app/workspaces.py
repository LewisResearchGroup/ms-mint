import re
import shutil

import pandas as pd

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate

from dash_table import DataTable
from dash.dependencies import Input, Output, State

from . import tools as T


ws_table = html.Div(id='ws-table-container', 
    style={'min-height':  100, 'margin': '5%'},
    children=[
        DataTable(id='ws-table',
            columns=[ {"name": i, "id": i, "selectable": True}  
                                for i in ['Workspace']],
                    data=[],
                    row_selectable='single',
                    row_deletable=False,
                    style_cell={'textAlign': 'left'},
                    sort_action='native'
                    )
])


_layout = html.Div([
    html.H3('Workspaces'),
    html.Button('Create Workspace', id='ws-create'),
    html.Button('Delete Workspace', id='ws-delete', style={'float': 'right'}),
    html.P(id='ws-created-output'),

    dcc.Markdown(id='ws-activate-output'),
    html.P(id='ws-delete-output'),
    ws_table,

    dbc.Modal(
        [dbc.ModalHeader("Create Workspace"),
         dbc.ModalBody(
             [dcc.Input(id='ws-create-input', placeholder='New workspace name'),
              html.P(id='ws-create-output')
             ]
         ),
         dbc.ModalFooter(
            html.Div([
                dbc.Button("Create", id="ws-create-confirmed", style={'margin-right': '10px'}),
                dbc.Button("Cancel", id="ws-create-cancel")
            ])
        )], id="ws-create-popup"
    ),

    dbc.Modal(
            [
                dbc.ModalHeader("Delete Workspace"),
                dbc.ModalBody("This will delete all files and results in the selected workspace."),
                dbc.ModalFooter(
                    html.Div([
                        dbc.Button("Delete", id="ws-delete-confirmed", style={'margin-right': '10px'}),
                        dbc.Button("Cancel", id="ws-delete-cancel")
                    ])
                ),
            ],
            id="ws-delete-popup",
        ),
])

def layout():
    return _layout

def callbacks(app, fsc, cache):

    @app.callback(
    Output('ws-table', 'data'),
    Input('ws-created-output', 'children'),
    Input('tab', 'value'),
    Input('ws-delete-output', 'children'),
    State('tmpdir', 'children')
    )
    def ws_table(value, tab, delete, tmpdir):
        T.maybe_migrate_workspaces(tmpdir)
        ws_names = T.get_workspaces(tmpdir)
        ws_names = [{'Workspace': ws_name} for ws_name in ws_names 
                        if not ws_name.startswith('.')]
        return ws_names


    @app.callback(
    Output('ws-activate-output', 'children'),
    Output('wdir', 'children'),
    Output('active-workspace', 'children'),
    Input('ws-table', 'derived_virtual_selected_rows'),
    Input('ws-delete-output', 'children'),
    State('ws-table', 'data'),
    State('tmpdir', 'children')
    )
    def ws_activate(ndx, deleted, data, tmpdir):
        prop_id = dash.callback_context.triggered[0]['prop_id']
        ws_name = T.get_actived_workspace(tmpdir)
        if prop_id == 'ws-delete-output.children':
            ndx = [0]
        data = pd.DataFrame(data)
        if len(ndx) > 0:
            ndx = ndx[0]
            ws_name = data.iloc[ndx]['Workspace']
        if not T.workspace_exists(tmpdir, ws_name):
            ws_name = None
        wdir = T.workspace_path(tmpdir, ws_name)
        if ws_name is not None: T.save_activated_workspace(tmpdir, ws_name)
        else: raise PreventUpdate
        message = f'Workspace __{ws_name}__ activated.'
        return message, wdir, ws_name


    @app.callback(
    Output('progress-bar', 'value'),
    Input('progress-interval', 'n_intervals'),
    )
    def set_progress(n):
        return fsc.get('progress')


    @app.callback(
    Output("ws-delete-popup", "is_open"),
    Input("ws-delete", "n_clicks"), 
    Input("ws-delete-cancel", "n_clicks"),
    Input('ws-delete-confirmed', 'n_clicks'),
    State("ws-delete-popup", "is_open")
    )
    def ws_delete(n1, n2, n3, is_open):
        if n1 is None:
            raise PreventUpdate
        if n1 or n2 or n3:
            return not is_open
        return is_open



    @app.callback(
        Output('ws-delete-output', 'children'),
        Input('ws-delete-confirmed', 'n_clicks'),
        State('ws-table', 'derived_virtual_selected_rows'),
        State('ws-table', 'data'),
        State('tmpdir', 'children')
    )
    def ws_delete_confirmed(n_clicks, ndxs, data, tmpdir):
        if n_clicks is None or len(ndxs) == 0:
            raise PreventUpdate
        for ndx in ndxs:
            ws_name = data[ndx]['Workspace']
            dirname = T.workspace_path( tmpdir, ws_name )
            shutil.rmtree(dirname)
        message = f'Worskpace {ws_name} deleted.'
        return message



    @app.callback(
    Output("ws-create-popup", "is_open"),
    Input("ws-create", "n_clicks"), 
    Input("ws-create-cancel", "n_clicks"),
    Input('ws-create-confirmed', 'n_clicks'),
    State("ws-create-popup", "is_open")
    )
    def ws_create(n1, n2, n3, is_open):
        if n1 is None:
            raise PreventUpdate
        if n1 or n2 or n3:
            return not is_open
        return is_open    


    @app.callback(
        Output('ws-create-output', 'children'),
        Input('ws-create-input', 'value'),
        State('tmpdir', 'children')
    )
    def ws_create_message(ws_name, tmpdir):
        if (ws_name is None) or (ws_name == ''):
            raise PreventUpdate
        if not re.match('^[\w_-]+$', ws_name):
            return 'Name can only contain: a-z, A-Z, 0-9, -,  _ and no blanks.'        
        if T.workspace_exists(tmpdir, ws_name):
            return 'Workspace already exists'
        else:
            return f'Can create workspace "{ws_name}"'


    @app.callback(
        Output('ws-created-output', 'children'),
        Input('ws-create-confirmed', 'n_clicks'),
        State('ws-create-input', 'value'),
        State('tmpdir', 'children')
    )
    def ws_create_confirmed(n_clicks, ws_name, tmpdir):
        if n_clicks is None: raise PreventUpdate
        T.create_workspace(tmpdir, ws_name)
        return None

