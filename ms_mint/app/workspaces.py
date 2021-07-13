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
    style={'minHeight':  100, 'margin': '5%'},
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
                dbc.Button("Create", id="ws-create-confirm", style={'marginRight': '10px'}),
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
                        dbc.Button("Delete", id="ws-delete-confirm", style={'marginRight': '10px'}),
                        dbc.Button("Cancel", id="ws-delete-cancel")
                    ])
                ),
            ],
            id="ws-delete-popup",
        ),
])


_outputs = html.Div(id='ws-outputs', 
    children=[
        html.Div(id={'index': 'ws-created-output',  'type': 'output'}),
        html.Div(id={'index': 'ws-activate-output', 'type': 'output'}),
        html.Div(id={'index': 'ws-delete-output',   'type': 'output'})
    ]
)

_label = 'Workspaces'

def layout():
    return _layout

def callbacks(app, fsc, cache):

    @app.callback(
    Output('ws-table', 'data'),
    Input({'index': 'ws-created-output', 'type': 'output'}, 'children'),
    Input('tab', 'value'),
    Input({'index': 'ws-delete-output', 'type': 'output'}, 'children'),
    State('tmpdir', 'children')
    )
    def ws_table(value, tab, delete, tmpdir):
        T.maybe_migrate_workspaces(tmpdir)
        ws_names = T.get_workspaces(tmpdir)
        ws_names.sort()
        ws_names = [{'Workspace': ws_name} for ws_name in ws_names 
                        if not ws_name.startswith('.')]
        return ws_names




    @app.callback(
    Output({'index': 'ws-activate-output', 'type': 'output'}, 'children'),
    Output('wdir', 'children'),
    Output('active-workspace', 'children'),
    Input('ws-table', 'derived_virtual_selected_rows'),
    Input({'index': 'ws-delete-output', 'type': 'output'}, 'children'),
    Input({'index': 'ws-created-output', 'type': 'output'}, 'children'),
    State('ws-table', 'data'),
    State('tmpdir', 'children')
    )
    def ws_activate(ndx, deleted, created, data, tmpdir):
        prop_id = dash.callback_context.triggered[0]['prop_id']        
        if tmpdir is None: raise PreventUpdate
        ws_names = T.get_workspaces( tmpdir )
        if ws_names is None or len(ws_names)==0 : 
            message = 'No workspace defined.'
            return dbc.Alert(message, color='danger'), '', ''
        if prop_id == 'ws-delete-output.children' or ndx is None:
            ndx = [0]

        data = pd.DataFrame(data)

        if len(ndx) == 1:
            if ndx[0] not in data.index:
                raise PreventUpdate
            ws_name = data.loc[ndx[0], 'Workspace']
        else:
            ws_name = T.get_active_workspace( tmpdir )
        
        if ws_name is None: raise PreventUpdate
        wdir = T.workspace_path(tmpdir, ws_name)

        if ws_name is not None: 
            T.save_activated_workspace(tmpdir, ws_name)
        else: 
            raise PreventUpdate
        message = f'Workspace {ws_name} activated.'
        if ws_name is None: ws_name = ''
        return dbc.Alert(message, color='info'), wdir, ws_name


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
    Input('ws-delete-confirm', 'n_clicks'),
    State("ws-delete-popup", "is_open")
    )
    def ws_delete(n1, n2, n3, is_open):
        if n1 is None:
            raise PreventUpdate
        if n1 or n2 or n3:
            return not is_open
        return is_open



    @app.callback(
        Output({'index': 'ws-delete-output', 'type': 'output'}, 'children'),
        Input('ws-delete-confirm', 'n_clicks'),
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
        return dbc.Alert(message, color='info')



    @app.callback(
    Output("ws-create-popup", "is_open"),
    Input("ws-create", "n_clicks"), 
    Input("ws-create-cancel", "n_clicks"),
    Input('ws-create-confirm', 'n_clicks'),
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
        Output('ws-create-confirm', 'disabled'),
        Input('ws-create-input', 'value'),
        State('tmpdir', 'children')
    )
    def ws_create_message(ws_name, tmpdir):
        if (ws_name is None):
            raise PreventUpdate
        elif ws_name == '':
            return dbc.Alert('Name cannot be empty', color='warning'), True
        elif not re.match('^[\w_-]+$', ws_name):
            return dbc.Alert('Name can only contain: a-z, A-Z, 0-9, -,  _ and no blanks.', color='warning'), True
        elif T.workspace_exists(tmpdir, ws_name):
            return dbc.Alert('Workspace already exists', color='warning'), True
        else:
            return dbc.Alert(f'Can create workspace "{ws_name}"', color='success'), False


    @app.callback(
        Output({'index': 'ws-created-output', 'type': 'output'}, 'children'),
        Input('ws-create-confirm', 'n_clicks'),
        State('ws-create-input', 'value'),
        State('tmpdir', 'children')
    )
    def ws_create_confirmed(n_clicks, ws_name, tmpdir):
        if n_clicks is None: raise PreventUpdate
        T.create_workspace(tmpdir, ws_name)
        return dbc.Alert(f'Workspace {ws_name} created.', color='success')


    @app.callback(
        Output('ws-table', 'selected_rows'),
        Input('tab', 'value'),
        Input('ws-table', 'data'),
        State('tmpdir', 'children')
    )
    def set_selected_row(tab, data, tmpdir):
        if tab != _label: raise PreventUpdate
        ws_names = T.get_workspaces(tmpdir)
        if len(ws_names) == 0: raise PreventUpdate
        data = pd.DataFrame(data)
        active_ws = T.get_active_workspace( tmpdir )
        if active_ws is None:
            ndx = data.index[0]
            active_ws = data.Workspace[0]
        else:
            ndx = ws_names.index(active_ws)
        if ndx is None:
            ndx = 0
        return [ndx]