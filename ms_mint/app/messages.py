import logging
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, ALL

_label = None

_layout = html.Div([
    html.Div(id='message')
])

def layout():
    return _layout

def callbacks(app, fsc, cache):

    @app.callback(
        Output('message', 'children'),
        [Input({'type': 'output', 'index': ALL}, 'children')],
        prevent_initial_call=True
        )

    def message(message):
        #print(f'Message: {message}')
        ctx = dash.callback_context
        message = ctx.triggered[0]
        if (message is None) or (message['value'] is None):
            #print(message)
            raise PreventUpdate
        _type = message['value']['type']
        _chil = message['value']['props']['children']
        #print(f'Message ({_type}):', _chil)
        if 'color' in message['value']['props'].keys():
            color = message['value']['props']['color']
        else: color = None
        if _type == 'Alert':
            return dbc.Alert(_chil, color=color)
        else:
            logging.warning(f'Undefined message type: {_type}, {_chil}')
        return dbc.Alert('Undefined message', 'danger')