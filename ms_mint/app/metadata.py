import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions.javascript import Namespace

from dash_tabulator import DashTabulator
from .tools import gen_tabulator_columns


ns = Namespace("myNamespace", "tabulator")

options = {
    #"groupBy": "Label", 
           "selectable": True,
           "headerFilterLiveFilterDelay":3000,
#           "dataSorted" : ns("dataSorted"),
           "layout": "fitDataFill",
           "height": "900px",
           }

downloadButtonType = {"css": "btn btn-primary", "text":"Export", "type":"csv"}
clearFilterButtonType = {"css": "btn btn-outline-dark", "text":"Clear Filters"}

meta_table = html.Div(id='meta-table-container', 
    style={'min-height':  100, 'margin': '0%'},
    children=[
        DashTabulator(id='meta-table',
            columns=gen_tabulator_columns(), 
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

meta_layout = html.Div([
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
    dcc.Markdown('---'),    
    dcc.Loading( meta_table ),
])