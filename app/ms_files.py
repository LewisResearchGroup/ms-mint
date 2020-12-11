import dash_html_components as html
import dash_core_components as dcc

from dash_table import DataTable


ms_table = html.Div(id='ms-table-container', 
    style={'min-height':  100, 'margin': '5%'},
    children=[
        DataTable(id='ms-table',
            columns=[ {"name": i, "id": i, "selectable": True}  
                                for i in ['MS-files']],
                    data=[],
                    row_selectable='multi',
                    row_deletable=False,
                    style_cell={'textAlign': 'left'},
                    sort_action='native'
                    )
])

ms_layout = html.Div([
    html.H3('MS-files'),
    dcc.Upload(
            id='ms-upload',
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
    html.Div(id='ms-upload-output'),
    ms_table, 
    html.Button('Delete selected files', id='ms_delete'),

])