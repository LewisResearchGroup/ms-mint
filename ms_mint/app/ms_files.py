import dash_html_components as html
import dash_core_components as dcc

from dash_table import DataTable


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

ms_layout = html.Div([
    html.H3('MS-file'),
    dcc.Markdown('''At the moment the upload is limited to ~10 files at a time.
    '''),
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
    html.Button('Convert to Feather', id='ms-convert'),
    html.Button('Delete selected files', id='ms-delete', style={'float': 'right'}),
    dcc.Loading( html.Div(id='ms-upload-output') ),
    html.Div(id='ms-convert-output'),
    html.Div(id='ms-delete-output'),
    html.Div(id='ms-save-output'),
    dcc.Loading( ms_table ), 
    #dcc.Input(id='ms-input'),
    #html.Button('Set Labels', id='ms-set-labels'),
    #html.Button('Set Batch', id='ms-set-batch'),
])