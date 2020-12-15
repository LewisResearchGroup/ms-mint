import dash_html_components as html
import dash_core_components as dcc
from dash_table import DataTable
from ms_mint.standards import PEAKLIST_COLUMNS

columns = [{"name": i, "id": i, 
            "selectable": True}  for i in PEAKLIST_COLUMNS] 


pkl_table = DataTable(
                id='pkl-table',
                columns=columns,
                data=None,
                sort_action="native",
                sort_mode="single",
                row_selectable=False,
                row_deletable=True,
                editable=True,
                column_selectable=False,
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 30,
                filter_action='native',                
                style_table={'overflowX': 'scroll'},
                style_as_list_view=True,
                style_cell={'padding-left': '5px', 
                            'padding-right': '5px'},
                style_header={'backgroundColor': 'white',
                              'fontWeight': 'bold'},
                export_format='csv',
                export_headers='display',
                merge_duplicate_headers=True
            ) 


pkl_layout = html.Div([
    html.H3('Peaklist'),
    dcc.Upload(
            id='pkl-upload',
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
    html.Div(id='pkl-upload-output'),
    html.Div(id='pkl-save-output'),

    pkl_table, 
    html.Button('Save peaklist', id='pkl-save'),
    html.Button('Delete selected peaks', id='pkl-delete', style={'float': 'right'}),
])