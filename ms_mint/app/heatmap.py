import dash_html_components as html
import dash_core_components as dcc


heat_layout = html.Div([
    html.Div(id='res-upload-output'),
    html.Div(id='heatmap-controls'),
    html.Div(id='heatmap-output')
])


heatmap_options = [
    { 'label': 'Normalized by biomarker', 'value': 'normed_by_cols'},
    { 'label': 'Cluster', 'value': 'clustered'},
    { 'label': 'Dendrogram', 'value': 'add_dendrogram'},
    { 'label': 'Transposed', 'value': 'transposed'},
    { 'label': 'Correlation', 'value': 'correlation'},
    { 'label': 'Show in new tab', 'value': 'call_show'},
    { 'label': 'log1p', 'value': 'log1p'},
    ]


heat_layout_empty = html.Div([
    dcc.Upload(
            id='res-upload',
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
            multiple=False
        ),
])


heat_layout_non_empty = html.Div([
    html.H3('Heatmap'),
    html.Button('Update', id='heatmap-update'),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Dropdown(id='heatmap-options', value=['normed_by_cols', 'clustered'],
        options=heatmap_options, multi=True),
    dcc.Dropdown(id='heatmap-ms-order', options=[{'value': 'MS-file', 'label': 'MS-file'},
                                                 {'value': 'Label', 'label': 'Label'},
                                                 {'value': 'Batch', 'label': 'Batch'},
                                                 {'value': 'Type', 'label': 'Type'}
                                                 ], 
                                                 placeholder='MS-file sorting', multi=True),
    dcc.Loading( 
        dcc.Graph(id='heatmap-figure', 
                  style={'margin-top': '50px'}) ),
])