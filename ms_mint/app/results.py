import dash_html_components as html
import dash_core_components as dcc


res_layout = html.Div([
    dcc.Markdown(id='res-delete-output'),
    html.Div(id='res-upload-output'),
    html.Div(id='res-controls'),
    html.Div(id='res-output')
])


heatmap_options = [
    { 'label': 'Normalized by biomarker', 'value': 'normed_by_cols'},
    { 'label': 'Cluster', 'value': 'clustered'},
    { 'label': 'Dendrogram', 'value': 'add_dendrogram'},
    { 'label': 'Transposed', 'value': 'transposed'},
    { 'label': 'Correlation', 'value': 'correlation'},
    { 'label': 'Show in new tab', 'value': 'call_show'}]


res_layout_empty = html.Div([
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

res_layout_non_empty = html.Div([
    html.Button('Delete results', id='ms-delete', style={'float': 'right'}),
    html.H3('Heatmap'),
    html.Button('Heatmap', id='res-heatmap'),
    dcc.Dropdown(id='res-heatmap-options', value=[],
        options=heatmap_options, multi=True),
    dcc.Loading( 
        dcc.Graph(id='res-heatmap-figure', 
                  style={'margin-top': '50px'}) ),
])