import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc



groupby_options = [{'label': 'Batch', 'value': 'Batch'},
                   {'label': 'Label', 'value': 'Label'},]

graph_options = [{'label': 'Histograms', 'value': 'hist'},
                 {'label': 'Boxplots', 'value': 'boxplot'}]


qc_layout = html.Div([
    html.H3('Quality Control'),
    html.Button('Update', id='qc-update'),
    dcc.Dropdown(id='qc-groupby', options=groupby_options, value=None, placeholder='Group by column'),
    dcc.Dropdown(id='qc-graphs', options=graph_options, value=None, multi=True, placeholder='Kinds of graphs'),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Checklist(id='qc-select', options=[{'label': 'Dense', 'value': 'Dense'}]),
    html.Div(id='qc-figures', style={'float': 'center'})
])

