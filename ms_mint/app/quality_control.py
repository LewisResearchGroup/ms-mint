import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc



groupby_options = [{'label': 'Batch', 'value': 'Batch'},
                   {'label': 'Label', 'value': 'Label'},
                   {'label': 'Type',  'value': 'Type'}]

graph_options = [{'label': 'Histograms', 'value': 'hist'},
                 {'label': 'Boxplots', 'value': 'boxplot'},
                 {'label': 'Propability Density', 'value': 'density'}]


qc_layout = html.Div([
    html.H3('Quality Control'),
    html.Button('Update', id='qc-update'),
    dcc.Dropdown(id='qc-groupby', options=groupby_options, value=None, placeholder='Group by column'),
    dcc.Dropdown(id='qc-graphs', options=graph_options, value=None, multi=True, placeholder='Kinds of graphs'),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Dropdown(id='peak-labels', options=[], placeholder='Limit to peak_labels', multi=True),
    dcc.Checklist(id='qc-select', options=[{'label': 'Dense', 'value': 'Dense'}], value=['Dense']),
    html.Div(id='qc-figures', style={'float': 'center'})
])

qc_layout_no_data = html.Div([
    dcc.Markdown('''### No results generated yet. 
    MINT has not been run yet. The Quality Control tabs uses the processed data. 
    To generate it please add MS-files as well as a valid peaklist. 
    Then execute MINT data processing routine witby clicking the `RUN MINT` button. 
    Once results have been produced you can access the QC tools.'''),
])