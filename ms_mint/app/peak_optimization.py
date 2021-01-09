import dash_html_components as html
import dash_core_components as dcc


info_txt = '''
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
'''


pko_layout = html.Div([
    html.H3('Peak Optimization'),
    html.Button('Generate peak previews', id='pko-peak-preview'),
    html.Button('Find closest peaks', id='pko-find-closest-peak'),
    dcc.Markdown('---'),
    html.Div(id='pko-peak-preview-output'),
    dcc.Markdown(id='pko-find-closest-peak-output'),

    html.Div(id='pko-controls'),
    dcc.Dropdown(
        id='pko-dropdown',
        options=[],
        value=None
    ),
    dcc.Loading( dcc.Graph('pko-figure') ),
    dcc.Markdown(id='pko-set-rt-output'),
    html.Button('Set RT to current view', id='pko-set-rt'),

    html.Div([
        html.Button('<< Previous', id='pko-prev'),
        html.Button('Next >>', id='pko-next')],
        style={'text-align': 'center', 'margin': 'auto'})
])

pko_layout_no_data = html.Div([
    dcc.Markdown('''### No peaklist found.
    You did not generate a peaklist yet.
    ''')
])