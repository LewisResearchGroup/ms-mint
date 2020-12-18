import dash_html_components as html
import dash_core_components as dcc


info_txt = '''
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
'''


pko_layout = html.Div([
    html.H3('Peak Optimization'),
    html.Button('Find closest peaks', id='pko-find-closest-peak'),
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
    html.Button('Fit Retention Time', id='pko-fit-rt'),
])