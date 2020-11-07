import platform

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import subprocess

from dash_table import DataTable

import numpy as np

from multiprocessing import cpu_count
from ms_mint import __version__
from ms_mint.standards import PEAKLIST_COLUMNS

from .button_style import button_style

n_cpus = cpu_count()

slider_style = {'marginBottom': '3em'}

info_style   =   {'margin-top': 'auto', 
                  'margin-bottom': 10, 
                  'margin-left': 'auto',
                  'display': 'inline', 
                  'float': 'right', 
                  'color': 'grey'
                  }

layout_style = {'max-width': '80%', 
                'margin': 'auto', 
                'margin-bottom': '10%'
                }

space = html.Div(style={'padding': 50})

config={'displaylogo': False}

def get_versions():
    string = ''
    try:
        string += subprocess.getoutput('conda env export --no-build')
    except:
        pass
    return string


ISSUE_TEXT = f'''
%0A%0A%0A%0A%0A%0A%0A%0A%0A
MINT version: {__version__}%0A
OS: {platform.platform()}%0A
Versions:
{get_versions()}
'''


status = html.Div([
    # Status
    html.P(f'Version: {__version__}', style={'text-align': 'right', 'color': 'grey'}),
    html.Div(id='n_peaklist_selected', children=0, style={'display': 'none'}),
    html.Div(id='n_files_selected', children=0, style={'display': 'none'}),
    html.Button('Reset', id='B_reset', style=button_style('warn')),
    html.A(href='https://soerendip.github.io/ms-mint/', 
         children=[html.Button('Help', id='B_help', style=button_style('help', float="right"))],
         target="_blank"),
    html.A(href=f'https://github.com/soerendip/ms-mint/issues/new?body={ISSUE_TEXT}', 
         children=[html.Button('Issues', id='B_issues', style=button_style('help', float="right"))],
         target="_blank"),         
])



ms_files = html.Div([
    # Buttons
    html.Button('Add MS-file(s)', id='B_add_files', style=button_style()),    
    dcc.Checklist(id='files-check', 
                  options=[{ 'label': 'Add files from directory', 'value': 'by-dir' }], 
                  value=['by-dir'], style={'display': 'inline-block'}),
    html.Div(id='files-text', children='', style=info_style),

    dcc.Loading( children=[ html.Div(id='table-ms-files-container', 
                                     style={'min-height':  100, 'margin-top': 10},
                                     children=[
                                         DataTable(id='table-ms-files',
                                                   columns=[ {"name": i, "id": i, "selectable": True}  
                                                             for i in ['MS-files']],
                                                   data=[],
                                                   row_selectable=False,
                                                   row_deletable=True,
                                                   style_cell={'textAlign': 'left'},
                                                   sort_action='native'
                                                   )
                                            ])
                        ]
                ),

    html.Div(style={'padding': 50}),
])



peaklist = html.Div(id='peaklist-div', 
  children=[
    html.H2('Peaklist'),
    html.Button('Import peaklist', id='B_peaklists', style=button_style('')),
    html.Button('Detect peaks', id='B_detect_peaks', style=button_style('')),
    html.Button('Add Peak', id='B_add_peak', style=button_style('')),
    html.Button('Clear', id='B_clear_peaklist', style=button_style('')),

    dcc.Loading( children=[ html.Div(id='table-peaklist-container', 
                                     style={'min-height':  100, 'margin-top': 10},
                                     children=[
                                         DataTable(id='table-peaklist',
                                                   columns=[ {"name": i, "id": i, "selectable": True}  
                                                             for i in PEAKLIST_COLUMNS],
                                                   data=[])
                                            ])
                        ]
                ),
    dcc.Input(id="int-threshold", type='number', placeholder="Intensity Threshold"),
    dcc.Input(id="mz-width", type='number', placeholder="m/z widths"),

    ], style={'display': 'inline'})



run_mint = html.Div(id='run-mint-div', 
  children=[
    html.H2('Run Mint'),
    html.Div(id='peaklist-text', children='', style=info_style),
    html.Div(id='cpu-text', children='', style=info_style),
    html.Br(),
    html.P("Select number of cores:", style={'display': 'inline-block', 'margin-top': 30}),
    html.Div(dcc.Slider( id='n_cpus', 
            min=1, max=n_cpus,
            step=1, value=n_cpus,
            marks={i: f'{i} cpus' for i in [1, n_cpus]}),
        style=slider_style),
    html.Button('Run', id='B_run'),
    html.A(html.Button('Export', id='B_export'), href="export"),
    # Progress bar
    dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=False),
    dbc.Progress(id="progress-bar", value=0),
    html.Div(id='progress', children=[], style=info_style),    
  ], style={'display': 'none'})



results = html.Div(
  id = 'results-div', 
  style = {'visibility': 'hidden'},
  children = [    
    html.H2("Table View", style={'margin-top': 100}),

    dcc.Dropdown(id='table-value-select', value='full',
        options=[ 
            {'label': 'Full table', 'value': 'full'},
            {'label': 'Peak area', 'value': 'peak_area'},
            {'label': 'Retention time of maximum', 'value': 'peak_rt_of_max'},
            {'label': 'N Datapoints', 'value': 'peak_n_datapoints'},
            {'label': 'Peak maximum', 'value': 'peak_max'},
            {'label': 'Peak mininimum', 'value': 'peak_min'},
            {'label': 'Peak median', 'value': 'peak_median'},
            {'label': 'Peak mean', 'value': 'peak_mean'},
            {'label': 'First minus last intensity', 'value': 'peak_delta_int'}
        ]),
    
    dcc.Loading( children=[ html.Div(id='run-out', 
                                     style={'min-height':  0, 'margin-top': 10},
                                     children=[
                                        DataTable( id='table', data=np.array([]) )
                                        ]
                                    )
                        ]
                ),

    dcc.Input(id="label-regex", type='text', placeholder="Label Selector"),
    
    html.Button('Reload', id='B_reload'),
    
    html.Div(style={'padding': 50}),
        
    html.H2("Peak Shapes"),
    
    dcc.Dropdown(id='peakshapes-selection',
        options=[],
        value=[],
        multi=True
        ), 
    
    html.Button('Peak Shapes', id='B_shapes', style=button_style()),
    
    dcc.Checklist(id='check_peakShapes', 
        options=[{'label': 'Show legend', 'value': 'legend'},
                 {'label': 'Horizontal legend', 'value': 'legend_horizontal'},
                 {'label': 'Show in new tab', 'value': 'new_tab'}],
        value=['legend'], style={'display': 'inline-block'}),
    
    html.Div(dcc.Slider(id='n_cols', min=1, max=12, step=1, value=2,
            marks={i: f'{i} columns' for i in range(1, 13)}),
        style=slider_style),
    
    dcc.Loading( children=[ dcc.Graph(id='peakShape', figure={}, config=config) ]),
    
    html.Div(style={'padding': 50}),
    
    html.H2("Heatmap"),
    
    html.Button('Heatmap', id='B_heatmap', style=button_style()),
    
    dcc.Checklist(id='checklist', 
        options=[
            { 'label': 'Normalized by biomarker', 'value': 'normed'},
            { 'label': 'Cluster', 'value': 'clustered'},
            { 'label': 'Dendrogram', 'value': 'dendrogram'},
            { 'label': 'Transposed', 'value': 'transposed'},
            { 'label': 'Correlation', 'value': 'correlation'},
            { 'label': 'Show in new tab', 'value': 'new_tab'}],

        value=['normed'], style={'display': 'inline-block'}),

    html.P(id='heatmap-message'),
    
    dcc.Loading(children=[ dcc.Graph(id='heatmap', figure={}, config=config) ]),
                
    html.Div(style={'padding': 50}),

    html.H2("Peak Shapes 3D"),
    
    html.Button('Peak Shapes 3D', id='B_shapes3d', style=button_style()),
    
    dcc.Checklist(id='check_peakShapes3d', 
        options=[{'label': 'Show legend', 'value': 'legend'},
                 {'label': 'Horizontal legend', 'value': 'legend_horizontal'},
                 {'label': 'Show in new tab', 'value': 'new_tab'}],
        value=['legend'], style={'display': 'inline-block'}),
    
    dcc.Dropdown(id='peak-select', options=[]),
    dcc.Loading([ dcc.Graph(id='peakShape3d', figure={}, config=config) ])
  ])

Layout = html.Div(
  [   
    html.H1('MINT', style={'margin-top': '10%'}),
    html.Div(id='storage', style={'display': 'none'}),
    status,   space,
    ms_files, space,
    peaklist, space,
    run_mint, space,
    results
    
], style=layout_style)
