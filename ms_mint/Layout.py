import platform

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_table import DataTable

import numpy as np

from multiprocessing import cpu_count
from ms_mint import __version__

button_color = 'lightgreen'
button_color_warn = '#DC7633'
button_style = {'margin-right': 20, 'margin-bottom': '1.5em','background-color': button_color}
button_style_warn = button_style.copy()
button_style_warn['background-color'] = button_color_warn


slider_style = {'marginBottom': '3em'}
info_style = {'margin-top': 10, 'margin-bottom': 10, 'margin-left': 10,
              'display': 'inline-block', 'float': 'right', 'color': 'grey'}

help_button_style = button_style.copy()
help_button_style.update({'float': 'right', 
                          'background-color': 'white'})

n_cpus = cpu_count()


ISSUE_TEXT = f'''
%0A%0A%0A%0A%0A%0A%0A%0A%0A
MINT: {__version__}%0A
OS: {platform.platform()}%0A
'''

Layout = html.Div(
    [   
     
        html.H1("MINT", style={'margin-top': '10%'}),
        
        html.Div(id='storage', style={'display': 'none'}),
    
        html.Button('Select peaklist file(s)', id='B_select-peaklists', style=button_style),
    
        html.Button('Add MS-file(s)', id='B_add-files', style=button_style),
            
        html.Button('Clear files', id='B_files-clear', style=button_style_warn),
        
        html.A(href=f'https://github.com/soerendip/ms-mint/issues/new?body={ISSUE_TEXT}', 
               children=[html.Button('Help / Issues', id='B_help', style=help_button_style)],
               target="_blank"),

        html.Br(),
        
        dcc.Checklist(id='files-check', 
                    options=[{ 'label': 'Add files from directory', 'value': 'by-dir'}], 
                    value=['by-dir'], style={'display': 'inline-block'}),
        html.Br(),
        
        html.Div(id='files-text', children='', style=info_style),
        
        html.Div(id='peaklist-text', children='', style=info_style),
        
        html.Div(id='cpu-text', children='', style=info_style),
        
        html.Br(),

        html.P("Select number of cores:", style={'display': 'inline-block', 'margin-top': 30}),
        
        html.Div(dcc.Slider( id='n_cpus', 
                            min=1, max=n_cpus,
                            step=1, value=n_cpus,
                            marks={i: f'{i} cpus' for i in [1, n_cpus]}),
                style=slider_style),
        
        html.Button('Run', id='run', style=button_style),
        
        html.A(html.Button('Export', id='export', 
                        style={'background-color': button_color}),
        href="export"),
        
        dcc.Interval(id="progress-interval", n_intervals=0, interval=1000),
        
        dbc.Progress(id="progress-bar", value=100),
        
        html.Div(id='progress', children=[], style=info_style),
        
        html.Br(),
        
   
        
        html.H2("Table View", style={'margin-top': 100}),
        
        dcc.Dropdown(id='table-value-select', value='full',
                     options=[ 
                              {'label': 'Full Table', 'value': 'full'},
                              {'label': 'Peak Area', 'value': 'peakArea'},
                              {'label': 'Retention time of peak maximum', 'value': 'rt_max_intensity'}
                     ]),
                      
                            # 'peakArea', 'rt_max_intensity',
                            # 'intensity_median', 'intensity_max', 'intensity_min'
        
        html.Div(id='run-out', 
                style={'min-height':  0, 'margin-top': 10},
                children=[DataTable(id='table', data=np.array([]))]),
        
        dcc.Input(id="label-regex", type='text', placeholder="Label Selector"),
        
        
        html.H2("Heatmap"),
        
        html.Button('Heatmap', id='B_peakAreas', style=button_style),
        
        dcc.Checklist(id='checklist', 
                      options=[{ 'label': 'Normalized by peak', 'value': 'normed'},
                               { 'label': 'Cluster', 'value': 'clustered'},
                               { 'label': 'Dendrogram', 'value': 'dendrogram'},
                               { 'label': 'Transposed', 'value': 'transposed'},
                               { 'label': 'Correlation', 'value': 'corr'} ], 
                      value=['normed'], style={'display': 'inline-block'}),
        
        dcc.Graph(id='heatmap', figure={}),
        
        
        html.H2("Peak Shapes"),
        
        html.Button('Peak Shapes', id='B_peakShapes', style=button_style),
        
        dcc.Checklist(id='check_peakShapes', 
                      options=[{'label': 'Show Legend', 'value': 'legend'},
                               {'label': 'Horizontal legend', 'value': 'legend_horizontal'}],
                      value=['legend'], style={'display': 'inline-block'}),
        
        html.Div(dcc.Slider(id='n_cols', min=1, max=5, step=1, value=2,
                            marks={i: f'{i} columns' for i in range(1, 6)}),
                 style=slider_style),
        
        dcc.Graph(id='peakShape', figure={}),
        
        
        html.H2("Peak Shapes 3D"),
        
        html.Button('Peak Shapes 3D', id='B_peakShapes3d', style=button_style),
        
        dcc.Checklist(id='check_peakShapes3d', 
                    options=[{'label': 'Show Legend', 'value': 'legend'},
                            {'label': 'Horizontal legend', 'value': 'legend_horizontal'}], 
                    value=['legend'], style={'display': 'inline-block'}),
        
        dcc.Dropdown(id='peak-select', options=[]),
        
        dcc.Graph(id='peakShape3d', figure={}, style={'height': 800}),

    ], style={'max-width': '80%', 'margin': 'auto', 'margin-bottom': '10%'}

)
