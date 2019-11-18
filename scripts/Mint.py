import io
import re
import sys

import colorlover as cl
import time
import numpy as np
import pandas as pd
import uuid

from datetime import date, datetime
from flask import send_file
from functools import lru_cache
from glob import glob
from multiprocessing import cpu_count
from plotly.subplots import make_subplots
from tkinter import Tk, filedialog

from os.path import basename, isfile, abspath, join

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

from ms_mint.backend import Mint, STANDARD_PEAKFILE

mint = Mint()
mint.peaklist = STANDARD_PEAKFILE

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://codepen.io/chriddyp/pen/bWLwgP.css"]
)
app.title = 'MINT'

button_color = 'lightgreen'
button_color_warn = '#DC7633'
button_style = {'margin-right': 20, 'margin-bottom': '1.5em','background-color': button_color}
button_style_warn = button_style.copy()
button_style_warn['background-color'] = button_color_warn
slider_style = {'marginBottom': '3em'}
info_style = {'margin-top': 10, 'margin-bottom': 5, 'margin-left': 10,
              'display': 'inline-block', 'float': 'right', 'color': 'grey'}

n_cpus = cpu_count()

app.layout = html.Div(
    [   
        html.H1("MINT", style={'margin-top': '10%'}),
        html.Div(id='storage', style={'display': 'none'}),
        html.Button('Add file(s)', id='files', style=button_style),
    
        html.Button('Select peaklist file(s)', id='peaklist', style=button_style),
        html.Button('Clear files', id='files-clear', style=button_style_warn),
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
        dcc.Interval(id="progress-interval", n_intervals=0, interval=1000),
        dbc.Progress(id="progress-bar", value=50),
        html.Div(id='progress', children=[], style=info_style),
        
        html.H2("Table View", style={'margin-top': 100}),
        dcc.Dropdown(id='table-value-select', options=[ {'label': i, 'value': i} for i in ['peakArea', 'rt_max_intensity',
                            'intensity_median', 'intensity_max', 'intensity_min'] ], value='peakArea'),
        html.Div(id='run-out', 
                style={'min-height':  0, 'margin-top': 10},
                children=[DataTable(id='table', data=np.array([]))]),
        dcc.Input(
            id="label-regex",
            type='text',
            placeholder="Label Selector",
        ),
        html.A(html.Button('Export', id='export', 
                           style={'float': 'right', 'background-color': button_color}), href="export") ,

        html.H2("Heatmap"),
        html.Button('Heatmap', id='b_peakAreas', style=button_style),
        dcc.Checklist(id='checklist', 
                    options=[{ 'label': 'Normalized', 'value': 'normed'},
                            { 'label': 'Cluster', 'value': 'clustered'},
                            { 'label': 'Dendrogram', 'value': 'dendrogram'}], 
                    value=['normed'], style={'display': 'inline-block'}),
        dcc.Graph(id='peakAreas', figure={}),
        
        
        html.H2("Peak Shapes"),
        html.Button('Peak Shapes', id='b_peakShapes', style=button_style),
        dcc.Checklist(id='check_peakShapes', 
                    options=[{'label': 'Show Legend', 'value': 'legend'},
                            {'label': 'Horizontal legend', 'value': 'legend_horizontal'}], 
                    value=['legend'], style={'display': 'inline-block'}),
        
        html.Div(dcc.Slider(id='n_cols', min=1, max=5, step=1, value=2,
                            marks={i: f'{i} columns' for i in range(1, 6)}),
                style=slider_style),
        dcc.Graph(id='peakShape', figure={}),
        
        html.H2("Peak Shapes 3D"),
        html.Button('Peak Shapes 3D', id='b_peakShapes3d', style=button_style),
        dcc.Checklist(id='check_peakShapes3d', 
                    options=[{'label': 'Show Legend', 'value': 'legend'},
                            {'label': 'Horizontal legend', 'value': 'legend_horizontal'}], 
                    value=['legend'], style={'display': 'inline-block'}),
        dcc.Dropdown(id='peak-select', options=[]),
        dcc.Graph(id='peakShape3d', figure={}, style={'height': 800}),

    ], style={'max-width': '80%', 'margin': 'auto', 'margin-bottom': '10%'}
)

@app.callback(
    [Output("progress-bar", "value"), 
     Output("progress", 'children')],
    [Input("progress-interval", "n_intervals")])
def update_progress(n):
    progress = mint.progress
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, (f"Progress: {progress} %" if progress >= 5 else "")

### Load MS-files
@app.callback(
    Output('files', 'value'),
    [Input('files', 'n_clicks')],
    [State('files-check', 'value')] )
def select_files(n_clicks, options):
    if n_clicks is not None:
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        if 'by-dir' not in options:
            files = filedialog.askopenfilename(multiple=True)
            files = [abspath(i) for i in files]
            for i in files:
                assert isfile(i)
        else:
            dir_ = filedialog.askdirectory()
            if isinstance(dir_ , tuple):
                dir_ = []
            if len(dir_) != 0:
                files = glob(join(dir_, join('**', '*.mzXML')), recursive=True)
            else:
                files = []
        if len(files) != 0:
            mint.files += files
        root.destroy()
    return str(n_clicks)

### Clear files
@app.callback(
    Output('files-clear', 'value'),
    [Input('files-clear', 'n_clicks')])
def clear_files(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    mint.files = []
    return str(n_clicks)
    
### Update n-files text when files added or cleared
@app.callback(
    Output('files-text', 'children'),
    [Input('files', 'value'),
    Input('files-clear', 'value')])    
def update_files_text(n,k):
        return '{} data files selected.'.format(mint.n_files)



### Load peaklist files
@app.callback(
    Output('peaklist-text', 'children'),
    [Input('peaklist', 'n_clicks')] )
def select_peaklist(n_clicks):
    if n_clicks is not None:
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        files = filedialog.askopenfilename(multiple=True)
        files = [i  for i in files if i.endswith('.csv')]
        if len(files) != 0:
            mint.peaklist = files
        root.destroy()
    return '{} peaklist-files selected.'.format(mint.n_peaklist_files)

@app.callback(
    Output('cpu-text', 'children'),
    [Input('n_cpus', 'value')] )
def run_mint(value):
    return f'Using {value} cores.'

@app.callback(
    Output('storage', 'children'),
    [Input('run', 'n_clicks')],
    [State('n_cpus', 'value')])
def run_mint(n_clicks, n_cpus):
    if n_clicks is not None:
        mint.run(nthreads=n_cpus)
        #mint.results.to_csv('/tmp/mint_results.csv')
    return mint.results.to_json(orient='split')


### Data Table
@app.callback(
    [Output('run-out', 'children'),
     Output('peak-select', 'options')],
    [Input('storage', 'children'),
     Input('label-regex', 'value'),
     Input('table-value-select', 'value')])
def get_table(json, label_regex, col_value):
    df = pd.read_json(json, orient='split').round(0)
    
    if len(df) == 0:
        raise PreventUpdate
    
    df = pd.crosstab(df.peakLabel, 
                    df.mzxmlFile, 
                    df[col_value], 
                    aggfunc=sum).astype(np.float64).T
    
    biomarker_names = df.columns
    df.index.name = 'FileName'
    df.reset_index(inplace=True)
    df['FileName'] = df['FileName'].apply(basename).apply(lambda x: x.split('.')[0])
    if label_regex is not None:
        try:
            labels = [ i.split('_')[int(label_regex)] for i in df.FileName ]
            df['FileName'] = labels
        except:
            pass
    df.columns = df.columns.astype(str)
    
    table = DataTable(
                id='table',
                columns=[{"name": i, "id": i, "selectable": True} for i in df.columns],
                data=df.to_dict('records'),
                sort_action="native",
                sort_mode="single",
                row_selectable="multi",
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 30,
                style_table={'overflowX': 'scroll'},
                style_as_list_view=True,
                style_cell={'padding': '5px'},
                style_header={'backgroundColor': 'white',
                            'fontWeight': 'bold'},
            )
    return table, [ {'label': i, 'value': i} for i in biomarker_names]

@app.callback(
    Output('peakAreas', 'figure'),
    [Input('b_peakAreas', 'n_clicks'),
     Input('checklist', 'value')],
    [State('table', 'derived_virtual_indices'),
     State('table', 'data')])
def plot_0(n_clicks, options,ndxs, data):
    if n_clicks is None:
        raise PreventUpdate
    title = 'Heatmap'
    df = pd.DataFrame(data).set_index('FileName')
    if n_clicks is None:
        return {}
        
    if 'normed' in options:
        df = (df / df.max()).fillna(0)
        title = f'Normalized {title}'
                    
    if 'clustered' in options:
        D = squareform(pdist(df, metric='euclidean'))
        Y = linkage(D, method='complete')
        Z = dendrogram(Y, orientation='left', no_plot=True)['leaves']
        df = df.iloc[Z,:]
        dendro_side = ff.create_dendrogram(df, orientation='right')

    heatmap = go.Heatmap(z=df.values,
                        x=df.columns,
                        y=df.index,
                        colorscale = 'Blues')
    
    if (not 'dendrogram' in options) or (not 'clustered' in options):
        fig = go.Figure(heatmap)
        fig.update_layout(
            title={'text': title,  },
            yaxis={'title': '', 
                'tickmode': 'array', 
                'automargin': True}) 
        fig.update_layout({'height':800, 
                        'hovermode': 'closest'})
        return fig
    
    fig = go.Figure()
    
    for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

    for data in dendro_side['data']:
        fig.add_trace(data)
        
    y_labels = heatmap['y']
    
    heatmap['y'] = dendro_side['layout']['yaxis']['tickvals']
    dendro_side['layout']['yaxis']['ticktext'] = y_labels
    
    fig.add_trace(heatmap)     
            
    fig.update_layout({'height':800, 'showlegend':False, 'hovermode': 'closest'})
    fig.update_layout(
            title={'text': title,  },
            yaxis={'title': '', 
                'tickmode': 'array', 
                'automargin': True}) 


    fig.update_layout(xaxis2={'domain': [0, .1],
                            'mirror': False,
                            'showgrid': True,
                            'showline': False,
                            'zeroline': False,
                            'showticklabels': False,
                            'ticks':""})        
    
    fig.update_layout(xaxis={'domain': [.11, 1],
                            'mirror': False,
                            'showgrid': False,
                            'showline': False,
                            'zeroline': False,
                            'showticklabels': True,
                            'ticks':""})
    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, 1],
                            'mirror': False,
                            'showgrid': False,
                            'showline': False,
                            'zeroline': False,
                            'showticklabels': False,
                            })
    
    fig.update_layout(yaxis_ticktext=y_labels)
    fig.update_layout({'paper_bgcolor': 'white',
                    'plot_bgcolor': 'white'})
    return fig

@lru_cache(maxsize=32)
@app.callback(
    Output('peakShape', 'figure'),
    [Input('b_peakShapes', 'n_clicks')],
    [State('n_cols', 'value'),
     State('check_peakShapes', 'value')])
def plot_1(n_clicks, n_cols, options):
    if (n_clicks is None) or (mint.rt_projections is None):
        raise PreventUpdate
    files = mint.crosstab.columns
    labels = mint.crosstab.index
    fig = make_subplots(rows=len(labels)//n_cols+1, 
                        cols=n_cols, 
                            subplot_titles=labels)
    
    if len(files) < 13:
        # largest color set in colorlover is 12
        colors = cl.scales['12']['qual']['Paired']
    else:
        colors = cl.interp( cl.scales['12']['qual']['Paired'], len(files))
            
    for label_i, label in enumerate(labels):
        for file_i, file in enumerate(files):

            data = mint.rt_projections[label][file]
            ndx_r = (label_i // n_cols)+1
            ndx_c = label_i % n_cols + 1
            
            fig.add_trace(
                go.Scatter(x=data.index, 
                        y=data.values,
                        name=basename(file),
                        mode='lines',
                        legendgroup=file_i,
                        showlegend=(label_i == 0),  
                        marker_color=colors[file_i],
                        text=file),
                row=ndx_r,
                col=ndx_c,
            )
            
            fig.update_xaxes(title_text="Retention Time", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    fig.update_layout(height=200*len(labels), title_text="Peak Shapes")
    fig.update_layout(xaxis={'title': 'test'})
    
    if 'legend_horizontal' in options:
        fig.update_layout(legend_orientation="h")

    if 'legend' in options:
        fig.update_layout(showlegend=True)
    return fig

@lru_cache(maxsize=32)
@app.callback(
    Output('peakShape3d', 'figure'),
    [Input('b_peakShapes3d', 'n_clicks'),
    Input('peak-select', 'value'),
    Input('check_peakShapes3d', 'value')])
def plot_3d(n_clicks, peakLabel, options):
    if (n_clicks is None) or (mint.rt_projections is None) or (peakLabel is None):
        raise PreventUpdate
    data = mint.rt_projections[peakLabel]
    samples = []
    for i, key in enumerate(list(data.keys())):
        sample = data[key].to_frame().reset_index()
        sample.columns = ['retentionTime', 'intensity']
        sample['peakArea'] = sample.intensity.sum()
        sample['FileName'] = basename(key)
        samples.append(sample)
    samples = pd.concat(samples)
    fig = px.line_3d(samples, x='retentionTime', y='peakArea' , z='intensity', color='FileName')
    fig.update_layout({'height': 800})
    if 'legend_horizontal' in options:
        fig.update_layout(legend_orientation="h")

    if not 'legend' in options:
        fig.update_layout(showlegend=False)
    fig.update_layout({'title': peakLabel})
    return fig


## Results Export (Download)
@app.server.route('/export/')
def download_csv():
    file_buffer = io.BytesIO()
    
    writer = pd.ExcelWriter(file_buffer) #, engine='xlsxwriter')
    mint.results.to_excel(writer, 'Results Complete', index=False)
    mint.crosstab.T.to_excel(writer, 'PeakArea Summary', index=True)
    meta = pd.DataFrame({'Version': [mint.version], 
                            'Date': [str(date.today())]}).T[0]
    meta.to_excel(writer, 'MetaData', index=True, header=False)
    writer.close()
    
    file_buffer.seek(0)

    now = datetime.now().strftime("%Y-%m-%d")
    uid = str(uuid.uuid4()).split('-')[-1]
    return send_file(file_buffer,
                     attachment_filename=f'MINT__results_{now}-{uid}.xlsx',
                     as_attachment=True,
                     cache_timeout=0)


if __name__ == '__main__':
    
    args = sys.argv
    
    if '--debug' in args:
        DEBUG = True
    else:
        DEBUG = False
    
    if DEBUG:
        mint.files = glob('/data/metabolomics_storage/**/*.mzXML', recursive=True)[-4:]
        if isfile('/tmp/mint_results.csv'):
            mint._results = pd.read_csv('/tmp/mint_results.csv')
        for i in mint.files:
            assert isfile(i)
        for i in mint.peaklist_files:
            assert isfile(i)
    
    app.run_server(debug=DEBUG, port=9999)
