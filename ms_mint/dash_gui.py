
import io

import numpy as np
import pandas as pd
import uuid

from datetime import date, datetime
from flask import send_file
from functools import lru_cache
from glob import glob
from tkinter import Tk, filedialog

from os.path import basename, isfile, abspath, join

import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable
import dash_html_components as html

import plotly.graph_objects as go
import plotly.figure_factory as ff

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

from ms_mint.Mint import Mint
from ms_mint.dash_layout import Layout
from ms_mint.button_style import button_style
from ms_mint.plotly_tools import plot_peak_shapes, plot_peak_shapes_3d

mint = Mint()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 
                                                "https://codepen.io/chriddyp/pen/bWLwgP.css"])
app.title = 'MINT'
app.layout = Layout

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
    Output('B_add_files', 'value'),
    [Input('B_add_files', 'n_clicks')],
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
                files = glob(join(dir_, join('**', '*.mz*ML')), recursive=True)
            else:
                files = []
        if len(files) != 0:
            mint.files += files
            mint.progress = 0
        root.destroy()
    return str(n_clicks)

### Clear files
@app.callback(
    Output('B_reset', 'value'),
    [Input('B_reset', 'n_clicks')])
def clear_files(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    mint.reset()
    return str(n_clicks)
    
### Update n-files text when files added or cleared
@app.callback(
    [Output('files-text', 'children'),
     Output('n_files_selected', 'children')],
    [Input('B_add_files', 'value'),
     Input('B_reset', 'value')])    
def update_files_text(n_clicks, n_clicks_clear):
    return '{} data files selected.'.format(mint.n_files), mint.n_files


### Load peaklist files
@app.callback(
    [Output('peaklist-text', 'children'),
     Output('n_peaklist_selected', 'children')],
    [Input('B_peaklists', 'n_clicks'),
     Input('B_reset', 'value')] )
def select_peaklist(nc_peaklists, nc_reset):
    clear = dash.callback_context.triggered[0]['prop_id'].startswith('B_reset')
    if (nc_peaklists is not None) and (not clear):
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        files = filedialog.askopenfilename(multiple=True)
        files = [i  for i in files if i.endswith('.csv')]
        if len(files) != 0:
            mint.peaklist_files = files
        root.destroy()
    if len(mint.messages) == 0:
        return '{} peaklist-files selected.'.format(mint.n_peaklist_files), mint.n_peaklist_files
    else:
        return '\n'.join(mint.messages).upper(), -1


### Button styles
@app.callback(
    [Output('B_peaklists', 'style'),
     Output('B_add_files', 'style'),
     Output('B_run', 'style'),
     Output('B_export', 'style'),
     Output('B_heatmap', 'style'),
     Output('B_shapes', 'style'),
     Output('B_shapes3d', 'style')],
    [Input('B_peaklists', 'n_clicks'),
     Input('B_add_files', 'n_clicks'),
     Input('B_reset', 'n_clicks'),
     Input('progress-bar', 'value')])
def run_button_style(nc_peaklists, nc_files, nc_reset, progress):
    style_peaklists = button_style('next')
    style_files     = button_style('wait')    
    style_run       = button_style('wait')
    style_export    = button_style('wait')
    style_heatmap   = button_style('wait')
    if mint.n_peaklist_files == -1: 
        style_peaklists = button_style('error')
    elif mint.n_peaklist_files != 0:
        style_peaklists = button_style('ready')
        style_files     = button_style('next') 
    if mint.n_files != 0:
        style_files= button_style('ready')
        style_run = button_style('next')
    if (mint.n_files != 0) and (mint.n_files != 0) and (progress == 100):
        style_run = button_style('ready')
        style_export = button_style('next')
    if len(mint.results) > 0:
        style_heatmap = button_style('ready')
         
    return (style_peaklists, style_files, style_run, 
            style_export, style_heatmap, style_heatmap, 
            style_heatmap)

### CPU text
@app.callback(
    Output('cpu-text', 'children'),
    [Input('n_cpus', 'value')] )
def mint_cpu_info(value):
    return f'Using {value} cores.'


### Storage
@app.callback(
    Output('storage', 'children'),
    [Input('B_run', 'n_clicks'),
     Input('B_reset', 'value')],
    [State('n_cpus', 'value')])
def run_mint(n_clicks, n_clicks_clear, n_cpus):
    clear = dash.callback_context.triggered[0]['prop_id'].startswith('B_reset')
    if mint.status == 'running':
        raise PreventUpdate
    if (n_clicks is not None) and (not clear):
        mint.run(nthreads=n_cpus)
    return mint.results.to_json(orient='split')


### Data Table
@app.callback(
    [Output('run-out', 'children'),
     Output('peak-select', 'options'),
     Output('peakshapes-selection', 'options'),
     Output('analysis', 'style')],
    [Input('storage', 'children'),
     Input('label-regex', 'value'),
     Input('table-value-select', 'value')])
def get_table(json, label_regex, col_value):
    df = pd.read_json(json, orient='split')
     
    # Columns to show in frontend    
    cols = ['Label', 'peak_label', 'peak_area', 'peak_max', 'peak_min',
            'peak_median', 'peak_mean', 'file_size', 'peak_delta_int', 'peak_rt_of_max']     
    
    # Don't update without data
    if len(df) == 0:
        raise PreventUpdate
    
    # Only show file name, not complete path
    if df['ms_file'].apply(basename).value_counts().max() > 1:
        df['ms_file'] = df['ms_file'].apply(basename)
    
    # Generate labels 
    try:
        labels = [ '.'.join(i.split('.')[:-1]).split('_')[int(label_regex)] for i in df.ms_file ]
        df['Label'] = labels
    except:
        df['Label'] = df.ms_file
        
    # Extract names of biomarkers for other fontend elements
    # before reducing it to frontend version
    biomarkers = df.peak_label.drop_duplicates().sort_values().values        
    biomarker_options = [ {'label': i, 'value': i} for i in biomarkers]
    
    # Order dataframe columns               
    df = df[cols]
    
    if col_value != 'full':
        df = pd.crosstab(df.peak_label, df.Label, df[col_value], aggfunc=np.mean).astype(np.float64).T
        if col_value in ['peakArea']:
            df = df.round(0)
        df.reset_index(inplace=True)
        df.fillna(0, inplace=True)
    
    df.columns = df.columns.astype(str)
        
    table = DataTable(
                id='table',
                columns=[{"name": i, "id": i, "selectable": True} for i in df.columns],
                data=df.to_dict('records'),
                sort_action="native",
                sort_mode="single",
                row_selectable=False,
                row_deletable=False,
                column_selectable=False,
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
    
    analysis_style = {'display': 'inline'}
    
    return table, biomarker_options, biomarker_options, analysis_style


# HEATMAP TOOL
@app.callback(
    Output('heatmap', 'figure'),
    [Input('B_heatmap', 'n_clicks'),
     Input('checklist', 'value')],
    [State('table', 'derived_virtual_indices'),
     State('table', 'data'),
     State('table-value-select', 'value')])
def plot_0(n_clicks, options, ndxs, data, column):
    if (n_clicks is None) or (column == 'full') or (len(data) == 0):
        raise PreventUpdate
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = pd.DataFrame(data).set_index('Label').select_dtypes(include=numerics)
    
    # Same order as data-table
    df = df.iloc[ndxs]
    
    max_is_not_zero = df.max(axis=1) != 0
    non_zero_labels = max_is_not_zero[max_is_not_zero].index
    df = df.loc[non_zero_labels]

    plot_type = 'Heatmap'
    colorscale = 'Blues'
    plot_attributes = []
    
    if 'normed' in options:
        df = df.divide(df.max()).fillna(0)
        plot_attributes.append('normalized')

    if 'transposed' in options:
        df = df.T
        
    if 'corr' in options:
        plot_type = 'Correlation'
        df = df.corr()
        colorscale = [[0.0, "rgb(165,0,38)"],
                [0.1111111111111111, "rgb(215,48,39)"],
                [0.2222222222222222, "rgb(244,109,67)"],
                [0.3333333333333333, "rgb(253,174,97)"],
                [0.4444444444444444, "rgb(254,224,144)"],
                [0.5555555555555556, "rgb(224,243,248)"],
                [0.6666666666666666, "rgb(171,217,233)"],
                [0.7777777777777778, "rgb(116,173,209)"],
                [0.8888888888888888, "rgb(69,117,180)"],
                [1.0, "rgb(49,54,149)"]]
    else:
        plot_type = 'Heatmap'
        
    if 'clustered' in options:
        D = squareform(pdist(df, metric='seuclidean'))
        Y = linkage(D, method='complete')
        Z = dendrogram(Y, orientation='left', no_plot=True)['leaves']
        Z.reverse()
        df = df.iloc[Z,:]
        if 'corr' in options:
            df = df[df.index]
            
        dendro_side = ff.create_dendrogram(df, orientation='right', labels=df.index)

    heatmap = go.Heatmap(z=df.values,
                         x=df.columns,
                         y=df.index,
                         colorscale = colorscale)
    
    title = f'{plot_type} of {",".join(plot_attributes)} {column}'

    # Figure without side-dendrogram
    if (not 'dendrogram' in options) or (not 'clustered' in options):
        fig = go.Figure(heatmap)
        fig.update_layout(
            title={'text': title,  },
            yaxis={'title': '', 
                   'tickmode': 'array', 
                   'automargin': True}) 
        fig.update_layout({'height':800, 
                           'hovermode': 'closest'})
        
    else:  # Figure with side-dendrogram
        fig = go.Figure()
        
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        for data in dendro_side['data']:
            fig.add_trace(data)
            
        y_labels = heatmap['y']
        heatmap['y'] = dendro_side['layout']['yaxis']['tickvals']
        
        fig.add_trace(heatmap)     

        fig.update_layout(
                {'height':800,
                 'showlegend':False,
                 'hovermode': 'closest',
                 'paper_bgcolor': 'white',
                 'plot_bgcolor': 'white'
                },
                title={'text': title},
                
                # X-axis of main figure
                xaxis={'domain': [.11, 1],        
                       'mirror': False,
                       'showgrid': False,
                       'showline': False,
                       'zeroline': False,
                       'showticklabels': True,
                       'ticks':""
                      },
                # X-axis of side-dendrogram
                xaxis2={'domain': [0, .1],  
                        'mirror': False,
                        'showgrid': True,
                        'showline': False,
                        'zeroline': False,
                        'showticklabels': False,
                        'ticks':""
                       },
                # Y-axis of main figure
                yaxis={'domain': [0, 1],
                       'mirror': False,
                       'showgrid': False,
                       'showline': False,
                       'zeroline': False,
                       'showticklabels': False,
                      })
        fig['layout']['yaxis']['ticktext'] = np.asarray(y_labels)
        fig['layout']['yaxis']['tickvals'] = np.asarray(dendro_side['layout']['yaxis']['tickvals'])
    return fig



# PEAKEXPLORER
@lru_cache(maxsize=32)
@app.callback(
    Output('peakShape', 'figure'),
    [Input('B_shapes', 'n_clicks')],
    [State('n_cols', 'value'),
     State('check_peakShapes', 'value'),
     State('peakshapes-selection', 'value')])
def plot_1(n_clicks, n_cols, options, biomarkers):
    if (len(mint.results) == 0) or (n_clicks == 0):
        raise PreventUpdate
    new_fig = plot_peak_shapes(mint, n_cols, biomarkers, options)
    if new_fig is None:
        raise PreventUpdate
    return new_fig


@lru_cache(maxsize=32)
@app.callback(
    Output('peakShape3d', 'figure'),
    [Input('B_shapes3d', 'n_clicks'),
     Input('peak-select', 'value'),
     Input('check_peakShapes3d', 'value')])
def plot_3d(n_clicks, peak_label, options):
    if (n_clicks is None) or (len(mint.results) == 0) or (peak_label is None):
        raise PreventUpdate
    new_fig = plot_peak_shapes_3d(mint, peak_label, options)
    if new_fig is None:
        raise PreventUpdate
    return new_fig

@app.callback(
    Output('heatmap-message', 'children'),
    [Input('table-value-select', 'value')]
)
def return_messave(value):
    if value == 'full':
        return 'Heatmap does not work with "Full Table"'
    else:
        return None


## Results Export (Download)
@app.server.route('/export/')
def download_csv():
    file_buffer = io.BytesIO()
    writer = pd.ExcelWriter(file_buffer) #, engine='xlsxwriter')
    mint.results.to_excel(writer, 'Results Complete', index=False)
    mint.crosstab().T.to_excel(writer, 'PeakArea Summary', index=True)
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

