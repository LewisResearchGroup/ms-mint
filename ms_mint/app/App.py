import io

import numpy as np
import pandas as pd
import uuid

from datetime import date, datetime
from flask import send_file
from functools import lru_cache
from glob import glob
from tkinter import Tk, filedialog
from pathlib import Path as P

from os.path import basename, isfile, abspath, join

import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable

from ..Mint import Mint
from .Layout import Layout
from .button_style import button_style

from ms_mint.vis.plotly.plotly_tools import plot_peak_shapes, plot_peak_shapes_3d, plot_heatmap
   
from ms_mint.peaklists import read_peaklists, standardize_peaklist, diff_peaklist
from ms_mint.standards import PEAKLIST_COLUMNS
from ms_mint.helpers import remove_all_zero_columns, sort_columns_by_median

mint = Mint()

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP, 
    "https://codepen.io/chriddyp/pen/bWLwgP.css"
                                                ])
app.title = 'MINT'
app.layout = Layout

app.config['suppress_callback_exceptions'] = True

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
    [Output('B_add_files', 'value'),
     Output('table-ms-files', 'data')],
    [Input('B_add_files', 'n_clicks')],
    [State('files-check', 'value'),
     State('table-ms-files', 'data')] )
def select_files(n_clicks, options, ms_files):
    files = []
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
            mint.ms_files += files
            #mint.progress = 0
        root.destroy()
        ms_files +=  [{'MS-files': fn } for fn in files]
    ms_files = [i for n, i in enumerate(ms_files) if i not in ms_files[n + 1:]]
    return str(n_clicks), ms_files

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
     Input('B_reset', 'value'),
     Input('table-ms-files', 'data')])    
def update_files_text(n_clicks, n_clicks_clear, ms_files):
    return '{} data files selected.'.format(mint.n_files), len(ms_files)


### Edit peaklist
@app.callback(
    [Output('table-peaklist-container', 'children'),
     Output('peaklist-div', 'style')],
    [Input('B_peaklists', 'n_clicks'),
     Input('B_reset', 'value'),
     Input('B_add_peak', 'n_clicks'),
     Input('B_detect_peaks', 'n_clicks'),
     Input('B_clear_peaklist', 'n_clicks'),
     Input('int-threshold', 'value'),
     Input('mz-width', 'value')],
    [State('table-peaklist', 'data')])
def select_peaklist(nc_peaklists, nc_reset, add_row, detect_peaks,
                    clear_peaklist, int_thresh, mz_width, data):
    clear = None
    if len( dash.callback_context.triggered ) == 0:
        set_thresh = False
        set_mzwidth = False
    else:
        clear = ( dash.callback_context.triggered[0]['prop_id'].startswith('B_reset') or 
                  dash.callback_context.triggered[0]['prop_id'].startswith('B_clear_peaklist') )
        add_row = dash.callback_context.triggered[0]['prop_id'].startswith('B_add_peak')   
        set_thresh = dash.callback_context.triggered[0]['prop_id'].startswith('int-threshold') 
        set_mzwidth = dash.callback_context.triggered[0]['prop_id'].startswith('mz-width') 
        detect_peaks = dash.callback_context.triggered[0]['prop_id'].startswith('B_detect_peaks')

    columns = [{"name": i, "id": i, 
                "selectable": True}  for i in PEAKLIST_COLUMNS] 
    
    if add_row: 
        defaults = ['', 0 , 0, 0, 0, 0, 'manual']
        append = {c['id']: d for c,d in zip(columns, defaults)}
        data.append(append)
    
    elif detect_peaks:
        mint.detect_peaks()
        peaklist = mint.peaklist
        data = peaklist.to_dict('records')

    elif set_thresh:
        data = pd.DataFrame(data)
        data['intensity_threshold'] = int_thresh
        data = data.to_dict('records')
    
    elif set_mzwidth:
        data = pd.DataFrame(data)
        data['mz_width'] = mz_width
        data = data.to_dict('records')
    
    elif clear:
        mint.clear_peaklist()
        data = mint.peaklist.to_dict('records')

    elif (nc_peaklists is not None) and (not clear):
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        files = filedialog.askopenfilename(multiple=True)
        files = [i  for i in files if (i.endswith('.csv') or (i.endswith('.xlsx')))]
        if len(files) != 0:
            df = read_peaklists(files)
            data = df.to_dict('records')
        root.destroy()
    
    table = DataTable(
                id='table-peaklist',
                columns=columns,
                data=data,
                sort_action="native",
                sort_mode="single",
                row_selectable=False,
                row_deletable=True,
                editable=True,
                column_selectable=False,
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 30,
                filter_action='native',                
                style_table={'overflowX': 'scroll'},
                style_as_list_view=True,
                style_cell={'padding-left': '5px', 
                            'padding-right': '5px'},
                style_header={'backgroundColor': 'white',
                              'fontWeight': 'bold'},
                export_format='csv',
                export_headers='display',
                merge_duplicate_headers=True
            )    

    style = {'display': 'inline'}   
    return [table], style 

@app.callback(
    Output('run-mint-div', 'style'),
    [Input('table-peaklist', 'data')]
    )
def show_run_control(table):
    if len(table) == 0:
        style = {'display': 'none'}
    else:
        style = {'display': 'inline'}
    return style

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
     Input('progress-bar', 'value')],
    [State('table-peaklist', 'data')])
def run_button_style(nc_peaklists, nc_files, nc_reset, progress, peaklist):
    style_peaklists = button_style('next')
    style_files     = button_style('next')    
    style_run       = button_style('wait')
    style_export    = button_style('wait')
    style_heatmap   = button_style('wait')
    if len(peaklist) == 0: 
        style_peaklists = button_style('next')
    elif len(peaklist) != 0:
        style_peaklists = button_style('ready')
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
     Input('B_reset', 'value'),
     Input('table-ms-files', 'data'),
     Input('table-peaklist', 'data')],
    [State('n_cpus', 'value'),
     State('storage', 'children'),
    ])
def run_mint(n_clicks, n_clicks_clear, ms_files, peaklist, n_cpus, old_results):
    
    if n_clicks == 0:
        raise PreventUpdate

    ms_files = pd.DataFrame(ms_files)
    if len(ms_files) == 0 :
        files = []
    else:
        files = ms_files['MS-files'].values
        files = [str(P(i)) for i in files]


    peaklist = pd.DataFrame(peaklist)
    if 'peak_label' not in peaklist.columns:
        print('No column "peak_label" in peaklist.')
        raise PreventUpdate

    peaklist = standardize_peaklist(peaklist)    

    if mint.status == 'running' or len(peaklist) == 0 :
        print('MINT status:', mint.status)
        print('Len peaklist:', len(peaklist))
        raise PreventUpdate
    
    #reset = dash.callback_context.triggered[0]['prop_id'].startswith('B_reset')
    run_mint = dash.callback_context.triggered[0]['prop_id'].startswith('B_run')

    #mint = Mint()
    if run_mint:
        print(files)
        mint.peaklist = peaklist
        mint.files = files

    if old_results is not None:
        #old_results = pd.read_json(old_results, orient='split')
        old_results = mint.results

        old_peaklist_features = standardize_peaklist(old_results[PEAKLIST_COLUMNS].drop_duplicates())
        new_peaklist_features = standardize_peaklist(peaklist[PEAKLIST_COLUMNS].drop_duplicates())
    
        diff = diff_peaklist(old_peaklist_features, 
                             new_peaklist_features)

        if len(diff) == 0:
            print('No difference in peaklist')
        else:
            print(diff)

    if run_mint:
        print("Running MINT")
        mint.run(nthreads=n_cpus)
    
    new_results = mint.results

    print('Old results:', old_results)

    # Update results data with new data
    if (old_results is not None) and (len(old_results) > 0):        
        old_results = old_results.set_index(['peak_label', 'ms_file'])
        new_results = mint.results.set_index(['peak_label', 'ms_file'])
        old_results = old_results.drop(new_results.index, errors='ignore')
        new_results = pd.concat([old_results, new_results]).reset_index()

    # Restrict results to files in file list and peak_labels in peaklist
    print('Results lenght before:', len(new_results))
    print('Files:', files)
    print('Files in results:', new_results.ms_file.drop_duplicates().values)
    print('Peak labels:', peaklist.peak_label)
    print('Peak labels in results:', new_results.peak_label.values)

    new_results = new_results[new_results.ms_file.isin(files) & 
                              new_results.peak_label.isin(peaklist.peak_label)]

    #print('Results columns:', new_results.columns)
    print('Results length after:', len(new_results))

    return new_results.to_json(orient='split')


### Data Table
@app.callback(
    [Output('run-out', 'children'),
     Output('peak-select', 'options'),
     Output('peakshapes-selection', 'options'),
     Output('results-div', 'style')],
    [Input('storage', 'children'),
     Input('label-regex', 'value'),
     Input('table-value-select', 'value'),
     Input('B_reload', 'n_clicks')])
def get_table(json, label_regex, col_value, clicks):
    if json is None:
        raise PreventUpdate

    df = pd.read_json(json, orient='split')
    
    # Columns to show in frontend    
    cols = ['ms_file', 'peak_label', 'peak_area', 'peak_n_datapoints', 'peak_max', 'peak_min',
            'peak_median', 'peak_mean', 'file_size', 'peak_delta_int',
            'peak_rt_of_max']     
    
    # Don't update without data
    if len(df) == 0:
        raise PreventUpdate
    
    # Only show file name, not complete path
    if df['ms_file'].apply(basename).value_counts().max() > 1:
        df['ms_file'] = df['ms_file'].apply(basename)
    
        
    # Extract names of biomarkers for other fontend elements
    # before reducing it to frontend version
    biomarkers = df.groupby('peak_label').mean().peak_max.sort_values(ascending=False).index.astype(str)        
    biomarker_options = [ {'label': i, 'value': i} for i in biomarkers]
    
    # Order dataframe columns               
    df = df[cols]
    
    if col_value != 'full':
        df = pd.crosstab(df.peak_label, df.ms_file, df[col_value].astype(np.float64), aggfunc=np.mean).T
        df = df.loc[:, (df != 0).any(axis=0)]  # remove columns with only zeros
        df = remove_all_zero_columns(df)
        df = sort_columns_by_median(df)
        if col_value in ['peakArea']:
            df = df.round(0)
        df.reset_index(inplace=True)
        df.fillna(0, inplace=True)
        
    # Generate labels 
    if (label_regex is not None) and (label_regex != ''):
        labels = [ '.'.join(i.split('.')[:-1]).split('_')[int(label_regex)] for i in df.ms_file ]
        df['Label'] = labels
    else:
        df['Label'] = df.ms_file
    
    df = df.set_index('Label').reset_index()

    df.columns = df.columns.astype(str)
        
    table = DataTable(
                id='table',
                columns=[{"name": i, "id": i, 
                          "selectable": True,
                          "deletable": True} for i in df.columns],
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
                filter_action='native',
                style_header={'backgroundColor': 'white',
                              'fontWeight': 'bold'},
            )
    
    analysis_style = {}
    
    return table, biomarker_options, biomarker_options, analysis_style


# HEATMAP TOOL
@app.callback(
    [Output('heatmap', 'figure'),
     Output('heatmap', 'style')],
    [Input('B_heatmap', 'n_clicks')],
    [State('checklist', 'value'),
     State('table', 'derived_virtual_indices'),
     State('table', 'data'),
     State('table-value-select', 'value')])
def plot_0(n_clicks, options, ndxs, data, column):
    if (n_clicks is None) or (column == 'full') or (len(data) == 0):
        return {}, {'display': 'none'}
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = pd.DataFrame(data).set_index('Label').select_dtypes(include=numerics)
    
    # Same order as data-table
    df = df.iloc[ndxs]
    print('Heatmap options:', options)
    fig = plot_heatmap(df, 
                       normed_by_cols = 'normed' in options,
                       transposed = 'transposed' in options,
                       correlation = 'correlation' in options,
                       call_show = 'new_tab' in options,
                       add_dendrogram = 'dendrogram' in options,
                       clustered = 'clustered' in options,
                       name=column)    

    if ('new_tab' in options) or (fig is None):
        return {}, {'display': 'none'}
    else:
        return fig, {'min-height': 200, 'width': '100%', 
                     'display': 'inline-block'}


# PEAKEXPLORER
# @lru_cache(maxsize=32)
@app.callback(
    [Output('peakShape', 'figure'),
     Output('peakShape', 'style')],
    [Input('B_shapes', 'n_clicks')],
    [State('n_cols', 'value'),
     State('check_peakShapes', 'value'),
     State('peakshapes-selection', 'value')])
def plot_1(n_clicks, n_cols, options, biomarkers):
    if (len(mint.results) == 0) or (n_clicks == 0):
        return {}, {'display': 'none'}

    if 'legend_horizontal' in options:
        legend_orientation = "h"
    else:
        legend_orientation = 'v'


    fig = plot_peak_shapes(mint.results, n_cols, biomarkers, 
                               legend='legend' in options, 
                               legend_orientation=legend_orientation,
                               call_show='new_tab' in options)

    if ('new_tab' in options) or (fig is None):
        return {}, {'display': 'none'}
    else:
        return fig, {'display': 'inline-block', 'width': '100%'}


@lru_cache(maxsize=32)
@app.callback(
    [Output('peakShape3d', 'figure'),
     Output('peakShape3d', 'style')],
    [Input('B_shapes3d', 'n_clicks')],
    [State('peak-select', 'value'),
     State('check_peakShapes3d', 'value')])
def plot_3d(n_clicks, peak_label, options):
    if (n_clicks is None) or (len(mint.results) == 0) or (peak_label is None):
        return {}, {'display': 'none'}

    if 'legend_horizontal' in options:
        legend_orientation = "h"
    else:
        legend_orientation = 'v'

    fig = plot_peak_shapes_3d(mint.results, peak_label, legend='showlegend' in options, 
                              legend_orientation=legend_orientation, 
                              call_show='new_tab' in options)

    if ('new_tab' in options) or (fig is None):
        return {}, {'display': 'none'}
    else:
        return fig, {'height': 800, 'display': 'inline-block'}


@app.callback(
    Output('heatmap-message', 'children'),
    [Input('table-value-select', 'value')]
)
def return_message(value):
    if value == 'full':
        return 'Heatmap does not work with "Full Table"'
    else:
        return None


## Results Export (Download)
@app.server.route('/export/')
def download_csv():
    file_buffer = io.BytesIO()
    writer = pd.ExcelWriter(file_buffer) #, engine='xlsxwriter')
    mint.peaklist.to_excel(writer, 'Peaklist', index=False)    
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

