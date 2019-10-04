import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px

from dash.dependencies import Input, Output
from tkinter import Tk, filedialog

from mint.backend import Mint
import numpy as np
import pandas as pd

mint = Mint()

DEVEL = True

if DEVEL:
    from glob import glob
    mint.mzxml_files = glob('/data/Mint_Demo_Files/**/*.mzXML', recursive=True)

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

app.layout = html.Div(
    [
        html.H1("Mint-Dash"),
        html.Button('Select mzXML file(s)', id='mzxml'),
        html.Button('Select peaklist file(s)', id='peaklist'),
        html.Button('Run', id='run'),
        #dbc.Progress(id="progress", value=0, striped=True, animated=True),
        #dcc.Interval(id="interval", interval=1000, n_intervals=0),
        html.Div(id='mzxml-out', children=''),
        html.Div(id='peaklist-out', children=''),
        html.Div(id='run-out', children=[
                dash_table.DataTable(id='results', data=np.array([]))
            ]),
        html.Button('Plot 1', id='plot'),
        html.Button('Plot 2', id='plot2'),
        html.Button('Plot 3', id='plot3'),
        html.Button('Plot 4', id='plot4'),

        dcc.Graph(id='figure', figure={}),
        dcc.Graph(id='figure2', figure={}),
        dcc.Graph(id='figure3', figure={}),
        dcc.Graph(id='figure4', figure={}),
    ]
)

@app.callback(
    Output('mzxml-out', 'children'),
    [Input('mzxml', 'n_clicks')] )
def select_mzxml(n_clicks):
    if n_clicks is not None:
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        files = filedialog.askopenfilename(multiple=True)
        files = [i  for i in files if i.endswith('.csv')]
        if len(files) != 0:
            mint.mzxml_files = files
        root.destroy()
    if len(mint.mzxml_files) == 0:
        return 'Select mzXML file(s)'
    else:
        return '{} mzxml-files selected'.format(len(mint.mzxml_files))

@app.callback(
    Output('peaklist-out', 'children'),
    [Input('peaklist', 'n_clicks')] )
def select_peaklist(n_clicks):
    if n_clicks is not None:
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        files = filedialog.askopenfilename(multiple=True)
        files = [i  for i in files if i.endswith('.csv')]
        if len(files) != 0:
            mint.peaklist_files = files
        root.destroy()
    if len(mint.peaklist_files) == 0:
        return 'Select peaklist file(s)'
    else:
        return '{} peaklist-files selected'.format(len(mint.peaklist_files))

@app.callback(
    Output('run-out', 'children'),
    [Input('run', 'n_clicks')] )
def run_mint(n_clicks):
    if n_clicks is not None:
        mint.process_files()
    df = mint.results
    return dash_table.DataTable(
        id='results',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        #editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable="multi",
        #row_deletable=True,
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10)

@app.callback(
    Output('figure', 'figure'),
    [Input('plot', 'n_clicks'),
     Input('results', 'data')])
def plot_selected_peaks(n_clicks, data):
    if n_clicks is None:
        return {}
    df = mint.results.copy()
    df['logPeakArea'] = df.peakArea.apply(np.log1p)
    fig = px.scatter(df, 
        x="rtmin", y="logPeakArea", color="logPeakArea",
        animation_frame="mzxmlFile",
        range_y=[0.1, max(df.logPeakArea * 1.5)],
        range_x=[0, max(df.rtmax)], size_max=20,
        hover_data=['peakLabel'], title='Plot 1')
    return fig

@app.callback(
    Output('figure2', 'figure'),
    [Input('plot2', 'n_clicks'),
     Input('results', 'data')])
def plot_selected_peaks(n_clicks, data):
    if n_clicks is None:
        return {}
    df = mint.results.copy()
    df['logPeakArea'] = df.peakArea.apply(np.log1p)
    fig = px.scatter(df, 
        x="rtmin", y="peakMz", color="logPeakArea",
        size="logPeakArea",
        animation_frame="mzxmlFile",
        range_y=[0, max(df.peakMz * 1.5)],
        range_x=[0, max(df.rtmax)], size_max=20,
        hover_data=['peakLabel'], title='Plot 2')
    return fig

@app.callback(
    Output('figure3', 'figure'),
    [Input('plot3', 'n_clicks')])
def plot_selected_peaks(n_clicks):
    if n_clicks is None:
        return {}
    df = mint.all_df.copy()
    cols = ['retentionTime',
        'm/z array',
        'intensity array',
        'mzxmlFile']

    fig = px.density_contour(mint.all_df[cols][mint.all_df['intensity array']>100000], 
            x='retentionTime',
            y='m/z array',
            z='intensity array',
            animation_frame='mzxmlFile',
            histfunc='sum',
            range_x=[0, mint.all_df['retentionTime'].max()],
            range_y=[0, mint.all_df['m/z array'].max()],
            nbinsx=150,
            nbinsy=150,
            marginal_x='histogram',
            marginal_y='histogram',
            title='Plot 3')
    return fig

@app.callback(
    Output('figure4', 'figure'),
    [Input('plot4', 'n_clicks')])
def plot_selected_peaks(n_clicks):
    if n_clicks is None:
        return {}
    df = mint.all_df.copy()
    cols = ['retentionTime',
        'm/z array',
        'intensity array',
        'mzxmlFile']

    fig = px.density_contour(mint.all_df[cols][mint.all_df['intensity array']>100000], 
            x='retentionTime',
            y='m/z array',
            z='intensity array',
            color='mzxmlFile',
            histfunc='sum',
            range_x=[0, mint.all_df['retentionTime'].max()],
            range_y=[0, mint.all_df['m/z array'].max()],
            nbinsx=150,
            nbinsy=150,
            marginal_x='histogram',
            marginal_y='histogram',
            title='Plot 4')
    fig.update_layout(legend_orientation="h")        
    return fig




#@app.callback(Output("progress", "value"), 
#              [Input("interval", "n_intervals")])
#def advance_progress(n):
#    value = int(100 * mint._n_files_processed / mint._n_files)
#    print('Progress:', value)
#    return value

def callback_progress(i, j):
    print(i,j)
    return i

mint.callback_progress = callback_progress

app.run_server(debug=True, port=9995)
