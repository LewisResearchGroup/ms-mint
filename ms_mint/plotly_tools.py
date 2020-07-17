import colorlover as cl
import pandas as pd
import plotly_express as px

import plotly.graph_objects as go

from collections.abc import Iterable
from os.path import basename
from plotly.subplots import make_subplots


def plot_peak_shapes(mint, n_cols=3, biomarkers=None, options=None, verbose=False):
    '''
    Returns a plotly multiplost of all peak_shapes in mint.results
    grouped by peak_label.
    '''
    
    res = mint.results[mint.results.peak_area > 0]
    files = list(res.ms_file.drop_duplicates())
    labels = list(mint.peaklist.peak_label.drop_duplicates())
    
    res = res.set_index(['peak_label', 'ms_file'])
    
    if options is None:
        options = []

    if len(biomarkers) != 0:
        labels = [str(i) for i in biomarkers]
            
    # Calculate neccessary number of rows
    n_rows = len(labels)//n_cols
    if n_rows*n_cols < len(labels):
        n_rows += 1
    
    if verbose:
        print(n_rows, n_cols)
        print('ms_files:', files)
        print('peak_labels:', labels)
        print('Data:', res)
    
    fig = make_subplots(rows=max(1, n_rows), 
                        cols=max(1, n_cols), 
                        subplot_titles=labels)
    if len(files) < 13:
        colors = cl.scales['12']['qual']['Paired']
    else:
        colors = cl.interp( cl.scales['12']['qual']['Paired'], len(files))

    # Create sub-plots
    for label_i, label in enumerate(labels):
        for file_i, file in enumerate(files):
            try:
                data = res.loc[(label, file), 'peak_shape']
            except:
                continue
            if not isinstance(data,  Iterable):
                continue
            
            ndx_r = (label_i // n_cols)+1
            ndx_c = label_i % n_cols + 1
                        
            if len(data) == 1:
                mode='markers'
            else:
                mode='lines'
            
            fig.add_trace(
                go.Scatter(
                        x=data.index, 
                        y=data.values,
                        name=basename(file),
                        mode=mode,
                        legendgroup=file_i,
                        showlegend=(label_i == 0),  
                        marker_color=colors[file_i],
                        text=file),
                row=ndx_r,
                col=ndx_c,
            )

            fig.update_xaxes(title_text="Retention Time", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    # Layout
    if 'legend_horizontal' in options:
        fig.update_layout(legend_orientation="h")
    if 'legend' in options:
        fig.update_layout(showlegend=True)
    fig.update_layout(height=400*n_rows, title_text="Peak Shapes")
    return fig


def plot_peak_shapes_3d(mint, peak_label, options=None):
    '''
    Returns a plotly 3D plot of all peak_shapes in mint.results
    where mint.results.peak_label == peak_label.
    '''
    if options is None:
        options = []
    data = mint.results[mint.results.peak_label == peak_label].groupby('ms_file')
    filenames = mint.files
    # Peak labels are supposed to be strings
    # Sometimes they are converted to int though
   
    samples = []
    for i, fn in enumerate(filenames):
        try:
            sample = data.get_group(fn)['peak_shape'].values[0]
        except:
            continue
        if not isinstance(sample, Iterable):
            continue
        else:
            sample = sample.to_frame().reset_index()
        sample.columns = ['retentionTime', 'intensity']
        sample['peak_area'] = sample.intensity.sum()
        sample['ms_file'] = basename(fn)
        samples.append(sample)
    
    if len(samples) == 0:
        return None
    
    samples = pd.concat(samples)
    fig = px.line_3d(samples, x='retentionTime', y='peak_area' , z='intensity', color='ms_file')
    fig.update_layout({'height': 800})
    if 'legend_horizontal' in options:
        fig.update_layout(legend_orientation="h")
    if not 'legend' in options:
        fig.update_layout(showlegend=False)
    fig.update_layout({'title': peak_label, 'title_x': 0.5})
    return fig