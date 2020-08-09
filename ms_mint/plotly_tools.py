import colorlover as cl
import pandas as pd
import plotly_express as px

import plotly.graph_objects as go

from collections.abc import Iterable
from os.path import basename
from plotly.subplots import make_subplots



def plot_peak_shapes(mint_results, n_cols=3, biomarkers=None, legend=True, 
                     verbose=False, legend_orientation='v', call_show=False):
    '''
    Returns a plotly multiplost of all peak_shapes in mint.results
    grouped by peak_label.
    '''
    mint_results = mint_results.copy()
    mint_results.ms_file = [basename(i) for i in mint_results.ms_file]
    
    res = mint_results[mint_results.peak_area > 0]
    files = list(res.ms_file.drop_duplicates())
    labels = list(mint_results.peak_label.drop_duplicates())
    
    res = res.set_index(['peak_label', 'ms_file'])

    if biomarkers is None:
        biomarkers = []

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
                x, y = res.loc[(label, file), ['peak_shape_rt', 'peak_shape_int']]

            except:
                continue
            if not isinstance(x,  Iterable):
                continue
            
            if isinstance(x, str):
                x = x.split(',')
                y = y.split(',')
                
            ndx_r = (label_i // n_cols)+1
            ndx_c = label_i % n_cols + 1
                        
            if len(x) == 1:
                mode='markers'
            else:
                mode='lines'
            
            fig.add_trace(
                go.Scatter(
                        x=x, 
                        y=y,
                        name=basename(file),
                        mode=mode,
                        legendgroup=file_i,
                        showlegend=(label_i == 0),  
                        marker_color=colors[file_i],
                        text=file),
                row=ndx_r,
                col=ndx_c,
            )

            fig.update_xaxes(title_text="Scan Time", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    # Layout
    if legend:
        fig.update_layout(legend_orientation=legend_orientation)
    
    fig.update_layout(showlegend=legend)
    
    fig.update_layout(height=400*n_rows, title_text="Peak Shapes")
    if call_show: fig.show(config={'displaylogo': False})
    return fig


def plot_peak_shapes_3d(mint_results, peak_label=None, legend=True, 
                        legend_orientation='v', call_show=False, verbose=False):
    '''
    Returns a plotly 3D plot of all peak_shapes in mint.results
    where mint.results.peak_label == peak_label.
    '''

    mint_results = mint_results.copy()
    mint_results.ms_file = [basename(i) for i in mint_results.ms_file]

    data = mint_results[mint_results.peak_label == peak_label]
    files = list( data.ms_file.drop_duplicates() )

    grps = data.groupby('ms_file')

    # Peak labels are supposed to be strings
    # Sometimes they are converted to int though
   
    samples = []
    for i, fn in enumerate(files):
        grp = grps.get_group(fn)
        try:
            x, y, peak_max = grp[['peak_shape_rt', 'peak_shape_int', 'peak_max']].values[0]
        except:
            continue
            
        if isinstance(x, str):
            x = x.split(',')
            y = y.split(',')
        sample = pd.DataFrame({'Scan Time': x, 'Intensity': y})
        sample['peak_max'] = peak_max
        sample['ms_file'] = basename(fn)
        samples.append(sample)
    
    if len(samples) == 0:
        return None
    
    samples = pd.concat(samples)
    
    fig = px.line_3d(samples, x='Scan Time', y='peak_max' , z='Intensity', color='ms_file')
    fig.update_layout({'height': 1000, 'width': 1000})
    
    # Layout
    if legend:
        fig.update_layout(legend_orientation=legend_orientation)
    
    fig.update_layout(showlegend=legend)
        
    fig.update_layout({'title': peak_label, 'title_x': 0.5})
    if call_show: fig.show(config={'displaylogo': False})
    return fig