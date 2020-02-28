import colorlover as cl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from os.path import basename
import numpy as np
from collections.abc import Iterable


def plot_peak_shapes(mint, n_cols=3, options=None):
    res = mint.results.set_index(['peak_label', 'ms_file'])
     
    if options is None:
        options = []
        
    files = mint.files
    labels = mint.peaklist.peak_label.values
    print('Files:', files)
    print('Labels:', labels)
    
    # Calculate neccessary number of rows
    n_rows = len(labels)//n_cols
    if n_rows*n_cols < len(labels):
        n_rows += 1
    
    fig = make_subplots(rows=n_rows, 
                        cols=n_cols, 
                        subplot_titles=labels)
    if len(files) < 13:
        colors = cl.scales['12']['qual']['Paired']
    else:
        colors = cl.interp( cl.scales['12']['qual']['Paired'], len(files))

    # Create sub-plots
    for label_i, label in enumerate(labels):
        for file_i, file in enumerate(files):
            data = res.loc[(label, file), 'peak_shape']
            if not isinstance(data,  Iterable):
                continue
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

            fig.update_xaxes(title_text="Scan Time", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    # Layout
    if 'legend_horizontal' in options:
        fig.update_layout(legend_orientation="h")
    if 'legend' in options:
        fig.update_layout(showlegend=True)
    fig.update_layout(height=400*n_rows, title_text="Peak Shapes")

    return fig

