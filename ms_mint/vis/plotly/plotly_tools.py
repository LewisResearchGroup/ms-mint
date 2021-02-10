import numpy as np

import colorlover as cl
import pandas as pd
import plotly_express as px

import plotly.graph_objects as go
import plotly.figure_factory as ff

from collections.abc import Iterable
from os.path import basename
from plotly.subplots import make_subplots

import plotly.io as pio


def set_template():
    pio.templates["draft"] = go.layout.Template(
        layout=dict(font={'size': 10}),
    )

    pio.templates.default = "draft"

set_template()

def plot_peak_shapes(mint_results, n_cols=1, biomarkers=None, peak_labels=None, legend=True, 
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
    
    if peak_labels is None:
        peak_labels = []
        #peak_labels = mint_results.groupby('peak_label').mean().peak_max.sort_values(ascending=False).index.astype(str) 

    if biomarkers is None:
        biomarkers = peak_labels

    if biomarkers is None:
        biomarkers = []
        #biomarkers = mint_results.groupby('peak_label').mean().peak_max.sort_values(ascending=False).index.astype(str) 

    if isinstance(biomarkers, str):
        biomarkers = [biomarkers]

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

    if call_show: 
        fig.show(config={'displaylogo': False})
    else:
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
    
    if call_show: 
        fig.show(config={'displaylogo': False})
    else:
        return fig



def plot_heatmap(df, normed_by_cols=False, transposed=False, clustered=False,
                 add_dendrogram=False, name='', x_tick_colors=None,
                 correlation=False, call_show=False, verbose=False):


    max_is_not_zero = df.max(axis=1) != 0
    non_zero_labels = max_is_not_zero[max_is_not_zero].index
    df = df.loc[non_zero_labels]

    plot_type = 'Heatmap'
    colorscale = 'Bluered'
    plot_attributes = []
    
    if normed_by_cols:
        df = df.divide(df.max()).fillna(0)
        plot_attributes.append('normalized')

    if transposed:
        df = df.T
        
    if correlation:
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
        
    if clustered:
        dendro_side = ff.create_dendrogram(df, orientation='right', labels=df.index.to_list(),
                                           color_threshold=0, colorscale=['black']*8)
        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        df = df.loc[dendro_leaves,:]
        if correlation:
            df = df[df.index]        

    x = df.columns
    if clustered:
        y = dendro_leaves
    else:
        y = df.index.to_list()
    z = df.values

    heatmap = go.Heatmap(x=x, y=y, z=z, colorscale=colorscale)
    
    title = f'{plot_type} of {",".join(plot_attributes)} {name}'

    # Figure without side-dendrogram
    if (not add_dendrogram) or (not clustered):
        fig = go.Figure(heatmap)
        fig.update_layout(
            {'title_x': 0.5},
            title={'text': title},
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
                {'height': 800,
                 'showlegend':False,
                 'hovermode': 'closest',
                 'paper_bgcolor': 'white',
                 'plot_bgcolor': 'white',
                 'title_x': 0.5
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

    fig.update_layout(
        margin=dict( l=50, r=10, b=200, t=50, pad=0 ),
        hovermode='closest')

    if call_show: 
        fig.show(config={'displaylogo': False})
    else:
        return fig