import os
import random

from tqdm import tqdm

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns  

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State, ALL

import plotly.graph_objects as go

from ms_mint.Mint import Mint
from ms_mint.peak_optimization.RetentionTimeOptimizer import RetentionTimeOptimizer as RTOpt

from . import tools as T

info_txt = '''
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
'''


_layout = html.Div([
    html.H3('Peak Optimization'),
    html.Button('Generate peak previews', id='pko-peak-preview'),
    html.Button('Find largest peaks for all peaks', 
        id='pko-find-closest-peak', style={'visibility': 'hidden'}),
    dcc.Markdown('---'),
    html.Div(id='pko-peak-preview-output', 
        style={"maxHeight": "300px", "overflowY": "scroll", 'padding': 'auto'}),
    dcc.Markdown('---'),
    dcc.Markdown(id='pko-find-closest-peak-output'),
    dcc.Markdown(id='pko-find-largest-peak-output', 
            style={'visibility': 'hidden'}),

    html.Div(id='pko-controls'),
    dcc.Dropdown(
        id='pko-dropdown',
        options=[],
        value=None
    ),
    dcc.Loading( dcc.Graph('pko-figure') ),
    dcc.Markdown(id='pko-set-rt-output'),
    html.Button('Set RT to current view', id='pko-set-rt'),
    html.Button('Find largest peak', id='pko-find-largest-peak'),

    html.Button('Remove Peak', id='pko-delete', style={'float': 'right'}),

    html.Div([
        html.Button('<< Previous', id='pko-prev'), html.Button('Next >>', id='pko-next')],
        style={'text-align': 'center', 'margin': 'auto', 'margin-top': '10%'}),
    html.Div(id='pko-image-clicked-output'),
    html.Div(id='pko-delete-output')
])

pko_layout_no_data = html.Div([
    dcc.Markdown('''### No peaklist found.
    You did not generate a peaklist yet.
    ''')
])


def layout():
    return _layout 


def callbacks(app, fsc, cache):
        
    @app.callback(
    Output('pko-dropdown', 'options'),
    Input('tab', 'value'),
    Input('pko-delete-output', 'children'),
    State('wdir', 'children')
    )
    def pko_controls(tab, peak_deleted, wdir):
        if tab != 'pko':
            raise PreventUpdate
        peaklist = T.get_peaklist( wdir )
        if peaklist is None:
            raise PreventUpdate
        options = [{'label':label, 'value': i} for i, label in enumerate(peaklist.index)]
        return options


    @app.callback(
    Output('pko-figure', 'figure'),
    Input('pko-dropdown', 'value'),
    Input('pko-set-rt-output', 'children'),
    Input('pko-dropdown', 'options'),
    Input('pko-find-largest-peak-output', 'children'),
    State('wdir', 'children'),
    State('pko-figure', 'figure')
    )
    def pko_figure(peak_label_ndx, n_clicks, options_changed, 
            find_largest_peak, wdir, fig):
        if peak_label_ndx is None:
            raise PreventUpdate
        
        peaklist = T.get_peaklist( wdir ).reset_index()
        ms_files = T.get_ms_fns( wdir )
        random.shuffle(ms_files)
        ms_files = ms_files[:60]

        peak_label_ndx = peak_label_ndx % len(peaklist)
        mz_mean, mz_width, rt, rt_min, rt_max, label = \
            peaklist.loc[peak_label_ndx, ['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max', 'peak_label']]

        if (rt_min is None) or np.isnan(rt_min): rt_min = rt-0.2
        if (rt_max is None) or np.isnan(rt_max): rt_max = rt+0.2
        if (rt is None) or np.isnan(rt): rt = np.mean([rt_min, rt_max])

        if True or fig is None:
            fig = go.Figure()
            fig.layout.hovermode = 'closest'
            fig.layout.xaxis.range=[rt_min, rt_max]

            fig.update_layout(
                yaxis_title="MS-Intensity",
                xaxis_title="Retention Time [min]",                
                xaxis=dict( 
                    rangeslider=dict( 
                        visible=True
                    )
                )
            )
            fig.update_layout(title=label)

        fig.add_vline(rt)
        fig.add_vrect(x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1)

        n_files = len(ms_files)
        for i, fn in tqdm(enumerate(ms_files)):
            fsc.set('progress', int(100*(i+1)/n_files))
            name = os.path.basename(fn)
            name, _ = os.path.splitext(name)
            chrom = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
            fig.add_trace(
                go.Scatter(x=chrom['retentionTime'], 
                        y=chrom['intensity array'], 
                        name=name)
            )
            fig.update_layout(showlegend=False)
        return fig


    @app.callback(
    Output('pko-find-closest-peak-output', 'children'),
    Input('pko-find-closest-peak', 'n_clicks'),
    State('wdir', 'children'))
    def pko_find_closest_peak(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate
        fn_pkl = os.path.join( wdir, 'peaklist', 'peaklist.csv') 
        
        def set_progress(x):
            fsc.set('progress', x)

        mint = Mint(verbose=True, progress_callback=set_progress)
        mint.peaklist_files = fn_pkl
        mint.optimize_retention_times()
        mint.peaklist.to_csv(fn_pkl)


    @app.callback(
        Output('pko-set-rt-output', 'children'),
        Input('pko-set-rt', 'n_clicks'),
        State('pko-dropdown', 'value'),
        State('pko-figure', 'figure'),
        State('wdir', 'children')
    )
    def pko_set_rt(n_clicks, peak_label, fig, wdir):
        if n_clicks is None:
            raise PreventUpdate
        rt_min, rt_max = fig['layout']['xaxis']['range']
        rt_min, rt_max = np.round(rt_min, 4), np.round(rt_max, 4)
        T.update_peaklist(wdir, peak_label, rt_min, rt_max)
        return f'Set RT span to ({rt_min},{rt_max})'


    @app.callback(
        Output('pko-dropdown', 'value'),
        Input('pko-prev', 'n_clicks'),
        Input('pko-next', 'n_clicks'),
        Input('pko-image-clicked-output', 'children'),
        State('pko-dropdown', 'value'),
        State('pko-dropdown', 'options')
    )
    def pko_prev_next(n_prev, n_next, image_clicked, value, options):
        if n_prev is None and n_next is None and image_clicked is None:
            raise PreventUpdate
        prop_id = dash.callback_context.triggered[0]['prop_id']        
        if prop_id.startswith('pko-image-clicked-output'):
            for entry in options:
                if entry['label'] == image_clicked:
                    return entry['value']
        elif value is None:
            return 0
        elif prop_id.startswith('pko-prev'):
            return (value - 1) % len(options)
        elif prop_id.startswith('pko-next'):
            return (value + 1) % len(options)

    @app.callback(
        Output('pko-peak-preview-output', 'children'),
        Input('pko-image-store', 'children'),
        Input('tab', 'value')
    )
    def pko_show_images(images, tab):
        if tab != 'pko': raise PreventUpdate
        return images

    @app.callback(
    Output('pko-image-store', 'children'),
    Input('pko-peak-preview', 'n_clicks'),
    State('wdir', 'children')
    )
    def peak_preview(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate 
        sns.set_context('paper')
        ms_files = T.get_ms_fns(wdir)
        random.shuffle(ms_files)
        ms_files = ms_files[:40]
        peaklist = T.get_peaklist(wdir)
        n_total = len(peaklist)
        images = []
        for i, (ndx, row) in tqdm( enumerate(peaklist.iterrows()), total=n_total ):
            peak_label = ndx
            fsc.set('progress', int(100*(i+1)/n_total))
            mz_mean, mz_width, rt, rt_min, rt_max = row[['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']]

            if np.isnan(rt_min) or rt_min is None: rt_min = rt-0.5
            if np.isnan(rt_max) or rt_max is None: rt_max = rt+0.5

            plt.figure(figsize=(4,2.5))
            for fn in ms_files:
                try:        
                    fn_chro = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
                    fn_chro = fn_chro[(rt_min < fn_chro['retentionTime']) &
                                      (fn_chro['retentionTime'] < rt_max)]
                    plt.plot(fn_chro['retentionTime'], fn_chro['intensity array'], lw=1, color='k')
                except:
                    pass

            plt.gca().set_title(ndx, y=1.0, pad=15)
            plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.xlabel('Retention Time [min]')  
            plt.ylabel('MS-Intensity')  

            T.savefig(kind='peak-preview', wdir=wdir, label=ndx)
            
            src = T.fig_to_src(dpi=80)

            images.append(
                html.A(id={'index': peak_label, 'type': 'image'}, children=html.Img(src=src), style={'float': 'center'})
            )
            #images.append( dbc.Tooltip(ndx, target=_id, style={'font-size=': 'large'}) )
        return images

    @app.callback(
        Output('pko-image-clicked-output', 'children'),
        Input({'type': 'image', 'index': ALL}, 'n_clicks'),
    )
    def pko_image_clicked(ndx):
        if ndx is None or len(ndx)==0: raise PreventUpdate
        ctx = dash.callback_context
        print(ctx.triggered[0]['prop_id'])
        clicked = ctx.triggered[0]['prop_id']
        clicked = clicked.replace('{"index":"', '')
        clicked = clicked.split('","type":')[0].replace('\\', '')
        print('Clicked:', clicked)
        return clicked

    @app.callback(
        Output('pko-delete-output', 'children'),
        Input('pko-delete', 'n_clicks'),
        State('pko-dropdown', 'value'),
        State('wdir', 'children')
    )
    def plk_delete(n_clicks, peak_label, wdir):
        if n_clicks is None:
            raise PreventUpdate
        fn = T.get_peaklist_fn( wdir )
        peaklist = T.get_peaklist( wdir ).reset_index()
        peaklist = peaklist.drop( peak_label, axis=0 )
        peaklist.to_csv(fn, index=False)
        return f'{peak_label} removed from peaklist.'  

    @app.callback(
        Output('pko-find-largest-peak-output', 'children'),
        Input('pko-find-largest-peak', 'n_clicks'),
        State('pko-dropdown', 'value'),
        State('wdir', 'children')
    )
    def find_largest_peak(n_clicks, peak_label_ndx, wdir):
        if n_clicks is None: raise PreventUpdate
        if peak_label_ndx is None: raise PreventUpdate
        peaklist = T.get_peaklist( wdir )
        ms_files = T.get_ms_fns( wdir )
        random.shuffle(ms_files)
        ms_files = ms_files[:50]
        row = peaklist.iloc[peak_label_ndx]
        mz_mean, mz_width = row.loc[['mz_mean', 'mz_width']]
        chromatograms = [T.get_chromatogram(fn, mz_mean, mz_width, wdir).set_index('retentionTime')['intensity array'] for fn in ms_files]
        rt_min, rt_max = None, None
        rt_min, rt_max = RTOpt().find_largest_peak(chromatograms)
        peaklist = peaklist.reset_index()
        peaklist.loc[peak_label_ndx, ['rt_min', 'rt_max']] = rt_min, rt_max
        peaklist.to_csv( T.get_peaklist_fn( wdir ), index=False )   
        return f'Set rt_min, rt_max to {rt_min}, {rt_max} respectively.'


    def create_preview_peakshape():
        pass
