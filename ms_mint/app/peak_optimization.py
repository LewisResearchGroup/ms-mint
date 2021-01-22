import os

from glob import glob
from tqdm import tqdm

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns  

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State

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
    html.Button('Find closest peaks', id='pko-find-closest-peak'),
    dcc.Markdown('---'),
    html.Div(id='pko-peak-preview-output'),
    dcc.Markdown(id='pko-find-closest-peak-output'),

    html.Div(id='pko-controls'),
    dcc.Dropdown(
        id='pko-dropdown',
        options=[],
        value=None
    ),
    dcc.Loading( dcc.Graph('pko-figure') ),
    dcc.Markdown(id='pko-set-rt-output'),
    html.Button('Set RT to current view', id='pko-set-rt'),
    html.Button('Remove Peak', id='pko-delete', style={'float': 'right'}),

    html.Div([
        html.Button('<< Previous', id='pko-prev'),
        html.Button('Next >>', id='pko-next')],
        style={'text-align': 'center', 'margin': 'auto'}),
    
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
    State('wdir', 'children'),
    State('pko-figure', 'figure')
    )
    def pko_figure(peak_label_ndx, n_clicks, options_changed, wdir, fig):
        if peak_label_ndx is None:
            raise PreventUpdate
        
        peaklist = T.get_peaklist( wdir ).reset_index()
        ms_files = T.get_ms_fns( wdir )

        ms_files = ms_files[:100]

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
        mint.ms_files = glob( os.path.join( wdir, 'ms_files', '*.*'))
        rtopt = RTOpt(mint)
        new_peaklist = rtopt.fit_transform(margin=1, how='max')
        new_peaklist.to_csv(fn_pkl)


    @app.callback(
    Output('pko-set-rt-output', 'children'),
    Input('pko-set-rt', 'n_clicks'),
    [State('pko-dropdown', 'value'),
    State('pko-figure', 'figure'),
    State('wdir', 'children')]
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
    State('pko-dropdown', 'value'),
    State('pko-dropdown', 'options')
    )
    def pko_prev_next(n_prev, n_next, value, options):
        if n_prev is None and n_next is None:
            raise PreventUpdate
        prop_id = dash.callback_context.triggered[0]['prop_id']
        if value is None:
            return 0
        if prop_id.startswith('pko-prev'):
            return (value - 1) % len(options)
        if prop_id.startswith('pko-next'):
            return (value + 1) % len(options)

    TIMEOUT = 60
    @app.callback(
    Output('pko-peak-preview-output', 'children'),
    Input('pko-peak-preview', 'n_clicks'),
    State('wdir', 'children')
    )
    @cache.memoize(timeout=TIMEOUT)
    def peak_preview(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate 
        sns.set_context('paper')
        ms_files = T.get_ms_fns(wdir)
        ms_files = ms_files[:100]
        peaklist = T.get_peaklist(wdir)
        n_total = len(peaklist)
        images = []
        for i, (ndx, row) in tqdm( enumerate(peaklist.iterrows()), total=n_total ):
            fsc.set('progress', int(100*(i+1)/n_total))
            mz_mean, mz_width, rt, rt_min, rt_max = row[['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']]

            if np.isnan(rt_min) or rt_min is None: rt_min = rt-0.5
            if np.isnan(rt_max) or rt_max is None: rt_max = rt+0.5

            plt.figure(figsize=(3.5,2.5), dpi=50)
            for fn in ms_files:
                try:        
                    fn_chro = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
                    fn_chro = fn_chro[(rt_min < fn_chro['retentionTime']) &
                                    (fn_chro['retentionTime'] < rt_max)]
                    plt.plot(fn_chro['retentionTime'], fn_chro['intensity array'], lw=1, color='k')
                except:
                    pass
            plt.title(ndx)
            plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))        
            src = T.fig_to_src()
            _id = f'peak-preview-{i}'
            images.append(html.Img(id=_id, src=src))
            images.append( dbc.Tooltip(ndx, target=_id, style={'font-size=': 'large'}) )
        images.append( dcc.Markdown('---') )
        return images


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