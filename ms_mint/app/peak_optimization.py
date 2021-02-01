import os
import shutil

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

from dash.dependencies import Input, Output, State, ALL

import plotly.graph_objects as go

from ms_mint.peak_optimization.RetentionTimeOptimizer import RetentionTimeOptimizer as RTOpt

from . import tools as T


info_txt = '''
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
'''


_layout = html.Div([
    html.H3('Peak Optimization'),
    dcc.Dropdown(id='pko-ms-selection',
        options=[
            {'label': 'Use selected files from metadata table (PeakOpt)', 'value': 'peakopt'},
            {'label': 'Use all files (may take a long time)', 'value': 'all'}], 
        value='peakopt',
        clearable=False),
    html.Button('Update peak previews', id='pko-peak-preview'),
    html.Button('Regenerate all figures', id='pko-peak-preview-from-scratch'),
    html.Button('Find largest peaks for all peaks', 
        id='pko-find-largest-peak-for-all', style={'float': 'right', 'visibility': 'visible'}),
    dcc.Markdown('---'),
    html.Div(id='pko-peak-preview-output', 
        style={"maxHeight": "300px", "overflowY": "scroll", 'padding': 'auto'}),
    dcc.Markdown('---'),
    dcc.Markdown(id='pko-find-largest-peak-for-all-output'),
    dcc.Markdown(id='pko-find-largest-peak-output', 
            style={'visibility': 'hidden'}),

    html.Div(id='pko-controls'),
    dcc.Dropdown(
        id='pko-dropdown',
        options=[],
        value=None
    ),
    dcc.Loading( dcc.Graph('pko-figure') ),
    dcc.Checklist(id='pko-figure-options', 
                  options=[{'value': 'log','label': 'Logarithmic y-scale'}], 
                  value=[]),
    dcc.Markdown(id='pko-set-rt-output'),
    dcc.Markdown(id='pko-confirm-rt-output'),

    html.Button('Set RT to current view', id='pko-set-rt'),
    html.Button('Find largest peak', id='pko-find-largest-peak'),
    html.Button('Confirm retention time', id='pko-confirm-rt'),
    html.Button('Remove Peak', id='pko-delete', style={'float': 'right'}),

    html.Div([
        html.Button('<< Previous', id='pko-prev'), 
        html.Button('Suggest', id='pko-suggest-next'),
        html.Button('Next >>', id='pko-next')],
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
    State('wdir', 'children'),
    State('pko-dropdown', 'options'),
    )
    def pko_controls(tab, peak_deleted, wdir, old_options):
        if tab != 'pko':
            raise PreventUpdate
        peaklist = T.get_peaklist( wdir )
        if peaklist is None:
            raise PreventUpdate
        options = [{'label':label, 'value': i} for i, label in enumerate(peaklist.index)]
        if options == old_options:
            raise PreventUpdate
        return options


    @app.callback(
    Output('pko-figure', 'figure'),
    Input('pko-dropdown', 'value'),
    Input('pko-figure-options', 'value'),
    Input('pko-set-rt-output', 'children'),
    Input('pko-dropdown', 'options'),
    Input('pko-find-largest-peak-output', 'children'),
    Input('pko-confirm-rt-output', 'children'),
    State('pko-ms-selection', 'value'),
    State('wdir', 'children'),
    State('pko-figure', 'figure')
    )
    def pko_figure(peak_label_ndx, options, n_clicks, options_changed, 
                   find_largest_peak, rt_set, ms_selection, wdir, fig):
        if peak_label_ndx is None:
            raise PreventUpdate
        peaklist = T.get_peaklist( wdir ).reset_index()
        if ms_selection == 'peakopt':
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == 'all':
            ms_files = T.get_ms_fns(wdir)

        cols = ['mz_mean', 'mz_width', 'rt', 
                'rt_min', 'rt_max', 'peak_label']

        peak_label_ndx = peak_label_ndx % len(peaklist)
        mz_mean, mz_width, rt, rt_min, rt_max, label = \
            peaklist.loc[peak_label_ndx, cols]

        if (rt_min is None) or np.isnan(rt_min): rt_min = rt-0.2
        if (rt_max is None) or np.isnan(rt_max): rt_max = rt+0.2

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
            if 'log' in options: fig.update_yaxes(type="log")

        if not np.isnan(rt):
            fig.add_vline(rt)

        if (not np.isnan(rt_min)) and (not np.isnan(rt_max)):
            fig.add_vrect(x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1)

        n_files = len(ms_files)
        for i, fn in tqdm(enumerate(ms_files), total=n_files, desc='PKO-figure'):
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
    Output('pko-find-largest-peak-for-all-output', 'children'),
    Input('pko-find-largest-peak-for-all', 'n_clicks'),
    State('pko-ms-selection', 'value'),
    State('wdir', 'children'))
    def pko_find_closest_peak(n_clicks, ms_selection, wdir):
        if n_clicks is None:
            raise PreventUpdate
        peaklist = T.get_peaklist( wdir )

        if ms_selection == 'peakopt':
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == 'all':
            ms_files = T.get_ms_fns(wdir)

        n_peaks = len(peaklist)
        for i, (peak_label, row) in tqdm( enumerate(peaklist.iterrows()), total=n_peaks ):
            fsc.set('progress', int(100*(1+i)/n_peaks))
            mz_mean, mz_width = row.loc[['mz_mean', 'mz_width']]
            chromatograms = [T.get_chromatogram(fn, mz_mean, mz_width, wdir)\
                .set_index('retentionTime')['intensity array'] for fn in ms_files]
            rt_min, rt_max = None, None
            rt_min, rt_max = RTOpt().find_largest_peak(chromatograms)
            peaklist.loc[peak_label, ['rt_min', 'rt_max']] = rt_min, rt_max
        
        T.write_peaklist( peaklist, wdir)
        return 'Peak optimization done.'

    '''
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
    '''

    @app.callback(
        Output('pko-confirm-rt-output', 'children'),
        Input('pko-confirm-rt', 'n_clicks'),
        State('pko-dropdown', 'value'),
        State('pko-figure', 'figure'),
        State('wdir', 'children')
    )
    def pko_set_rt(n_clicks, peak_label, fig, wdir):
        if n_clicks is None:
            raise PreventUpdate
        rt_min, rt_max = fig['layout']['xaxis']['range']
        rt_min, rt_max = np.round(rt_min, 4), np.round(rt_max, 4)
        
        image_label = f'{peak_label}_{rt_min}_{rt_max}'

        _, fn = T.get_figure_fn(kind='peak-preview', wdir=wdir, 
            label=image_label, format='png')
        
        rt = np.mean([rt_min, rt_max])
        T.update_peaklist(wdir, peak_label, rt=rt)
        if os.path.isfile(fn): os.remove(fn)

        return f'Set RT span to ({rt_min},{rt_max})'


    @app.callback(
        Output('pko-dropdown', 'value'),
        Input('pko-prev', 'n_clicks'),
        Input('pko-suggest-next', 'n_clicks'),
        Input('pko-next', 'n_clicks'),
        Input('pko-image-clicked-output', 'children'),
        State('pko-dropdown', 'value'),
        State('pko-dropdown', 'options'),
        State('wdir', 'children')
    )
    def pko_prev_next_suggest(n_prev, n_suggest, n_next, image_clicked, 
            value, options, wdir):
        if n_prev is None\
            and n_next is None \
            and image_clicked is None\
            and n_suggest is None:
                raise PreventUpdate

        prop_id = dash.callback_context.triggered[0]['prop_id']        

        if prop_id.startswith('pko-suggest'):
            peaklist = T.get_peaklist( wdir ).reset_index()
            rt_means = peaklist[['rt_min', 'rt_max']].mean(axis=1)
            peak_label_ndx = np.argmax( (peaklist.rt-rt_means).abs() )
            return peak_label_ndx      

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


    #@app.callback(
    #    Output('pko-creating-chromatograms', 'children'),
    #    Input('tab', 'value'),
    #    State('wdir', 'children')
    #)
    def create_chromatograms(tab, wdir):
        if tab != 'pko':
            raise PreventUpdate
        ms_files = T.get_ms_fns_for_peakopt(wdir)
        peaklist = T.get_peaklist(wdir)
        T.Chromatograms(peaklist=peaklist, ms_files=ms_files, wdir=wdir).create_all()


    @app.callback(
    Output('pko-peak-preview-output', 'children'),
    Input('pko-peak-preview', 'n_clicks'),
    Input('pko-peak-preview-from-scratch', 'n_clicks'),
    Input('pko-find-largest-peak-output', 'children'),
    Input('pko-set-rt-output', 'children'),
    State('pko-ms-selection', 'value'),
    State('wdir', 'children')
    )
    def peak_preview(n_clicks, from_scratch, peak_opt, set_rt, ms_selection, wdir):
        prop_id = dash.callback_context.triggered[0]['prop_id']
        regenerate = prop_id.startswith('pko-peak-preview-from-scratch')
        if regenerate:
            print('Regenerate previews:', regenerate)
            image_path = os.path.join( wdir, 'figures', 'peak-preview')
            if os.path.isdir(image_path):
                shutil.rmtree(image_path)

        if ms_selection == 'peakopt':
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == 'all':
            ms_files = T.get_ms_fns(wdir)
        else:
            assert False, ms_selection

        if len(ms_files)==0:
            return 'No files selected for peak optimization.'
        else:
            print(f'Using {len(ms_files)} files for peak preview. ({ms_selection})')

        peaklist = T.get_peaklist(wdir)
        n_total = len(peaklist)
        
        sns.set_context('paper')
        images = []
        for i, (peak_label, row) in tqdm( enumerate( peaklist.iterrows() ), total=n_total):
            fsc.set('progress', int(100*(i+1) / n_total ))
            print(peak_label)
            mz_mean, mz_width, rt, rt_min, rt_max = \
                row[['mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']]

            if np.isnan(rt_min) or rt_min is None: rt_min = 0
            if np.isnan(rt_max) or rt_max is None: rt_max = 15

            image_label = f'{peak_label}_{rt_min}_{rt_max}'

            _, fn = T.get_figure_fn(kind='peak-preview', wdir=wdir, 
                label=image_label, format='png')

            if not os.path.isfile( fn ) or regenerate:
                create_preview_peakshape(ms_files, mz_mean, mz_width, rt, 
                    rt_min, rt_max, image_label, wdir, title=peak_label)

            if os.path.isfile(fn):
                src = T.png_fn_to_src(fn)
            else:
                src = None

            _id = {'index': peak_label, 'type': 'image'}
            image_id = f'image-{i}'
            images.append(
                html.A(id=_id, 
                children=html.Img(src=src, height=300, id=image_id, 
                    style={'margin': '10px'}))
            )
            images.append( 
                dbc.Tooltip(peak_label, target=image_id, style={'font-size': '50'})
            )
        return images


    @app.callback(
        Output('pko-image-clicked-output', 'children'),
        # Input needs brakets to make prevent_initital_call work
        [Input({'type': 'image', 'index': ALL}, 'n_clicks')],
        prevent_initial_call=True
        )
    def pko_image_clicked(ndx):
        if ndx is None or len(ndx)==0: raise PreventUpdate
        ctx = dash.callback_context
        clicked = ctx.triggered[0]['prop_id']
        clicked = clicked.replace('{"index":"', '')
        clicked = clicked.split('","type":')[0].replace('\\', '')
        if len( dash.callback_context.triggered) > 1: raise PreventUpdate
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
        peaklist = T.get_peaklist( wdir ).reset_index()
        peaklist = peaklist.drop( peak_label, axis=0 )
        T.write_peaklist(peaklist, wdir)
        return f'{peak_label} removed from peaklist.'  


    @app.callback(
        Output('pko-find-largest-peak-output', 'children'),
        Input('pko-find-largest-peak', 'n_clicks'),
        State('pko-dropdown', 'value'),
        State('pko-ms-selection', 'value'),
        State('wdir', 'children')
    )
    def find_largest_peak(n_clicks, peak_label_ndx, ms_selection, wdir):
        if n_clicks is None: raise PreventUpdate
        if peak_label_ndx is None: raise PreventUpdate
        peaklist = T.get_peaklist( wdir )
        if ms_selection == 'peakopt':
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == 'all':
            ms_files = T.get_ms_fns(wdir)           
        row = peaklist.iloc[peak_label_ndx]
        mz_mean, mz_width = row.loc[['mz_mean', 'mz_width']]
        chromatograms = [T.get_chromatogram(fn, mz_mean, mz_width, wdir)\
            .set_index('retentionTime')['intensity array'] for fn in ms_files]
        rt_min, rt_max = None, None
        rt_min, rt_max = RTOpt().find_largest_peak(chromatograms)
        peaklist = peaklist.reset_index()
        peaklist.loc[peak_label_ndx, ['rt_min', 'rt_max']] = rt_min, rt_max
        peaklist.to_csv( T.get_peaklist_fn( wdir ), index=False )   
        return f'Set rt_min, rt_max to {rt_min}, {rt_max} respectively.'



def create_preview_peakshape(ms_files, mz_mean, mz_width, rt, 
        rt_min, rt_max, image_label, wdir, title):
    print('create image')
    #plt.close()
    plt.figure(figsize=(4,2.5), dpi=30)
    y_max = 0 
    for fn in ms_files:
        #try:
        fn_chro = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
        fn_chro = fn_chro[(rt_min < fn_chro['retentionTime']) &
                             (fn_chro['retentionTime'] < rt_max)   ]
        plt.plot(fn_chro['retentionTime'], fn_chro['intensity array'], lw=1, color='k')
        y_max = max(y_max, fn_chro['intensity array'].max())
        #except:
        #    pass
    print(rt, rt_min, rt_max)    
    if (not np.isnan(rt)) \
        and not (np.isnan(rt_max)) \
        and not (np.isnan(rt_min)):

        x = max(min(rt, rt_max), rt_min)
        rt_mean = np.mean([rt_min, rt_max])
        color_value = np.abs(rt_mean-rt)
        color = T.float_to_color(color_value, vmin=0, vmax=1, cmap='coolwarm')
        plt.vlines(x, 0, y_max, lw=3, color=color)
    plt.gca().set_title(title, y=1.0, pad=15)
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel('Retention Time [min]')  
    plt.ylabel('MS-Intensity')
    filename = T.savefig(kind='peak-preview', wdir=wdir, label=image_label)
    plt.close()
    print('done')
    return filename 
