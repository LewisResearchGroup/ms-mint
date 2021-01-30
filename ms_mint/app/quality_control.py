from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns  

import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from . import tools as T

groupby_options = [{'label': 'Batch', 'value': 'Batch'},
                   {'label': 'Label', 'value': 'Label'},
                   {'label': 'Type',  'value': 'Type'}]

graph_options = [{'label': 'Histograms', 'value': 'hist'},
                 {'label': 'Boxplots', 'value': 'boxplot'},
                 {'label': 'Probability Density', 'value': 'density'}]


_layout = html.Div([
    html.H3('Quality Control'),
    html.Button('Update', id='qc-update'),
    dcc.Dropdown(id='qc-groupby', options=groupby_options, value=None, placeholder='Group by column'),
    dcc.Dropdown(id='qc-graphs', options=graph_options, value=['hist', 'boxplot', 'density'], multi=True, placeholder='Kinds of graphs'),
    dcc.Dropdown(id='file-types', options=[], placeholder='Types of files to include', multi=True),
    dcc.Dropdown(id='peak-labels', options=[], placeholder='Limit to peak_labels', multi=True),
    dcc.Checklist(id='qc-select', options=[{'label': 'Dense', 'value': 'Dense'}], value=['Dense']),
    html.Div(id='qc-figures', style={'float': 'center'})
])


layout_no_data = html.Div([
    dcc.Markdown('''### No results generated yet. 
    MINT has not been run yet. The Quality Control tabs uses the processed data. 
    To generate it please add MS-files as well as a valid peaklist. 
    Then execute MINT data processing routine witby clicking the `RUN MINT` button. 
    Once results have been produced you can access the QC tools.'''),
])


def layout():
    return _layout
    
def callbacks(app, fsc, cache):

    @app.callback(
    Output('qc-figures', 'children'),
    Input('qc-update', 'n_clicks'),
    State('tab', 'value'),
    State('qc-groupby', 'value'),
    State('qc-graphs', 'value'),
    State('qc-select', 'value'),
    State('file-types', 'value'),
    State('peak-labels', 'value'),
    State('wdir', 'children')
    )
    def qc_figures(n_clicks, tab, groupby, kinds, options, file_types, peak_labels, wdir):
        if n_clicks is None:
            raise PreventUpdate

        df = T.get_complete_results( wdir )

        if len(df) == 0: return 'No results yet. First run MINT.'

        print('Results', df)

        if file_types is not None and len(file_types) > 0:
            df = df[df.Type.isin(file_types)]

        if peak_labels is not None and len(peak_labels) > 0:
            df = df[df.peak_label.isin(peak_labels)]

        sns.set_context('paper')
        
        by_col = 'Label'
        by_col = 'Batch'
        quant_col = 'peak_max'
        quant_col = 'log(peak_max+1)'

        if by_col is not None:
            df = df.sort_values(['peak_label', by_col])

        if options is None:
            options = []

        figures = []
        n_total = len(df.peak_label.drop_duplicates())
        for i, (peak_label, grp) in tqdm( enumerate(df.groupby('peak_label')), total=n_total ):

            if not 'Dense' in options: figures.append(dcc.Markdown(f'#### `{peak_label}`', style={'float': 'center'}))
            fsc.set('progress', int(100*(i+1)/n_total))

            # Sorting to ensure similar legends
            if by_col is not None:
                grp = grp.sort_values(by_col).reset_index(drop=True)

            if len(grp) < 1:
                continue

            if 'hist' in kinds: 
                sns.displot(data=grp, x=quant_col, height=3, hue=groupby, aspect=1)
                plt.title(peak_label)
                fig_label = f'by-{groupby}__{quant_col}__{peak_label}'
                T.savefig(kind='hist', wdir=wdir, label=fig_label)
                src = T.fig_to_src(dpi=150)
                figures.append( html.Img(src=src, style={'width': '300px'}) )

            if 'density' in kinds:
                sns.displot(data=grp, x=quant_col, hue=groupby, kind='kde', 
                            common_norm=False, height=3,  aspect=1)
                plt.title(peak_label)
                fig_label = f'by-{groupby}__{quant_col}__{peak_label}'
                T.savefig(kind='density', wdir=wdir, label=fig_label)                
                src = T.fig_to_src(dpi=150)
                figures.append( html.Img(src=src, style={'width': '300px'}) )

            if 'boxplot' in kinds:
                n_groups = len( grp[groupby].drop_duplicates() )
                aspect = max(1, n_groups/10)
                sns.catplot(data=grp, y=quant_col, x=groupby, height=3, kind='box', aspect=aspect, color='w')
                if quant_col in ['peak_max', 'peak_area']:
                    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.title(peak_label)
                plt.xticks(rotation=90)
                fig_label = f'by-{groupby}__{quant_col}__{peak_label}'
                T.savefig(kind='boxplot', wdir=wdir, label=fig_label)                   
                src = T.fig_to_src(dpi=150)
                figures.append( html.Img(src=src, style={'width': '300px'}) )

            if not 'Dense' in options: figures.append(dcc.Markdown('---'))

            #if i == 3: break

        return figures


    @app.callback(
    Output('file-types', 'options'),
    Output('file-types', 'value'),
    Input('tab', 'value'),
    State('wdir', 'children')
    )
    def file_types(tab, wdir):
        if not tab in ['qc', 'heatmap']:
            raise PreventUpdate
        meta = T.get_metadata( wdir )
        if meta is None:
            raise PreventUpdate
        file_types = meta['Type'].drop_duplicates()
        options = [{'value': i, 'label': i} for i in file_types]
        return options, file_types