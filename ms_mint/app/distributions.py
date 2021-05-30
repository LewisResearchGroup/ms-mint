from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns  

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from . import tools as T

graph_options = [{'label': 'Histograms', 'value': 'hist'},
                 {'label': 'Boxplots', 'value': 'boxplot'},
                 {'label': 'Probability Density', 'value': 'density'}]

_label = 'Distributions'

_layout = html.Div([
    html.H3('Quality Control'),
    html.Button('Update', id='dist-update'),
    dcc.Dropdown(id='dist-graphs', options=graph_options, value=['hist'], multi=True, placeholder='Kinds of graphs'),
    dcc.Checklist(id='dist-select', options=[{'label': 'Dense', 'value': 'Dense'}], value=['Dense']),
    html.Div(id='dist-figures', style={'float': 'center'})
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
    Output('dist-figures', 'children'),
    Input('dist-update', 'n_clicks'),
    State('tab', 'value'),
    State('ana-groupby', 'value'),
    State('dist-graphs', 'value'),
    State('dist-select', 'value'),
    State('ana-file-types', 'value'),
    State('ana-peak-labels-include', 'value'),
    State('ana-peak-labels-exclude', 'value'),
    State('wdir', 'children')
    )
    def qc_figures(n_clicks, tab, groupby, kinds, options, file_types, 
            include_labels, exclude_labels, wdir):

        if n_clicks is None:
            raise PreventUpdate

        df = T.get_complete_results( wdir, include_labels=include_labels, 
                exclude_labels=exclude_labels, file_types=file_types )

        if len(df) == 0: return 'No results yet. First run MINT.'

        if 'boxplot' in kinds:
            if (groupby is None or len(groupby) ==0):
                return dbc.Alert('For boxplots a "Group-by" column has to be set', color='info')
            if len(df[groupby].drop_duplicates()) <= 1:
                return dbc.Alert('For boxplots at least two groups have to be defined in selected "Group-by" column in metadata sheet.', color='info')

        sns.set_context('paper')
        
        sort_by_col = 'Batch'
        quant_col = 'log(peak_max+1)'

        if sort_by_col is not None:
            df = df.sort_values(['peak_label', sort_by_col])

        if options is None:
            options = []
        
        figures = []
        n_total = len(df.peak_label.drop_duplicates())
        for i, (peak_label, grp) in tqdm( enumerate(df.groupby('peak_label')), total=n_total ):

            if not 'Dense' in options: figures.append(dcc.Markdown(f'#### `{peak_label}`', style={'float': 'center'}))
            fsc.set('progress', int(100*(i+1)/n_total))

            # Sorting to ensure similar legends
            if sort_by_col is not None:
                grp = grp.sort_values(sort_by_col).reset_index(drop=True)

            #if len(grp) < 1:
            #    continue

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
