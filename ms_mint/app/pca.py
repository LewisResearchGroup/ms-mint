
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..Mint import Mint

from . import tools as T

scaling_options = [{'value': i, 'label': i} for i in ['Standard']]

_layout = html.Div([
    html.H3('Decomposition'),
    html.Button('Run PCA',id='pca-update'),
    dcc.Dropdown(id='dec-scaling', options=scaling_options, 
        value=['Standard'], multi=True, placeholder='Scaling used before PCA'),
    html.Label('Number of PCA components'),

    dcc.Slider(id='pca-nvars', value=3, min=2, max=10, marks={i: f'{i}' for i in range(2, 11)}),
    

    dcc.Loading( html.Div(id='pca-figures', style={'margin': 'auto', 'text-align': 'center'}) ) 
    
])

_label = 'PCA'

def layout():
    return _layout


def callbacks(app, fsc, cache):
    
    @app.callback(
        Output('pca-figures', 'children'),
        Input('pca-update', 'n_clicks'),
        State('pca-nvars', 'value'),
        State('ana-groupby', 'value'),
        State('ana-peak-labels-include', 'value'),
        State('ana-peak-labels-exclude', 'value'),
        State('ana-file-types', 'value'),
        State('wdir', 'children')
    )
    def create_pca( n_clicks, n_vars, groupby, include_labels, exclude_labels, 
            file_types, wdir ):
        if n_clicks is None:
            raise PreventUpdate

        df = T.get_complete_results( wdir, include_labels=include_labels, 
            exclude_labels=exclude_labels, file_types=file_types )

        if file_types is not None and len(file_types) > 0:
            df = df[df['Type'].isin(file_types)]
        if groupby is not None and len(groupby) > 0:
            color_groups = df[['ms_file', groupby]].drop_duplicates().set_index('ms_file')
        else:
            color_groups = None
            groupby = None

        figures = []
        mint = Mint()
        mint.results = df
        mint.pca()

        ndx = mint.decomposition_results['df_projected'].index.to_list()

        mint.pca_plot_cumulative_variance()

        src = T.fig_to_src()
        figures.append( html.Img(src=src) )

        if color_groups is not None:
            color_groups = color_groups.loc[ndx].values

        mint.plot_pair_plot(group_name=groupby, color_groups=color_groups, n_vars=n_vars)

        src = T.fig_to_src()
        figures.append( html.Img(src=src) )

        return figures