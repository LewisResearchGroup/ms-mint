import os
import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from dash.exceptions import PreventUpdate
from dash_tabulator import DashTabulator

from ms_mint import MINT_DATA_PATH
from ms_mint.standards import M_PROTON
from . import tools as T


options = {
           "selectable": True,
           "headerFilterLiveFilterDelay":3000,
           "layout": "fitDataFill",
           "height": "600px",
           }

clearFilterButtonType = {"css": "btn btn-outline-dark", "text":"Clear Filters"}

fn_data = os.path.join(MINT_DATA_PATH, 'ChEBI.tsv')
chcbi_data = pd.read_csv(fn_data, sep='\t', nrows=None, index_col=0, low_memory=False)
chcbi_data = chcbi_data[chcbi_data['Charge'].isin([-1,0,1])]
chcbi_data = chcbi_data[chcbi_data['Monoisotopic Mass'].notna()]
chcbi_data = chcbi_data[chcbi_data['Monoisotopic Mass']<1000]
chcbi_data = chcbi_data[chcbi_data['Monoisotopic Mass']>50]
# Remove entries with mutiple molecules
# keep compounds without SMILES
chcbi_data = chcbi_data[~chcbi_data.SMILES.str.contains('\.').fillna(False)]

chcbi_data['Monoisotopic Mass'] = chcbi_data['Monoisotopic Mass'].astype(float)

#print('Available columns:')
#for col in chcbi_data: print(col)

chcbi_data = chcbi_data[['ChEBI ID', 'ChEBI Name', 'Formulae', 'Charge', 'Monoisotopic Mass', 'Synonyms', 'KEGG COMPOUND Database Links', 'SMILES']]


columns = T.gen_tabulator_columns(chcbi_data.columns, editor=None, col_width='auto')


add_metab_table = html.Div(id='add-metab-table-container', 
    style={'min-height':  100, 'margin-top': '10%'},
    children=[
        dcc.Dropdown(id='add-metab-ms-mode', placeholder='Select ionization mode',
            options=[{'label': 'Positive', 'value': 'Positive'},
                     {'label': 'Negative', 'value': 'Negative'}], 
            value='Negative'),
        DashTabulator(id='add-metab-table',
            columns=columns, 
            options=options,
            clearFilterButtonType=clearFilterButtonType,
            data=chcbi_data.to_dict('records')
        ),
    html.Button('Add selected metabolites to peaklist', id='add-metab'),
    html.Div(id='add-metab-output'),

])


_layout = html.Div([
    html.H3('Add Metabolites'),
    add_metab_table
    
])

def layout():
    return _layout


def callbacks(app, fsc, cache):

    @app.callback(
        Output('add-metab-output', 'children'),
        Input( 'add-metab', 'n_clicks'),
        State( 'add-metab-table', 'multiRowsClicked'),
        State( 'add-metab-ms-mode', 'value'),
        State( 'wdir', 'children')
    )
    def add_metab(n_clicks, rows, ms_mode, wdir):
        if n_clicks is None:
            raise PreventUpdate
        if ms_mode is None:
            return 'Please select ionization mode.'
        peaklist = T.get_peaklist( wdir )
        for row in rows:
            charge = int( row['Charge'] )
            if (ms_mode=='Negative') and (charge==1):
                continue
            if (ms_mode=='Positive') and (charge==-1):
                continue
            print(charge)
            if charge == 0:
                if ms_mode == 'Negative':         
                    peaklist.loc[row['ChEBI Name'], 'mz_mean'] = row['Monoisotopic Mass'] - M_PROTON
                elif ms_mode == 'Positive':
                    peaklist.loc[row['ChEBI Name'], 'mz_mean'] = row['Monoisotopic Mass'] + M_PROTON
            else:
                peaklist.loc[row['ChEBI Name'], 'mz_mean'] = row['Monoisotopic Mass']

            peaklist.loc[row['ChEBI Name'], 'mz_width'] = 10
            peaklist.loc[row['ChEBI Name'], 'rt'] = -1
            peaklist.loc[row['ChEBI Name'], 'rt_min'] = 0
            peaklist.loc[row['ChEBI Name'], 'rt_max'] = 15
            peaklist.loc[row['ChEBI Name'], 'intensity_threshold'] = 0

        T.write_peaklist( peaklist, wdir)
        
        return 'Metabolites added.'

        