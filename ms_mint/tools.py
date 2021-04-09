import pandas as pd
import numpy as np
from molmass import Formula

from sklearn.preprocessing import StandardScaler, RobustScaler

from .standards import M_PROTON


def get_mz_mean_from_formulas(formulas, ms_mode=None, verbose=False):
    if verbose: print(formulas)
    masses = []
    for formula in formulas:
        if verbose: print(f'Mass from formula: "{formula}"')
        try:
            mass = Formula(formula).isotope.mass  
        except:
            masses.append(None)
            continue
        if ms_mode == 'positive':
            mass += M_PROTON
        elif ms_mode == 'negative':
            mass -= M_PROTON
        mass = np.round(mass, 4)
        masses.append(mass)
    if verbose: print(masses)
    return masses


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def scale_dataframe(df, scaler='standard'):
    df = df.copy()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), 
                        columns=df.columns, index=df.index)