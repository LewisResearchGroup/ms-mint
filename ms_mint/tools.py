import numpy as np
from molmass import Formula

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