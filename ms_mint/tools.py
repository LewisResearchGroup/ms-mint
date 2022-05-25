import numpy as np
from molmass import Formula

from sklearn.preprocessing import StandardScaler, RobustScaler

from .standards import M_PROTON

from .filelock import FileLock


def lock(fn):
    """_summary_

    :param fn: _description_
    :type fn: function
    :return: _description_
    :rtype: _type_
    """
    return FileLock(f"{fn}.lock", timeout=1)


def get_mz_mean_from_formulas(formulas, ms_mode=None):
    """Calculate mz-mean vallue from formulas for specific ionization mode.

    :param formulas: List of molecular formulas e.g. ['H2O']
    :type formulas: list[str]
    :param ms_mode: Ionization mode, defaults to None
    :type ms_mode: str, optional
    :return: List of calculated masses.
    :rtype: list
    """
    masses = []
    for formula in formulas:
        try:
            mass = Formula(formula).isotope.mass
        except:
            masses.append(None)
            continue
        if ms_mode == "positive":
            mass += M_PROTON
        elif ms_mode == "negative":
            mass -= M_PROTON
        mass = np.round(mass, 4)
        masses.append(mass)
    return masses


def gaussian(x, mu, sig):
    """Simple gaussian function generator.

    :param x: _description_
    :type x: _type_
    :param mu: _description_
    :type mu: _type_
    :param sig: _description_
    :type sig: _type_
    :return: _description_
    :rtype: _type_
    """
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def scale_dataframe(df, scaler="standard", **kwargs):
    """Scale all columns in a dense dataframe.

    :param df: Dataframe to scale
    :type df: pandas.DataFrame
    :param scaler: Scaler to use ['robust', 'standard'], defaults to "standard"
    :type scaler: str, optional
    :return: Scaled dataframe.
    :rtype: pandas.DataFrame
    """
    df = df.copy()
    if scaler == "standard":
        scaler = StandardScaler(**kwargs)
    elif scaler == "robust":
        scaler = RobustScaler(**kwargs)
    df.loc[:, :] = scaler.fit_transform(df)
    return df
