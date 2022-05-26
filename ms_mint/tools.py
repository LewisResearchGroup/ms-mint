import os

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
        #try:
        mass = Formula(formula).isotope.mass
        #except:
        #masses.append(None)
        # continue
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


def df_diff(df1, df2, which="both"):
    """_summary_

    :param df1: _description_
    :type df1: _type_
    :param df2: _description_
    :type df2: _type_
    :param which: _description_, defaults to "both"
    :type which: str, optional
    :return: _description_
    :rtype: _type_
    """
    _df = df1.merge(df2, indicator=True, how="outer")
    diff_df = _df[_df["_merge"] != which]
    return diff_df.reset_index(drop=True)


def is_ms_file(fn):
    """_summary_

    :param fn: _description_
    :type fn: function
    :return: _description_
    :rtype: _type_
    """
    if (
        (fn.lower().endswith(".mzxml"))
        or (fn.lower().endswith(".mzml"))
        or (fn.lower().endswith(".mzmlb"))
        or (fn.lower().endswith(".mzhdf"))
        or (fn.lower().endswith(".raw"))
        or (fn.lower().endswith(".parquet"))
        or (fn.lower().endswith(".feather"))
    ):
        return True
    else:
        return False


def get_ms_files_from_results(results):
    """_summary_

    :param results: _description_
    :type results: _type_
    :return: _description_
    :rtype: _type_
    """
    ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files
