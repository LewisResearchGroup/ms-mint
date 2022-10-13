import logging

import numpy as np
import pandas as pd

from pathlib import Path as P
from molmass import Formula, FormulaError
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import find_peaks, peak_widths

from .standards import M_PROTON, TARGETS_COLUMNS
from .filelock import FileLock
from .matplotlib_tools import plot_peaks


def lock(fn):
    """
    File lock to ensure safe writing to file.

    :param fn: Filename to lock.
    :type fn: str or PosixPath
    :return: File lock object.
    :rtype: FileLock
    """
    return FileLock(f"{fn}.lock", timeout=1)


def formula_to_mass(formulas, ms_mode=None):
    """
    Calculate mz-mean vallue from formulas for specific ionization mode.

    :param formulas: List of molecular formulas e.g. ['H2O']
    :type formulas: list[str]
    :param ms_mode: Ionization mode, defaults to None
    :type ms_mode: str, optional
    :return: List of calculated masses
    :rtype: list
    """
    masses = []
    assert ms_mode in [None, "negative", "positive", "neutral"], ms_mode
    if isinstance(formulas, str):
        formulas = [formulas]
    for formula in formulas:
        try:
            mass = Formula(formula).isotope.mass
        except FormulaError as e:
            masses.append(None)
            logging.waringin(e)
        if ms_mode == "positive":
            mass += M_PROTON
        elif ms_mode == "negative":
            mass -= M_PROTON
        mass = np.round(mass, 4)
        masses.append(mass)
    return masses


def gaussian(x, mu, sig):
    """
    Simple gaussian function generator.

    :param x: x-values to generate function values
    :type x: np.array
    :param mu: Mean of gaussian
    :type mu: float
    :param sig: Sigma of gaussian
    :type sig: float
    :return: f(x)
    :rtype: np.array
    """
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def scale_dataframe(df, scaler="standard", **kwargs):
    """
    Scale all columns in a dense dataframe.

    :param df: Dataframe to scale
    :type df: pandas.DataFrame
    :param scaler: Scaler to use ['robust', 'standard'], defaults to "standard"
    :type scaler: str, optional
    :return: Scaled dataframe
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
    """
    Difference between two dataframes.

    :param df1: Reference dataframe
    :type df1: pandas.DataFrame
    :param df2: Dataframe to compare
    :type df2: pandas.DataFrame
    :param which: Direction in which to compare, defaults to "both"
    :type which: str, optional
    :return: DataFrame that contains unique rows.
    :rtype: pandas.DataFrame
    """
    _df = df1.merge(df2, indicator=True, how="outer")
    diff_df = _df[_df["_merge"] != which]
    return diff_df.reset_index(drop=True)


def is_ms_file(fn):
    """
    Check if file is a MS-file based on filename.

    :param fn: Filename
    :type fn: str or PosixPath
    :return: Whether or not the file is recognized as MS-file
    :rtype: bool
    """
    fn = str(fn)
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
    """
    Extract MS-filenames from Mint results.

    :param results: DataFrame in Mint fesults format
    :type results: pandas.DataFrame
    :return: List of filenames
    :rtype: list
    """
    ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
    ms_files = [P(ms_path)/ms_file for ms_path, ms_file in ms_files.values]
    return ms_files

def get_targets_from_results(results):
    """Extract targets dataframe from ms-mint results table.

    :param results: Mint results table
    :type results: pandas.DataFrame
    :return: Mint targets table
    :rtype: pandas.DataFrame
    """
    return results[
            [col for col in TARGETS_COLUMNS if col in results.columns]
        ].drop_duplicates()

def find_peaks_in_timeseries(series, prominence=None, plot=False, rel_height=0.9):
    """_summary_

    :param series: _description_
    :type series: _type_
    :param prominence: _description_, defaults to None
    :type prominence: _type_, optional
    :param plot: _description_, defaults to False
    :type plot: bool, optional
    :return: _description_
    :rtype: _type_
    """
    t = series.index
    x = series.values
    peak_ndxs, _ = find_peaks(x, prominence=prominence)
    widths, heights, left_ips, right_ips = peak_widths(x, peak_ndxs, rel_height=rel_height)
    times = series.iloc[peak_ndxs].index

    t_start = _map_ndxs_to_time(left_ips, min(t), max(t), 0, len(t))
    t_end = _map_ndxs_to_time(right_ips, min(t), max(t), 0, len(t))

    data = dict(
        ndxs=peak_ndxs,
        rt=times,
        rt_span=widths,
        peak_base_height=heights,
        peak_height=series.iloc[peak_ndxs].values,
        rt_min=t_start,
        rt_max=t_end,
    )

    peaks = pd.DataFrame(data)

    if plot:
        plot_peaks(series, peaks)

    return peaks


def _map_ndxs_to_time(x, t_min, t_max, x_min, x_max):
    assert t_min < t_max
    assert x_min < x_max
    t_span = t_max - t_min
    x_span = x_max - x_min
    m = t_span / x_span
    b = t_min
    x = np.array(x)
    result = (m * x + b).flatten()
    return result
