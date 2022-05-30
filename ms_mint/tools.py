import os

import numpy as np

import pandas as pd

from molmass import Formula

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler

from scipy.signal import find_peaks, peak_widths

from .standards import M_PROTON

from .filelock import FileLock


def lock(fn):
    """
    File lock to ensure safe writing to file.

    :param fn: Filename to lock.
    :type fn: str or PosixPath
    :return: File lock object.
    :rtype: FileLock
    """
    return FileLock(f"{fn}.lock", timeout=1)


def get_mz_mean_from_formulas(formulas, ms_mode=None):
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
    for formula in formulas:
        # try:
        mass = Formula(formula).isotope.mass
        # except:
        # masses.append(None)
        # continue
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
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files


def find_peaks_in_timeseries(series, prominence=None, plot=False):
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
    widths, heights, left_ips, right_ips = peak_widths(x, peak_ndxs, rel_height=0.9)  
    times = series.iloc[peak_ndxs].index
    
    t_start = _map_ndxs_to_time(left_ips, min(t), max(t), 0, len(t))
    t_end = _map_ndxs_to_time(right_ips, min(t), max(t), 0, len(t))
    
    data = dict(
        ndxs=peak_ndxs,
        rt=times,
        rt_span=widths,
        peak_height=heights,
        rt_min=t_start,
        rt_max=t_end
    )
    
    peaks = pd.DataFrame(data) 
    
    if plot:
        _plot_peaks(series, peaks)
    
    return peaks
        

def _plot_peaks(series, peaks, highlight=None):
    if highlight is None: 
        highlight = []
    series.plot()
    if peaks is not None:
        series.iloc[peaks.ndxs].plot(label='Peaks', marker='x', y='intensity', lw=0, ax=plt.gca())
        for ndx, (_, rt, rt_span, peak_height, rt_min, rt_max) in peaks.iterrows():
            if ndx in highlight:
                plt.axvspan(rt_min, rt_max, color='lightgreen', alpha=0.25)
            else:
                color = 'orange'
            plt.hlines(peak_height, rt_min, rt_max, color=color)    


def _map_ndxs_to_time(x, t_min, t_max, x_min, x_max):
    assert t_min < t_max
    assert x_min < x_max
    t_span = t_max - t_min
    x_span = x_max - x_min
    m = (t_span/x_span)
    b = t_min
    x = np.array(x)
    result = ( m * x + b ).flatten()   
    return result   