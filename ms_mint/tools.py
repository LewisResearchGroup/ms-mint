import os

import numpy as np

from molmass import Formula

from sklearn.preprocessing import StandardScaler, RobustScaler

from .standards import M_PROTON

from .filelock import FileLock


def lock(fn):
    """File lock to ensure safe writing to file.

    :param fn: Filename to lock.
    :type fn: str or PosixPath
    :return: File lock object.
    :rtype: FileLock
    """
    return FileLock(f"{fn}.lock", timeout=1)


def get_mz_mean_from_formulas(formulas, ms_mode=None):
    """Calculate mz-mean vallue from formulas for specific ionization mode.

    :param formulas: List of molecular formulas e.g. ['H2O']
    :type formulas: list[str]
    :param ms_mode: Ionization mode, defaults to None
    :type ms_mode: str, optional
    :return: List of calculated masses
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
    """Scale all columns in a dense dataframe.

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
    """Difference between two dataframes.

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
    """Check if file is a MS-file based on filename.

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
    """Extract MS-filenames from Mint results.

    :param results: DataFrame in Mint fesults format
    :type results: pandas.DataFrame
    :return: List of filenames
    :rtype: list
    """
    ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files
