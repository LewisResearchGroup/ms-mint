import logging
import numpy as np
import pandas as pd
from pathlib import Path as P
from molmass import Formula, FormulaError
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import find_peaks, peak_widths
from typing import Union, List, Tuple, Optional, Dict, Any, Callable, Literal

from .standards import M_PROTON, TARGETS_COLUMNS, MINT_METADATA_COLUMNS
from .filelock import FileLock
from .matplotlib_tools import plot_peaks


def log2p1(x: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Apply log2(x+1) transformation to numeric data.

    Args:
        x: Numeric value or array to transform.

    Returns:
        Transformed value(s).
    """
    return np.log2(x + 1)


def lock(fn: Union[str, P]) -> FileLock:
    """Create a file lock to ensure safe writing to file.

    Args:
        fn: Filename to lock.

    Returns:
        File lock object.
    """
    return FileLock(f"{fn}.lock", timeout=1)


def formula_to_mass(
    formulas: Union[str, List[str]],
    ms_mode: Optional[Literal["negative", "positive", "neutral"]] = None,
) -> List[Optional[float]]:
    """Calculate m/z values from molecular formulas for specific ionization mode.

    Args:
        formulas: List of molecular formulas (e.g., ['H2O']) or a single formula.
        ms_mode: Ionization mode. One of "negative", "positive", "neutral", or None.

    Returns:
        List of calculated masses. None values are included for invalid formulas.

    Raises:
        AssertionError: If ms_mode is not one of the allowed values.
    """
    masses = []
    assert ms_mode in [None, "negative", "positive", "neutral"], ms_mode
    if isinstance(formulas, str):
        formulas = [formulas]
    for formula in formulas:
        try:
            mass = Formula(formula).isotope.mass
            if ms_mode == "positive":
                mass += M_PROTON
            elif ms_mode == "negative":
                mass -= M_PROTON
            mass = np.round(mass, 4)
            masses.append(mass)
        except FormulaError as e:
            masses.append(None)
            logging.warning(e)  # Fixed typo: waringin → warning
    return masses


def gaussian(x: Union[List[float], np.ndarray], mu: float, sig: float) -> np.ndarray:
    """Generate values for a Gaussian function.

    Args:
        x: x-values to generate function values.
        mu: Mean of the Gaussian.
        sig: Standard deviation of the Gaussian.

    Returns:
        Array of Gaussian function values at the input x-values.
    """
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def scale_dataframe(
    df: pd.DataFrame, scaler: Union[str, Any] = "standard", **kwargs
) -> pd.DataFrame:
    """Scale all columns in a dense dataframe.

    Args:
        df: DataFrame to scale.
        scaler: Scaler to use. Either a string ('robust', 'standard', 'minmax')
            or a scikit-learn scaler instance.
        **kwargs: Additional arguments passed to the scaler constructor.

    Returns:
        Scaled DataFrame with the same shape as the input.
    """
    df = df.copy()
    if isinstance(scaler, str):
        if scaler == "standard":
            scaler = StandardScaler(**kwargs)
        elif scaler == "robust":
            scaler = RobustScaler(**kwargs)
        elif scaler == "minmax":
            scaler = MinMaxScaler(**kwargs)
    df.loc[:, :] = scaler.fit_transform(df)
    return df


def df_diff(df1: pd.DataFrame, df2: pd.DataFrame, which: str = "both") -> pd.DataFrame:
    """Find differences between two dataframes.

    Args:
        df1: Reference DataFrame.
        df2: DataFrame to compare.
        which: Direction in which to compare. Options are "both", "left_only", "right_only".

    Returns:
        DataFrame containing only the rows that differ according to the specified direction.
    """
    _df = df1.merge(df2, indicator=True, how="outer")
    diff_df = _df[_df["_merge"] != which]
    return diff_df.reset_index(drop=True)


def is_ms_file(fn: Union[str, P]) -> bool:
    """Check if a file is a recognized MS file format based on its extension.

    Args:
        fn: Filename or path to check.

    Returns:
        True if the file has a recognized MS file extension, False otherwise.
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


def get_ms_files_from_results(results: pd.DataFrame) -> List[Union[str, P]]:
    """Extract MS filenames from Mint results.

    Args:
        results: DataFrame in Mint results format.

    Returns:
        List of MS filenames.
    """
    # Old schema
    if "ms_path" in results.columns:
        ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
        ms_files = [P(ms_path) / ms_file for ms_path, ms_file in ms_files.values]
    else:
        ms_files = results.ms_file.unique()
    return ms_files


def get_targets_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Extract targets DataFrame from MS-MINT results table.

    Args:
        results: Mint results table.

    Returns:
        DataFrame containing target information extracted from results.
    """
    return results[[col for col in TARGETS_COLUMNS if col in results.columns]].drop_duplicates()


def find_peaks_in_timeseries(
    series: pd.Series,
    prominence: Optional[float] = None,
    plot: bool = False,
    rel_height: float = 0.9,
    **kwargs,
) -> pd.DataFrame:
    """Find peaks in a time series using scipy's peak finding algorithm.

    Args:
        series: Time series data to find peaks in.
        prominence: Minimum prominence of peaks. If None, all peaks are detected.
        plot: Whether to generate a plot of the detected peaks.
        rel_height: Relative height from the peak at which to determine peak width.
        **kwargs: Additional arguments passed to scipy.signal.find_peaks.

    Returns:
        DataFrame containing peak properties including retention times and heights.
    """
    t = series.index
    x = series.values
    peak_ndxs, _ = find_peaks(x, prominence=prominence, rel_height=rel_height, **kwargs)
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


def _map_ndxs_to_time(
    x: Union[List[float], np.ndarray], t_min: float, t_max: float, x_min: float, x_max: float
) -> np.ndarray:
    """Map indices to time values using linear interpolation.

    Args:
        x: Indices to map to time values.
        t_min: Minimum time value.
        t_max: Maximum time value.
        x_min: Minimum index value.
        x_max: Maximum index value.

    Returns:
        Array of time values corresponding to the input indices.

    Raises:
        AssertionError: If t_min ≥ t_max or x_min ≥ x_max.
    """
    assert t_min < t_max
    assert x_min < x_max
    t_span = t_max - t_min
    x_span = x_max - x_min
    m = t_span / x_span
    b = t_min
    x = np.array(x)
    result = (m * x + b).flatten()
    return result


def mz_mean_width_to_min_max(mz_mean: float, mz_width: float) -> Tuple[float, float]:
    """Convert m/z mean and width (in ppm) to min and max m/z values.

    Args:
        mz_mean: Mean m/z value.
        mz_width: Width in parts-per-million (ppm).

    Returns:
        Tuple of (mz_min, mz_max) defining the m/z range.
    """
    delta_mass = mz_width * mz_mean * 1e-6
    mz_min = mz_mean - delta_mass
    mz_max = mz_mean + delta_mass
    return mz_min, mz_max


def init_metadata() -> pd.DataFrame:
    """Initialize an empty metadata DataFrame with the standard columns.

    Returns:
        Empty DataFrame with standard metadata columns and 'ms_file_label' as index.
    """
    cols = MINT_METADATA_COLUMNS
    return pd.DataFrame(columns=cols).set_index("ms_file_label")


def fn_to_label(fn: Union[str, P]) -> str:
    """Convert a filename to a label by removing the file extension.

    Args:
        fn: Filename or path.

    Returns:
        Filename without extension.
    """
    return P(fn).with_suffix("").name
