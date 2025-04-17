# ms_mint/processing.py

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path as P
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from .tools import lock, mz_mean_width_to_min_max
from .io import ms_file_to_df
from .standards import RESULTS_COLUMNS, MINT_RESULTS_COLUMNS


def extract_chromatogram_from_ms1(
    df: pd.DataFrame, mz_mean: float, mz_width: float = 10
) -> pd.Series:
    """Extract single chromatogram of specific m/z value from MS-data.

    Args:
        df: MS-data with columns ['scan_time', 'mz', 'intensity'].
        mz_mean: Target m/z value.
        mz_width: m/z width in ppm. Default is 10.

    Returns:
        Chromatogram as a pandas Series with scan_time as index and intensity as values.
    """
    mz_min, mz_max = mz_mean_width_to_min_max(mz_mean, mz_width)
    chrom = df[(df.mz >= mz_min) & (df.mz <= mz_max)].copy()
    chrom["scan_time"] = chrom["scan_time"].round(3)
    chrom = chrom.groupby("scan_time").max()
    return chrom["intensity"]


def process_ms1_files_in_parallel(args: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Process MS files in parallel using the provided arguments.

    This is a pickleable function for parallel peak integration that can be used
    with multiprocessing.

    Args:
        args: Dictionary containing the following keys:
            - filename: Path to the MS file to process.
            - targets: DataFrame with target compounds information.
            - output_fn: Optional output filename to save results.
            - queue: Optional queue for progress reporting.

    Returns:
        DataFrame with processing results, or None if results were saved to a file.
    """
    filename = args["filename"]
    targets = args["targets"]
    output_fn = args["output_fn"]

    if "queue" in args.keys():
        q = args["queue"]
        q.put("filename")

    try:
        results = process_ms1_file(filename=filename, targets=targets)
    except Exception as e:
        logging.error(f"process_ms1_files_in_parallel(): {e}")
        results = pd.DataFrame()

    if (output_fn is not None) and (len(results) > 0):
        append_results(results, output_fn)
        return None

    return results


def append_results(results: pd.DataFrame, fn: str) -> None:
    """Append results to a CSV file with file locking for thread safety.

    Args:
        results: Results DataFrame to append.
        fn: Filename to append to.
    """
    with lock(fn):
        results.to_csv(fn, mode="a", header=False, index=False)


def process_ms1_file(filename: Union[str, P], targets: pd.DataFrame) -> pd.DataFrame:
    """Perform peak integration using a filename as input.

    Args:
        filename: Path to mzxml or mzml file.
        targets: DataFrame in target list format.

    Returns:
        DataFrame with processed peak intensities.
    """
    df = ms_file_to_df(filename)
    results = process_ms1(df, targets)
    results["total_intensity"] = df["intensity"].sum()
    results["ms_file"] = str(filename)
    results["ms_file_label"] = P(filename).with_suffix("").name
    results["ms_file_size_MB"] = os.path.getsize(filename) / 1024 / 1024
    results["peak_score"] = 0  # score_peaks(results)
    return results[MINT_RESULTS_COLUMNS]


def process_ms1(df: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Process MS-1 data with a target list.

    Args:
        df: MS-1 data with columns ['scan_time', 'mz', 'intensity'].
        targets: Target list DataFrame with required columns.

    Returns:
        DataFrame with peak integration results.
    """
    results = _process_ms1_from_df_(df, targets)
    results = pd.DataFrame(results, columns=["peak_label"] + RESULTS_COLUMNS)
    results = pd.merge(targets, results, on=["peak_label"])
    results = results.reset_index(drop=True)
    return results


def _process_ms1_from_df_(df: pd.DataFrame, targets: pd.DataFrame) -> List[List[Any]]:
    """Internal function to process MS-1 data from DataFrame.

    Args:
        df: MS data DataFrame.
        targets: Targets DataFrame.

    Returns:
        List of peak data for each target.
    """
    peak_cols = [
        "mz_mean",
        "mz_width",
        "rt_min",
        "rt_max",
        "intensity_threshold",
        "peak_label",
    ]
    array_peaks = targets[peak_cols].values
    array_data = df[["scan_time", "mz", "intensity"]].values
    result = process_ms1_from_numpy(array_data, array_peaks)
    return result


def process_ms1_from_numpy(array: np.ndarray, peaks: np.ndarray) -> List[List[Any]]:
    """Process MS1 data in numpy array format.

    Args:
        array: Input data array with columns [scan_time, mz, intensity].
        peaks: Peak data array with columns [mz_mean, mz_width, rt_min, rt_max,
            intensity_threshold, peak_label].

    Returns:
        List of extracted data for each peak.
    """
    results = []
    for mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label in peaks:
        props = _process_ms1_from_numpy(
            array,
            mz_mean=mz_mean,
            mz_width=mz_width,
            rt_min=rt_min,
            rt_max=rt_max,
            intensity_threshold=intensity_threshold,
            peak_label=peak_label,
        )
        if props is None:
            continue
        results.append([props[col] for col in ["peak_label"] + RESULTS_COLUMNS])
    return results


def _process_ms1_from_numpy(
    array: np.ndarray,
    mz_mean: float,
    mz_width: float,
    rt_min: float,
    rt_max: float,
    intensity_threshold: float,
    peak_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Internal function to process a single peak from numpy array.

    Args:
        array: Input data array with columns [scan_time, mz, intensity].
        mz_mean: Mean m/z value.
        mz_width: Width of m/z window in ppm.
        rt_min: Minimum retention time.
        rt_max: Maximum retention time.
        intensity_threshold: Minimum intensity threshold.
        peak_label: Label for the peak.

    Returns:
        Dictionary of peak properties or None if no data points were found.
    """
    _slice = slice_ms1_array(
        array=array,
        mz_mean=mz_mean,
        mz_width=mz_width,
        rt_min=rt_min,
        rt_max=rt_max,
        intensity_threshold=intensity_threshold,
    )
    props = extract_ms1_properties(_slice, mz_mean)
    if props is None:
        return None
    if peak_label is not None:
        props["peak_label"] = peak_label
    return props


def extract_ms1_properties(array: np.ndarray, mz_mean: float) -> Dict[str, Any]:
    """Extract peak properties from an MS-1 data slice.

    Args:
        array: MS-1 data slice array with columns [scan_time, mz, intensity].
        mz_mean: Mean m/z value for calculating mass accuracy.

    Returns:
        Dictionary of extracted peak properties.
    """
    float_list_to_comma_sep_str = lambda x: ",".join([str(np.round(i, 4)) for i in x])
    int_list_to_comma_sep_str = lambda x: ",".join([str(int(i)) for i in x])

    projection = pd.DataFrame(array[:, [0, 2]], columns=["rt", "int"])

    projection["rt"] = projection["rt"].round(2)
    projection["int"] = projection["int"].astype(int)
    projection = projection.groupby("rt").max().reset_index().values

    times = array[:, 0]
    masses = array[:, 1]
    intensities = array[:, 2]
    peak_n_datapoints = len(array)

    if peak_n_datapoints == 0:
        return dict(
            peak_area=0,
            peak_area_top3=0,
            peak_max=0,
            peak_min=0,
            peak_mean=None,
            peak_rt_of_max=None,
            peak_median=None,
            peak_delta_int=None,
            peak_n_datapoints=0,
            peak_mass_diff_25pc=None,
            peak_mass_diff_50pc=None,
            peak_mass_diff_75pc=None,
            peak_shape_rt="",
            peak_shape_int="",
            peak_score=None,
        )

    peak_area = intensities.sum()

    # Like El-Maven peakAreaTop
    # Alternative approach to calculate peak_area_top3
    ndx_max = np.argmax(projection[:, 1])
    peak_area_top3 = projection[:, 1][max(0, ndx_max - 1) : ndx_max + 2].sum() // 3

    peak_mean = intensities.mean()
    peak_max = intensities.max()
    peak_min = intensities.min()
    peak_median = np.median(intensities)

    peak_rt_of_max = times[intensities.argmax()]

    peak_delta_int = np.abs(intensities[0] - intensities[-1])

    peak_mass_diff_25pc, peak_mass_diff_50pc, peak_mass_diff_75pc = np.quantile(
        masses, [0.25, 0.5, 0.75]
    )

    peak_mass_diff_25pc -= mz_mean
    peak_mass_diff_50pc -= mz_mean
    peak_mass_diff_75pc -= mz_mean

    peak_mass_diff_25pc /= 1e-6 * mz_mean
    peak_mass_diff_50pc /= 1e-6 * mz_mean
    peak_mass_diff_75pc /= 1e-6 * mz_mean

    peak_shape_rt = float_list_to_comma_sep_str(projection[:, 0])
    peak_shape_int = int_list_to_comma_sep_str(projection[:, 1])

    # Check that peak projection arrays (rt, int) have same number of elements
    assert len(peak_shape_rt.split(",")) == len(peak_shape_int.split(","))

    return dict(
        peak_area=peak_area,
        peak_area_top3=peak_area_top3,
        peak_max=peak_max,
        peak_min=peak_min,
        peak_mean=peak_mean,
        peak_rt_of_max=peak_rt_of_max,
        peak_median=peak_median,
        peak_delta_int=peak_delta_int,
        peak_n_datapoints=peak_n_datapoints,
        peak_mass_diff_25pc=peak_mass_diff_25pc,
        peak_mass_diff_50pc=peak_mass_diff_50pc,
        peak_mass_diff_75pc=peak_mass_diff_75pc,
        peak_shape_rt=peak_shape_rt,
        peak_shape_int=peak_shape_int,
        peak_score=None,
    )


def slice_ms1_array(
    array: np.ndarray,
    rt_min: float,
    rt_max: float,
    mz_mean: float,
    mz_width: float,
    intensity_threshold: float,
) -> np.ndarray:
    """Slice MS1 data by m/z, retention time, and intensity threshold.

    Args:
        array: Input MS-1 data array with columns [scan_time, mz, intensity].
        rt_min: Minimum retention time for slice.
        rt_max: Maximum retention time for slice.
        mz_mean: Mean m/z value for slice.
        mz_width: Width of slice in ppm of mz_mean.
        intensity_threshold: Noise filter value.

    Returns:
        Filtered numpy array containing only data points meeting the criteria.
    """
    delta_mass = mz_width * mz_mean * 1e-6
    array = array[(array[:, 0] >= rt_min)]
    array = array[(array[:, 0] <= rt_max)]
    array = array[(np.abs(array[:, 1] - mz_mean) <= delta_mass)]
    array = array[(array[:, 2] >= intensity_threshold)]
    return array


def score_peaks(mint_results: pd.DataFrame) -> pd.Series:
    """Score the peak quality (experimental).

    Calculates a score from 0 to 1 where:
    - 1 means a good peak shape
    - 0 means a bad peak shape

    Args:
        mint_results: DataFrame in ms_mint results format.

    Returns:
        Series of scores for each peak.
    """
    R = mint_results.copy()
    scores = (
        (1 - R.peak_delta_int.apply(abs) / R.peak_max)
        * (np.tanh(R.peak_n_datapoints / 20))
        * (1 / (1 + abs(R.peak_rt_of_max - R[["rt_min", "rt_max"]].mean(axis=1))))
    )
    return scores


def get_chromatogram_from_ms_file(
    ms_file: Union[str, P], mz_mean: float, mz_width: float = 10
) -> pd.Series:
    """Get chromatogram data from an MS file.

    Args:
        ms_file: Path to the MS file.
        mz_mean: Mean m/z value to extract.
        mz_width: Width around the mean m/z in ppm to extract.

    Returns:
        Chromatogram data as a pandas Series with scan_time as index
        and intensity as values.
    """
    df = ms_file_to_df(ms_file)
    chrom = extract_chromatogram_from_ms1(df, mz_mean, mz_width=mz_width)
    return chrom
