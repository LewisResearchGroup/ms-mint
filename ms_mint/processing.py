# ms_mint/processing.py

import os
import pandas as pd
import numpy as np
import logging

from  pathlib import Path as P

from .tools import lock, mz_mean_width_to_min_max

from .io import ms_file_to_df

from .standards import RESULTS_COLUMNS, MINT_RESULTS_COLUMNS


def extract_chromatogram_from_ms1(df, mz_mean, mz_width=10):
    """
    Extract single chromatogram of specific m/z value from MS-data.

    :param df: MS-data
    :type df: pandas.DataFrame with columns ['rt', 'm/z', 'intensity']
    :param mz_mean: Target m/z value
    :type mz_mean: float
    :param mz_width: m/z width in ppm, defaults to 10
    :type mz_width: float
    :return: Chromatogram
    :rtype: pandas.DataFrame
    """
    mz_min, mz_max = mz_mean_width_to_min_max(mz_mean, mz_width)
    chrom = df[(df.mz >= mz_min) & (df.mz <= mz_max)].copy()
    chrom["scan_time"] = chrom["scan_time"].round(3)
    chrom = chrom.groupby("scan_time").max()
    return chrom["intensity"]


def process_ms1_files_in_parallel(args):
    """
    Pickleable function for (parallel) peak integration.
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


def append_results(results, fn):
    """
    Appends results to file.

    :param results: New results.
    :type results: pandas.DataFrame
    :param fn: Filename to append to.
    :type fn: str
    """
    with lock(fn):
        results.to_csv(fn, mode="a", header=False, index=False)


def process_ms1_file(filename, targets):
    """
    Peak integration using a filename as input.

    :param filename: Path to mzxml or mzml filename
    :type filename: str or PosixPath
    :param targets: DataFrame in target list format.
    :type targets: pandas.DataFrame
    :return: DataFrame with processd peak intensities.
    :rtype: pandas.DataFrame
    """
    df = ms_file_to_df(filename)
    results = process_ms1(df, targets)
    results["total_intensity"] = df["intensity"].sum()
    results["ms_file"] = filename
    results["ms_file_label"] = P(filename).with_suffix('').name
    results["ms_file_size_MB"] = os.path.getsize(filename) / 1024 / 1024
    results["peak_score"] = 0 #  score_peaks(results)
    return results[MINT_RESULTS_COLUMNS]


def process_ms1(df, targets):
    """
    Process MS-1 data with a target list.

    :param df: MS-1 data.
    :type df: pandas.DataFrame
    :param targets: Target list
    :type targets: pandas.DataFrame
    :return: Mint results.
    :rtype: pandas.DataFrame
    """
    results = _process_ms1_from_df_(df, targets)
    results = pd.DataFrame(results, columns=["peak_label"] + RESULTS_COLUMNS)
    results = pd.merge(targets, results, on=["peak_label"])
    results = results.reset_index(drop=True)
    return results


def _process_ms1_from_df_(df, targets):
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


def process_ms1_from_numpy(array, peaks):
    """
    Process MS1 data in numpy array format.

    :param array: Input data.
    :type array: numpy.Array
    :param peaks: Peak data np.array([[mz_mean_1, mz_width_1, rt_min_1, rt_max_1, intensity_threshold_1, peak_label_1], ...])
    :type peaks: numpy.Array
    :return: Extracted data.
    :rtype: list
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
    array, mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label=None
):
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
        return
    if peak_label is not None:
        props["peak_label"] = peak_label
    return props


def extract_ms1_properties(array, mz_mean):
    """
    Process MS-1 data in array format.

    :param array: MS-1 data slice.
    :type array: numpy.array
    :param mz_mean: mz_mean value to calculate mass accuracy.
    :type mz_mean: float
    :return: Extracted data.
    :rtype: dict
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
    # ndx_max = np.argmax(intensities)
    # peak_area_top3 = intensities[max(0, ndx_max - 1) : ndx_max + 2].mean()
    # Try an alternative apporach to calculate peak_area_top3
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
    array: np.array, rt_min, rt_max, mz_mean, mz_width, intensity_threshold
):
    """
    Slice MS1 data by m/z, mz_width, rt_min, rt_max

    :param array: Input MS-1 data.
    :type array: np.array
    :param rt_min: Minimum retention time for slice
    :type rt_min: float
    :param rt_max: Maximum retention time for slice
    :type rt_max: float
    :param mz_mean: Mean m/z value for slice
    :type mz_mean: float
    :param mz_width: Width of slice in [ppm] of mz_mean
    :type mz_width: float (>0)
    :param intensity_threshold: Noise filter value
    :type intensity_threshold: float (>0)
    :return: Slice of numpy array
    :rtype: np.Array
    """
    delta_mass = mz_width * mz_mean * 1e-6
    array = array[(array[:, 0] >= rt_min)]
    array = array[(array[:, 0] <= rt_max)]
    array = array[(np.abs(array[:, 1] - mz_mean) <= delta_mass)]
    array = array[(array[:, 2] >= intensity_threshold)]
    return array


def score_peaks(mint_results):
    """
    Score the peak quality (experimental).

    1 - means a good shape

    0 - means a bad shape

    :param mint_results: DataFrame in ms_mint results format.
    :type mint_results: pandas.DataFrame
    :return: Score
    :rtype: float
    """
    R = mint_results.copy()
    scores = (
        ((1 - R.peak_delta_int.apply(abs) / R.peak_max))
        * (np.tanh(R.peak_n_datapoints / 20))
        * (1 / (1 + abs(R.peak_rt_of_max - R[["rt_min", "rt_max"]].mean(axis=1))))
    )
    return scores


def get_chromatogram_from_ms_file(ms_file: str, mz_mean: float, mz_width: float = 10) -> pd.DataFrame:
    """
    Get chromatogram data from an MS file.

    :param ms_file: Path to the MS file.
    :param mz_mean: Mean m/z value.
    :param mz_width: Width around the mean m/z to extract.
    :return: Chromatogram data as a DataFrame.
    """
    df = ms_file_to_df(ms_file)
    chrom = extract_chromatogram_from_ms1(df, mz_mean, mz_width=mz_width)
    return chrom


def extract_chromatogram_from_ms1(ms1: pd.DataFrame, mz_mean: float, mz_width: float = 10) -> pd.DataFrame:
    """
    Extract chromatogram from MS1 data.

    :param ms1: MS1 data as a DataFrame.
    :param mz_mean: Mean m/z value.
    :param mz_width: Width around the mean m/z to extract.
    :return: Chromatogram data as a DataFrame.
    """
    mz_min, mz_max = mz_mean_width_to_min_max(mz_mean, mz_width)
    chrom = ms1[(ms1["mz"] >= mz_min) & (ms1["mz"] <= mz_max)].copy()
    chrom = chrom.groupby("scan_time", as_index=False).sum(numeric_only=True)
    return chrom.set_index("scan_time")["intensity"]