# ms_mint/processing.py

import os
import pandas as pd
import numpy as np
import logging

from .tools import lock

from .io import ms_file_to_df

from .standards import RESULTS_COLUMNS, MINT_RESULTS_COLUMNS


def extract_chromatogram_from_ms1(df, mz_mean, mz_width, unit="minutes"):
    dmz = mz_mean * 1e-6 * mz_width
    chrom = df[(df["mz"] - mz_mean).abs() <= dmz].copy()
    chrom["scan_time_min"] = chrom["scan_time_min"].round(3)
    chrom = chrom.groupby("scan_time_min").max()
    return chrom["intensity"]


def process_ms1_files_in_parallel(args):
    """
    Pickleable function for (parallel) peak integration.
    Expects a dictionary with keys:
        Mandatory:
        - 'filename': 'path to file to be processed',
        - 'targets': 'dataframe containing the targets'
        - 'mode': 'express' or 'standard'
            * 'express' omits calculcation of rt projections
        Optional:
        - 'queue': instance of multiprocessing.Manager().Queue()

    Returns tuple with two elements:
        1) results, dataframe with integration results
        2) rt_projection, dictionary of dictionaries with peak shapes
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
        logging.error(e)
        results = pd.DataFrame()

    if (output_fn is not None) and (len(results) > 0):
        append_results(results, output_fn)
        return None

    return results


def append_results(results, fn):
    with lock(fn):
        results.to_csv(fn, mode="a", header=False, index=False)


def process_ms1_file(filename, targets):
    """
    Peak integration using a filename as input.
    -----
    Args:
        - filename: str or PosixPath, path to mzxml or mzml filename
        - targets: pandas.DataFrame(), DataFrame in targets format
    Returns:
        pandas.DataFrame(), DataFrame with processd peak intensities
    """
    df = ms_file_to_df(filename)
    results = process_ms1(df, targets)
    results["total_intensity"] = df["intensity"].sum()
    results["ms_file"] = os.path.basename(filename)
    results["ms_path"] = os.path.dirname(filename)
    results["ms_file_size"] = os.path.getsize(filename) / 1024 / 1024
    results["peak_score"] = score_peaks(results)
    return results[MINT_RESULTS_COLUMNS]


def process_ms1(df, targets):
    results = process_ms1_from_df(df, targets)
    results = pd.DataFrame(results, columns=["peak_label"] + RESULTS_COLUMNS)
    results = pd.merge(targets, results, on=["peak_label"])
    results = results.reset_index(drop=True)
    return results


def process_ms1_from_df(df, targets):
    """
    Processes multiple targets returns numpy array.
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
    # if "ms_level" in df.columns:
    #    df = df[df.ms_level == 1]
    array_data = df[["scan_time_min", "mz", "intensity"]].values
    result = process_ms1_from_numpy(array_data, array_peaks)
    return result


def process_ms1_from_numpy(array, peaks):
    results = []
    for (mz_mean, mz_width, rt_min, rt_max, intensity_threshold, peak_label) in peaks:
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
    peak_area_top3 = np.sort(intensities)[:3].sum()
    peak_mean = intensities.mean()
    peak_max = intensities.max()
    peak_min = intensities.min()
    peak_median = np.median(intensities)

    peak_rt_of_max = times[masses.argmax()]

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
    delta_mass = mz_width * mz_mean * 1e-6
    array = array[(array[:, 0] >= rt_min)]
    array = array[(array[:, 0] <= rt_max)]
    array = array[(np.abs(array[:, 1] - mz_mean) <= delta_mass)]
    array = array[(array[:, 2] >= intensity_threshold)]
    return array


def score_peaks(mint_results):
    R = mint_results.copy()
    scores = (
        ((1 - R.peak_delta_int.apply(abs) / R.peak_max))
        * (np.tanh(R.peak_n_datapoints / 20))
        * (1 / (1 + abs(R.peak_rt_of_max - R[["rt_min", "rt_max"]].mean(axis=1))))
    )
    return scores
