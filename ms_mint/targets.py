# ms_mint/targets.py

import pandas as pd
import numpy as np
import logging

from pathlib import Path as P

from .standards import TARGETS_COLUMNS, DEPRECATED_LABELS
from .tools import get_mz_mean_from_formulas, df_diff


def read_targets(fns, ms_mode="negative"):
    """
    Extracts peak data from csv files that contain peak definitions.
    CSV files must contain columns:

    :param fns: List of filenames of target lists.
    :param ms_mode: "negative" or "positive"
    """
    if isinstance(fns, str):
        fns = [fns]
    targets = []

    for fn in fns:
        if fn.endswith(".csv"):
            df = pd.read_csv(fn)
        elif fn.endswith(".xlsx"):
            df = pd.read_excel(fn)
        df = standardize_targets(df)
        df["target_filename"] = P(fn).name
        targets.append(df)

    targets = pd.concat(targets)
    return targets


def standardize_targets(targets, ms_mode="neutral"):
    """Standardize target list.

    :param targets: _description_
    :type targets: _type_
    :param ms_mode: _description_, defaults to "neutral"
    :type ms_mode: str, optional
    :return: _description_
    :rtype: _type_
    """
    targets = targets.rename(columns=DEPRECATED_LABELS)
    assert pd.value_counts(targets.columns).max() == 1, pd.value_counts(targets.columns)
    cols = targets.columns
    if "formula" in targets.columns and not "mz_mean" in targets.columns:
        targets["mz_mean"] = get_mz_mean_from_formulas(targets["formula"], ms_mode)
    if "intensity_threshold" not in cols:
        targets["intensity_threshold"] = 0
    if "mz_width" not in cols:
        targets["mz_width"] = 10
    if "target_filename" not in cols:
        targets["target_filename"] = "unknown"
    for c in ["rt", "rt_min", "rt_max"]:
        if c not in cols:
            targets[c] = None
    del c
    if "peak_label" not in cols:
        targets["peak_label"] = [f"C_{i}" for i in range(len(targets))]
    targets["intensity_threshold"] = targets["intensity_threshold"].fillna(0)
    targets["peak_label"] = targets["peak_label"].astype(str)
    targets.index = range(len(targets))
    targets = targets[targets.mz_mean.notna()]
    targets = targets.replace(np.NaN, None)
    return targets[TARGETS_COLUMNS]


def check_targets(targets):
    """Check if targets are formated well.

    :param targets: Target list
    :type targets: pandas.DataFrame
    :return: Returns True if all checks pass, else False
    :rtype: bool
    """
    results = (
        isinstance(targets, pd.DataFrame),
        _check_target_list_columns_(targets),
        _check_labels_are_strings_(targets),
        _check_duplicated_labels_(targets),
        _check_targets_rt_values_(targets),
    )
    return all(results)


def _check_labels_are_strings_(targets):
    if not targets.dtypes["peak_label"] == np.dtype("O"):
        logging.warning("Target labels are not strings.")
        return False
    return True


def _check_duplicated_labels_(targets):
    max_target_label_count = targets.peak_label.value_counts().max()
    if max_target_label_count > 1:
        logging.warning("Target labels are not unique")
        return False
    return True


def _check_target_list_columns_(targets):
    if targets.columns.to_list() != TARGETS_COLUMNS:
        logging.warning("Target columns are wrong.")
        return False
    return True


def _check_targets_rt_values_(targets):
    missing_rt = targets.loc[targets[["rt_min", "rt_max"]].isna().max(axis=1)]
    if len(missing_rt) != 0:
        logging.warning("Some targets have missing rt_min or rt_max.")
        return False
    return True


def gen_target_grid(masses, dt, rt_max=10, mz_ppm=10, intensity_threshold=0):
    """
    Creates a targets from a list of masses.

    :param masses: Target m/z values.
    :param dt: Size of peak windows in time dimension [min]
    :param rt_max: Maximum time
    :param mz_ppm: Width of peak window in m/z dimension [ppm].
    """
    rt_cuts = np.arange(0, rt_max + dt, dt)
    targets = pd.DataFrame(index=rt_cuts, columns=masses).unstack().reset_index()
    del targets[0]
    targets.columns = ["mz_mean", "rt_min"]
    targets["rt_max"] = targets.rt_min + (1 * dt)
    targets["peak_label"] = (
        targets.mz_mean.apply(lambda x: "{:.3f}".format(x))
        + "__"
        + targets.rt_min.apply(lambda x: "{:2.2f}".format(x))
    )
    targets["mz_width"] = mz_ppm
    targets["intensity_threshold"] = intensity_threshold
    targets["targets_name"] = "gen_target_grid"
    return targets


def diff_targets(old_pklist, new_pklist):
    """Get the difference between two target lists.

    :param old_pklist: Old target list
    :type old_pklist: pandas.DataFrame
    :param new_pklist: New target list
    :type new_pklist: pandas.DataFrame
    :return: Target list with new/changed targets
    :rtype: pandas.DataFrame
    """
    df = df_diff(old_pklist, new_pklist)
    df = df[df["_merge"] == "right_only"]
    return df.drop("_merge", axis=1)
