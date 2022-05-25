# ms_mint/targets.py

import pandas as pd
import numpy as np
import logging

from pathlib import Path as P

from .standards import TARGETS_COLUMNS, DEPRECATED_LABELS
from .helpers import df_diff
from .tools import get_mz_mean_from_formulas


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
    """
    Test if
    1) targets has right type,
    2) all columns are present
    3) dtype of column peak_label is string
    4) rt_min and rt_max are set
    Returns a list of strings indicating identified errors.
    If list is empty targets is OK.
    """
    results = (
        isinstance(targets, pd.DataFrame),
        check_target_list_columns(targets),
        check_labels_are_strings(targets),
        check_duplicated_labels(targets),
        check_targets_rt_values(targets),
    )
    print(results)
    return all(results)


def check_labels_are_strings(targets):
    if not targets.dtypes["peak_label"] == np.dtype("O"):
        logging.warning('Target labels are not strings.')
        return False
    return True


def check_duplicated_labels(targets):
    max_target_label_count = targets.peak_label.value_counts().max()
    if max_target_label_count > 1:
        logging.warning('Target labels are not unique')
        return False
    return True


def check_target_list_columns(targets):
    if targets.columns.to_list() != TARGETS_COLUMNS:
        logging.warning('Target columns are wrong.')
        return False
    return True


def check_targets_rt_values(targets):
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
    df = df_diff(old_pklist, new_pklist)
    df = df[df["_merge"] == "right_only"]
    return df.drop("_merge", axis=1)


