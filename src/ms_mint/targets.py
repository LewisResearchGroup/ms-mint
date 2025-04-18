"""Everything related to target lists."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path as P
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any, Tuple, Callable, Literal

from .Chromatogram import Chromatogram
from .processing import get_chromatogram_from_ms_file, extract_chromatogram_from_ms1
from .io import ms_file_to_df
from .standards import TARGETS_COLUMNS, DEPRECATED_LABELS
from .tools import formula_to_mass, df_diff


def read_targets(fns: Union[str, List[str]], ms_mode: str = "negative") -> pd.DataFrame:
    """Extract peak data from files containing peak definitions.

    Args:
        fns: List of filenames of target lists or a single filename.
        ms_mode: Mass spectrometry ionization mode, either "negative" or "positive".

    Returns:
        DataFrame containing standardized target information from all input files.
    """
    if isinstance(fns, str):
        fns = [fns]
    targets = []

    for fn in fns:
        fn = str(fn)
        if fn.endswith(".csv"):
            df = pd.read_csv(fn)
        elif fn.endswith(".xlsx"):
            df = pd.read_excel(fn)
        df = standardize_targets(df)
        df["target_filename"] = P(fn).name
        targets.append(df)

    targets = pd.concat(targets)
    return targets


def standardize_targets(targets: pd.DataFrame, ms_mode: str = "neutral") -> pd.DataFrame:
    """Standardize target list format and units.

    This function:
    - Updates the target lists to newest format
    - Ensures peak labels are strings
    - Replaces np.nan with None
    - Converts retention times to seconds
    - Fills missing values with reasonable defaults

    Args:
        targets: DataFrame in target-list format.
        ms_mode: Ionization mode. Options are "neutral", "positive", or "negative".

    Returns:
        DataFrame in standardized target-list format.

    Raises:
        AssertionError: If there are duplicate column names in the input DataFrame.
    """
    targets = targets.rename(columns=DEPRECATED_LABELS)
    if targets.index.name == "peak_label":
        targets = targets.reset_index()

    assert pd.Series(targets.columns).value_counts().max() == 1, pd.Series(targets.columns).value_counts()

    cols = targets.columns
    if "formula" in targets.columns and not "mz_mean" in targets.columns:
        targets["mz_mean"] = formula_to_mass(targets["formula"], ms_mode)
    if "intensity_threshold" not in cols:
        targets["intensity_threshold"] = 0
    if "mz_width" not in cols:
        targets["mz_width"] = 10
    if "target_filename" not in cols:
        targets["target_filename"] = "unknown"
    if "rt_unit" not in targets.columns:
        targets["rt_unit"] = "min"

    # Standardize time units use SI abbreviations
    targets["rt_unit"] = targets["rt_unit"].replace("m", "min")
    targets["rt_unit"] = targets["rt_unit"].replace("minute", "min")
    targets["rt_unit"] = targets["rt_unit"].replace("minutes", "min")
    targets["rt_unit"] = targets["rt_unit"].replace("sec", "s")
    targets["rt_unit"] = targets["rt_unit"].replace("second", "s")
    targets["rt_unit"] = targets["rt_unit"].replace("seconds", "s")

    for c in ["rt", "rt_min", "rt_max"]:
        if c not in cols:
            targets[c] = None
            targets[c] = targets[c].astype(float)

    if "peak_label" not in cols:
        logging.warning(f'"peak_label" not in cols, assigning new labels:\n{targets}')
        targets["peak_label"] = [f"C_{i}" for i in range(len(targets))]

    targets["intensity_threshold"] = targets["intensity_threshold"].fillna(0)
    targets["peak_label"] = targets["peak_label"].astype(str)

    targets.index = range(len(targets))
    targets = targets[targets.mz_mean.notna()]
    targets = targets.replace(np.nan, None)
    fill_missing_rt_values(targets)
    convert_to_seconds(targets)

    if "rt" in targets.columns:
        targets["rt"] = targets["rt"].astype(float)

    return targets[TARGETS_COLUMNS]


def convert_to_seconds(targets: pd.DataFrame) -> None:
    """Convert retention time units to seconds.

    Args:
        targets: Mint target list to modify in-place.
    """
    for ndx, row in targets.iterrows():
        if row.rt_unit == "min":
            targets.loc[ndx, "rt_unit"] = "s"
            if targets.loc[ndx, "rt"]:
                targets.loc[ndx, "rt"] *= 60.0
            if targets.loc[ndx, "rt_min"]:
                targets.loc[ndx, "rt_min"] *= 60.0
            if targets.loc[ndx, "rt_max"]:
                targets.loc[ndx, "rt_max"] *= 60.0


def fill_missing_rt_values(targets: pd.DataFrame) -> None:
    """Fill missing rt values with mean of rt_min and rt_max.

    Args:
        targets: Mint target list to modify in-place.
    """
    for ndx, row in targets.iterrows():
        if (not row.rt) and (row.rt_min and row.rt_max):
            targets.loc[ndx, "rt"] = np.mean([row.rt_min, row.rt_max])


def check_targets(targets: pd.DataFrame) -> bool:
    """Check if targets are formatted correctly.

    Args:
        targets: Target list DataFrame to check.

    Returns:
        True if all checks pass, else False.
    """
    results = (
        isinstance(targets, pd.DataFrame),
        _check_target_list_columns_(targets),
        _check_labels_are_strings_(targets),
        _check_duplicated_labels_(targets),
    )
    result = all(results)
    if not result:
        print(results)
    return all(results)


def _check_labels_are_strings_(targets: pd.DataFrame) -> bool:
    """Check if peak labels are strings.

    Args:
        targets: Target list DataFrame to check.

    Returns:
        True if all peak labels are strings, else False.
    """
    if not targets.dtypes["peak_label"] == np.dtype("O"):
        logging.warning("Target labels are not strings.")
        return False
    return True


def _check_duplicated_labels_(targets: pd.DataFrame) -> bool:
    """Check if peak labels are unique.

    Args:
        targets: Target list DataFrame to check.

    Returns:
        True if all peak labels are unique, else False.
    """
    max_target_label_count = targets.peak_label.value_counts().max()
    if max_target_label_count > 1:
        logging.warning("Target labels are not unique")
        return False
    return True


def _check_target_list_columns_(targets: pd.DataFrame) -> bool:
    """Check if target list has the correct columns.

    Args:
        targets: Target list DataFrame to check.

    Returns:
        True if all required columns are present, else False.
    """
    if targets.columns.to_list() != TARGETS_COLUMNS:
        logging.warning("Target columns are wrong.")
        return False
    return True


def gen_target_grid(
    masses: List[float],
    dt: float,
    rt_max: float = 10,
    mz_ppm: float = 10,
    intensity_threshold: float = 0,
) -> pd.DataFrame:
    """Create a target grid from a list of masses.

    Generates a grid of targets by combining each mass with a series of
    retention time windows spanning from 0 to rt_max.

    Args:
        masses: List of target m/z values.
        dt: Size of peak windows in time dimension [min].
        rt_max: Maximum retention time to include.
        mz_ppm: Width of peak window in m/z dimension [ppm].
        intensity_threshold: Minimum intensity threshold for peaks.

    Returns:
        DataFrame containing generated target list.
    """
    rt_cuts = np.arange(0, rt_max + dt, dt)
    targets = pd.DataFrame(index=rt_cuts, columns=masses).unstack().reset_index()
    del targets[0]
    targets.columns = ["mz_mean", "rt_min"]
    targets["rt_max"] = targets.rt_min + (1 * dt)
    targets["peak_label"] = (
        targets.mz_mean.apply("{:.3f}".format) + "__" + targets.rt_min.apply("{:2.2f}".format)
    )
    targets["mz_width"] = mz_ppm
    targets["intensity_threshold"] = intensity_threshold
    targets["targets_name"] = "gen_target_grid"
    return targets


def diff_targets(old_pklist: pd.DataFrame, new_pklist: pd.DataFrame) -> pd.DataFrame:
    """Get the difference between two target lists.

    Args:
        old_pklist: Original target list.
        new_pklist: New target list to compare against the original.

    Returns:
        DataFrame containing only the targets that are new or changed.
    """
    df = df_diff(old_pklist, new_pklist)
    df = df[df["_merge"] == "right_only"]
    return df.drop("_merge", axis=1)
