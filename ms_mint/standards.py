"""
Contains standard column names and other values.
"""

import os

MINT_ROOT = os.path.dirname(__file__)

TARGETS_COLUMNS = [
    "peak_label",
    "mz_mean",
    "mz_width",
    "rt",
    "rt_min",
    "rt_max",
    "rt_unit",
    "intensity_threshold",
    "target_filename",
]

RESULTS_COLUMNS = [
    "peak_area",
    "peak_area_top3",
    "peak_n_datapoints",
    "peak_max",
    "peak_rt_of_max",
    "peak_min",
    "peak_median",
    "peak_mean",
    "peak_delta_int",
    "peak_shape_rt",
    "peak_shape_int",
    "peak_mass_diff_25pc",
    "peak_mass_diff_50pc",
    "peak_mass_diff_75pc",
    "peak_score",
]

MINT_RESULTS_COLUMNS = (
    ["ms_file"]
    + TARGETS_COLUMNS
    + RESULTS_COLUMNS
    + ["total_intensity", "ms_path", "ms_file_size"]
)

DEPRECATED_LABELS = {
    "peakLabel": "peak_label",
    "compound": "peak_label",
    "peakMz": "mz_mean",
    "medRt": "rt",
    "medMz": "mz_mean",
    "peakMzWidth[ppm]": "mz_width",
    "rtmin": "rt_min",
    "rtmax": "rt_max",
    "peaklist": "target_filename",
    "peaklist_name": "target_filename",
    "scan_time_min": "scan_time",
}

M_PROTON = 1.00782503223
