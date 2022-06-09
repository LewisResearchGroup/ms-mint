import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from ms_mint.chromatogram import Chromatogram
from ms_mint.tools import gaussian

from paths import TEST_MZML


def test__Chromatogram_without_filter_identifies_peaks_correctly():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1000)
    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_2, 100) * 5e5

    chrom = Chromatogram(scan_times=x, intensities=y, filter=None, expected_rt=2)
    chrom.find_peaks()

    rt_peak_1_pred, rt_peak_2_pred = chrom.peaks.rt.values

    assert [rt_peak_1_pred, rt_peak_2_pred] == [rt_peak_1, rt_peak_2], [
        rt_peak_1_pred,
        rt_peak_2_pred,
    ]


def test__Chromatogram__with_filter_identifies_peaks_correctly():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1000)
    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_2, 100) * 5e5

    chrom = Chromatogram(scan_times=x, intensities=y, filter=[], expected_rt=2)
    chrom.apply_filter()
    chrom.find_peaks()
    rt_peak_1_pred, rt_peak_2_pred = chrom.peaks.rt.values

    max_diff = max(abs(rt_peak_1 - rt_peak_1_pred), abs(rt_peak_2 - rt_peak_2_pred))

    assert max_diff < 1


def test__Chromatogram__with_filter_and_opt_identifies_peaks_correctly():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_2, 100) * 5e5

    chrom = Chromatogram(scan_times=x, intensities=y, filter=None, expected_rt=2)
    chrom.apply_filter()
    chrom.find_peaks()
    chrom.optimise_peak_times_with_diff()
    rt_peak_1_pred, rt_peak_2_pred = chrom.peaks.rt.values

    max_diff = max(abs(rt_peak_1 - rt_peak_1_pred), abs(rt_peak_2 - rt_peak_2_pred))

    assert max_diff < 1


def test__Chromatogram__select_peak_by_rt():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_2, 100) * 5e5

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=rt_peak_1
    )
    chrom.find_peaks()
    chrom.select_peak_by_rt()
    selected_ndx = chrom.selected_peak_ndxs

    expected_ndx = [0]

    assert selected_ndx == expected_ndx, selected_ndx


def test__Chromatogram__select_peak_by_rt_as_argument():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_2, 100) * 5e5

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=rt_peak_1
    )
    chrom.find_peaks()
    chrom.select_peak_by_rt(rt_peak_2)
    selected_ndx = chrom.selected_peak_ndxs

    expected_ndx = [1]

    assert selected_ndx == expected_ndx, selected_ndx


def test__Chromatogram__select_highest():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_1, 100) * 5e5

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=rt_peak_2
    )
    chrom.find_peaks()
    chrom.select_peak_by_highest_intensity()
    selected_ndx = chrom.selected_peak_ndxs

    expected_ndx = [0]

    assert selected_ndx == expected_ndx, selected_ndx


def test__Chromatogram__select_peak_with_gaussian_prefers_higher_peak():

    higher_peak_rt = 400
    higher_peak_in = 5e6
    lower__peak_rt = 600
    lower__peak_in = 5e3

    # Slightly shifted towards lower peak
    expected_rt = np.mean([higher_peak_rt, lower__peak_rt]) + 10

    # Gaussian mixture
    x = np.arange(1, 1000)

    y = (
        gaussian(x, higher_peak_rt, 200) * higher_peak_in
        + gaussian(x, lower__peak_rt, 100) * lower__peak_in
    )

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=expected_rt
    )
    chrom.find_peaks()
    chrom.select_peak_with_gaussian_weight()
    selected_ndx = chrom.selected_peak_ndxs

    expected_ndx = [0]

    assert selected_ndx == expected_ndx, selected_ndx


def test__Chromatogram__plot_runs_without_error_return_figure():

    peak_rt = 400
    peak_in = 5e6

    x = np.arange(1, 1000)

    y = gaussian(x, peak_rt, 200) * peak_in

    chrom = Chromatogram(scan_times=x, intensities=y, filter=None)
    chrom.find_peaks()

    fig = chrom.plot()

    assert isinstance(fig, plt.Figure), type(fig)


def test__Chromatogram_from_files_runs_through():

    chrom = Chromatogram()
    chrom.from_file(TEST_MZML, mz_mean=101.024323, mz_width=10)
    chrom.apply_filter()
    chrom.find_peaks()
    chrom.select_peak_by_highest_intensity()
    peaks = chrom.selected_peak_ndxs

    print(chrom.data)
    print(peaks)

    assert len(peaks) != 0


def test__Chromatogram__optimise_peak_times_with_diff_with_plot():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_1, 100) * 5e5

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=rt_peak_2
    )
    chrom.find_peaks()
    chrom.optimise_peak_times_with_diff(plot=True)


def test__Chromatogram__data():

    rt_peak_1 = 200
    rt_peak_2 = 500

    x = np.arange(1, 1000)

    y = gaussian(x, rt_peak_1, 20) * 5e6 + gaussian(x, rt_peak_1, 100) * 5e5

    chrom = Chromatogram(
        scan_times=x, intensities=y, filter=None, expected_rt=rt_peak_2
    )

    data = chrom.data

    assert isinstance(data, pd.DataFrame), type(data)
    assert len(chrom.data) == len(x)+1
    #assert all(data.index == x), data.index
    #assert all(data.intensity.values == [0]+y), data.values
