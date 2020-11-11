from ms_mint.peak_detection import OpenMSPeakDetection as OMS


def test__peaks_are_close__identical_peaks():
    peak_a = (100, 1, 2, 1.5)
    peak_b = (100, 1, 2, 1.5)
    result = OMS.peaks_are_close(peak_a, peak_b)
    expected = True
    assert result == expected, result


def test__peaks_are_close__different_masses():
    peak_a = (100, 1, 2, 1.5)
    peak_b = (101, 1, 2, 1.5)
    result = OMS.peaks_are_close(peak_a, peak_b)
    expected = False
    assert result == expected, result


def test__peaks_are_close__different_rt():
    peak_a = (100, 1, 2, 1.5)
    peak_b = (100, 2, 3, 2.5)
    result = OMS.peaks_are_close(peak_a, peak_b)
    expected = False
    assert result == expected, result


def test__peaks_are_close__different_rtminmax():
    peak_a = (100, 1, 2, None)
    peak_b = (100, 2, 3, None)
    result = OMS.peaks_are_close(peak_a, peak_b)
    expected = False
    assert result == expected, result


def test__peaks_are_close__close_mass():
    peak_a = (100, 1, 2, None)
    peak_b = (100.001, 1, 2, None)
    result = OMS.peaks_are_close(peak_a, peak_b)
    expected = True
    assert result == expected, result


def test__peaks_are_close__mass_delta():
    peak_a = (100, 1, 2, None)
    peak_b = (100.01, 1, 2, None)
    result = (OMS.peaks_are_close(peak_a, peak_b, max_delta_mz_ppm=10),
              OMS.peaks_are_close(peak_a, peak_b, max_delta_mz_ppm=100))
    expected = (False, True)
    assert result == expected, result


def test__peaks_are_close__rt_delta_rtminmax():
    peak_a = (100, 1, 2, None)
    peak_b = (100.001, 1.1, 2.1, None)
    result = (OMS.peaks_are_close(peak_a, peak_b, max_delta_rt=0.10001),
              OMS.peaks_are_close(peak_a, peak_b, max_delta_rt=0.1))
    expected = (True, False)
    assert result == expected, result


def test__peaks_are_close__rt_delta():
    peak_a = (100, None, None, 1)
    peak_b = (100.001, None, None, 1.1)
    result = (OMS.peaks_are_close(peak_a, peak_b, max_delta_rt=0.10001),
              OMS.peaks_are_close(peak_a, peak_b, max_delta_rt=0.1))
    expected = (True, False)
    assert result == expected, result


