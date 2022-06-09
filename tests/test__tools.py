import numpy as np
import pandas as pd
from ms_mint.tools import (
    formula_to_mass,
    gaussian,
    scale_dataframe,
    df_diff,
    is_ms_file,
)


def test__formula_to_mass():
    result = formula_to_mass(["C", "CCCC", "CNO"])
    expected = [12, 48, 41.998]
    assert result == expected, result


def test__formula_to_mass__positive_ion():
    result = formula_to_mass(["C", "CCCC", "CNO"], ms_mode="positive")
    expected = [13.0078, 49.0078, 43.0058]
    assert result == expected, result


def test__formula_to_mass__negative_ion():
    result = formula_to_mass(["C", "CCCC", "CNO"], ms_mode="negative")
    expected = [10.9922, 46.9922, 40.9902]
    assert result == expected, result


def test__gaussian():
    mu = 0
    sig = 1
    x = np.array([-2, -1, 0, 1, 2]) * sig + mu
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε < 1e-8, ε


def test__gaussian_mu1():
    mu = 1
    sig = 1
    x = np.array([-2, -1, 0, 1, 2]) * sig + mu
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε < 1e-8, ε


def test__gaussian_sig2():
    mu = 0
    sig = 2
    x = np.array([-2, -1, 0, 1, 2]) * sig + mu
    print(x)
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε < 1e-8, result


def test__gaussian_mu1_sig2():
    mu = 5
    sig = 2
    x = np.array([-2, -1, 0, 1, 2]) * sig + mu
    print(x)
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε < 1e-8, result


def test__scale_dataframe_standard():
    N = 100
    data = np.random.uniform(size=(N, N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    scaled = scale_dataframe(df, scaler="standard")
    ε = 1e-8
    means_are_zero = scaled.mean() < ε
    ε = 1e-2
    stds_are_one = scaled.std() - 1 < ε
    assert all(means_are_zero), scaled.mean()
    assert all(stds_are_one), scaled.std()


def test__scale_dataframe_robust():
    N = 100
    data = np.random.uniform(size=(N, N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    scaled = scale_dataframe(df, scaler="robust")
    ε = 1e-8
    medians_are_zero = scaled.median() < ε
    assert all(medians_are_zero), medians_are_zero


def test__df_diff():

    df1 = pd.DataFrame(
        {"A": [1, 1, 0, 0, 2], "B": [1, 0, 0, 1, 2]}, index=[1, 2, 3, 4, 5]
    )

    df2 = pd.DataFrame(
        {"A": [1, 1, 1, 1, 3], "B": [1, 1, 1, 1, 3]}, index=[1, 2, 3, 4, 6]
    )

    result = df_diff(df1, df2)

    _merge = pd.CategoricalIndex(
        ["left_only"] * 4 + ["right_only"],
        categories=["left_only", "right_only", "both"],
        ordered=False,
        dtype="category",
    )

    expected = pd.DataFrame(
        {"A": [1, 0, 0, 2, 3], "B": [0, 0, 1, 2, 3], "_merge": _merge}
    )

    assert result.equals(expected), result.values == expected.values


def test__is_ms_file():

    pos = [
        "file.mzML",
        "path/file.mzML",
        "file.mzXML",
        "path/file.mzXML",
        "file.RAW",
        "path/file.RAW",
        "file.mzMLb",
        "file.mzhdf",
        "file.parquet",
    ]

    neg = ["file.txt", "file.png", "file.ML"]

    res_pos = [is_ms_file(fn) for fn in pos]
    res_neg = [is_ms_file(fn) for fn in neg]

    assert all(res_pos), res_pos
    assert not any(res_neg), res_neg
