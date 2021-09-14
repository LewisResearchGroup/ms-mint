import numpy as np
import pandas as pd
from ms_mint.tools import get_mz_mean_from_formulas, gaussian, scale_dataframe


def test__get_mz_mean_from_formulas():
    result = get_mz_mean_from_formulas(['C', 'CCCC', 'CNO'])
    expected = [12, 48, 41.998]
    assert result==expected, result


def test__get_mz_mean_from_formulas__positive_ion():
    result = get_mz_mean_from_formulas(['C', 'CCCC', 'CNO'], ms_mode='positive')
    expected = [13.0078, 49.0078, 43.0058]
    assert result==expected, result

def test__get_mz_mean_from_formulas__negative_ion():
    result = get_mz_mean_from_formulas(['C', 'CCCC', 'CNO'], ms_mode='negative')
    expected = [10.9922, 46.9922, 40.9902]
    assert result==expected, result


def test__gaussian():
    mu = 0
    sig = 1
    x = np.array([-2, -1 , 0, 1, 2])  * sig + mu 
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε<1e-8, ε


def test__gaussian_mu1():
    mu = 1
    sig = 1
    x = np.array([-2, -1 , 0, 1, 2])  * sig + mu
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε<1e-8, ε


def test__gaussian_sig2():
    mu = 0
    sig = 2
    x = np.array([-2, -1 , 0, 1, 2]) * sig + mu
    print(x)
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε<1e-8, result


def test__gaussian_mu1_sig2():
    mu = 5
    sig = 2
    x = np.array([-2, -1 , 0, 1, 2]) * sig + mu
    print(x)
    result = gaussian(x, mu, sig)
    expected = np.array([0.13533528, 0.60653066, 1, 0.60653066, 0.13533528])
    ε = max(result - expected)
    assert ε<1e-8, result


def test__scale_dataframe_standard():
    N = 100
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    scaled = scale_dataframe(df, scaler='standard')
    ε = 1e-8
    means_are_zero = scaled.mean() < ε
    ε = 1e-2
    stds_are_one = scaled.std() - 1 < ε
    assert all(means_are_zero), scaled.mean()
    assert all(stds_are_one), scaled.std()


def test__scale_dataframe_robust():
    N = 100
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    scaled = scale_dataframe(df, scaler='robust')
    ε = 1e-8
    medians_are_zero = scaled.median() < ε
    assert all(medians_are_zero), medians_are_zero

