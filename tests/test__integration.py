import pandas as pd
import matplotlib.pyplot as plt

from ms_mint.Mint import Mint
from ms_mint.standards import MINT_RESULTS_COLUMNS

from paths import (
    TEST_MZML,
    TEST_MZXML,
    TEST_TARGETS_FN_V2_CSV_SEC,
    TEST_TARGETS_FN_V0,
    TEST_TARGETS_FN_V1,
)


mint = Mint(verbose=True)
mint_b = Mint(verbose=True)


class TestClass:
    def run_without_files(self):
        mint.run()

    def test__add_experimental_data(self):
        assert mint.n_files == 0, mint.n_files
        mint.ms_files = [TEST_MZXML]
        assert mint.n_files == 1, mint.n_files
        assert mint.ms_files == [TEST_MZXML]

    def test__mint_run_standard(self):
        mint.run()

    def test__results_is_dataframe(self):
        results = mint.results
        assert isinstance(results, pd.DataFrame), "Results is not a DataFrame"

    def test__results_lenght(self):
        actual = len(mint.results)
        expected = len(mint.targets) * len(mint.ms_files)
        assert (
            expected == actual
        ), f"Length of results ({actual}) does not equal expected length ({expected})"

    def test__results_columns(self):
        expected = MINT_RESULTS_COLUMNS
        actual = mint.results.columns
        assert (expected == actual).all(), actual

    def test__crosstab_is_dataframe(self):
        ct = mint.crosstab()
        assert isinstance(
            ct, pd.DataFrame
        ), f"Crosstab is not a DataFrame ({type(ct)})."

    def test__mint_run_parallel(self):
        mint.ms_files = [TEST_MZML, TEST_MZXML]
        mint.targets_files = TEST_TARGETS_FN_V2_CSV_SEC
        mint.run(nthreads=2)

    def test__mzxml_equals_mzml(self):
        mint1 = Mint()
        mint2 = Mint()

        mint1.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
        mint1.ms_files = [TEST_MZML]

        mint2.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
        mint2.ms_files = [TEST_MZXML]

        assert mint1.results.equals(mint2.results)

    def test__target_v0_equals_v1_results(self):
        mint1 = Mint()
        mint2 = Mint()

        print(TEST_TARGETS_FN_V0)
        print(TEST_TARGETS_FN_V1)

        mint1.load_targets(TEST_TARGETS_FN_V0)
        mint1.ms_files = [TEST_MZXML]

        mint2.load_targets(TEST_TARGETS_FN_V1)
        mint2.ms_files = [TEST_MZXML]

        assert mint1.results.equals(mint2.results)

    def test__status(self):
        mint.status == "waiting"

    def test__run_returns_none_without_target(self):
        mint.reset()
        mint.ms_files = [TEST_MZXML]
        assert mint.run() is None

    def test__run_returns_none_without_ms_files(self):
        mint.reset()
        mint.targets_files = TEST_TARGETS_FN_V0
        assert mint.run() is None


def test__optimize_targets_find_min_max():

    mint = Mint()
    mint.ms_files = [TEST_MZXML]
    mint.load_targets(TEST_TARGETS_FN_V0)

    mint.targets.rt = mint.targets[['rt_min', 'rt_max']].mean(axis=1)

    mint.targets['rt_min'] = None
    mint.targets['rt_max'] = None

    result = mint.opt.find_rt_min_max()

    (mint.targets.filter(regex='rt'))
    
    n_missing_values_in_rt_cols = mint.targets[['rt_min', 'rt_max']].isna().sum().sum()

    assert n_missing_values_in_rt_cols== 0, mint.targets.filter(regex='rt')
    assert mint.opt is result


def test__optimize_targets_find_min_max__returns_figure():

    mint = Mint()
    mint.ms_files = [TEST_MZXML]
    mint.load_targets(TEST_TARGETS_FN_V0)

    mint.targets.rt = mint.targets[['rt_min', 'rt_max']].mean(axis=1)

    result_1, result_2 = mint.opt.find_rt_min_max(plot=True)

    (mint.targets.filter(regex='rt'))
    
    assert mint.opt is result_1
    assert isinstance(result_2, plt.Figure), type(result_2)



