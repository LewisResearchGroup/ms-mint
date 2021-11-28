import os
import pandas as pd
import pytest

from ms_mint.Mint import Mint

mint_a = Mint()
mint_b = Mint()
mint_c = Mint()


@pytest.fixture(scope="session")
def tmp_path(tmpdir_factory):
    path = tmpdir_factory.mktemp("data")
    return path


class _TestExportAndLoad:
    def test__load_example_data(self):
        fn = "tests/data/demo_results.xlsx"
        mint_a.load(fn)
        assert len(mint_a.results) > 0, "Could not load example data."

    def test__export_to_excel(self, tmp_path):
        fn = os.path.join(tmp_path, "results.xlsx")
        mint_a.export(fn)
        assert os.path.isfile(fn), "Output file not generated"

    def test__export_to_csv(self, tmp_path):
        fn = os.path.join(tmp_path, "results.csv")
        mint_a.export(fn)
        assert os.path.isfile(fn), "Output file not generated"

    def test__load_from_excel(self, tmp_path):
        fn = os.path.join(tmp_path, "results.xlsx")
        print("Filename:", fn)
        mint_b.load(fn)

        actual_results = mint_b.results.round(8)
        expect_results = mint_a.results.round(8)
        actual_files = mint_b.ms_files
        expect_files = mint_a.ms_files

        print("Expected results:\n", expect_results)
        print("Actual results:\n", actual_results)
        print("N diff dtypes:", (actual_results.dtypes != expect_results.dtypes).sum())
        print(
            "N diff columns:", (actual_results.columns != expect_results.columns).sum()
        )
        print("N diff index:", (actual_results.columns != expect_results.columns).sum())
        print("N diff values:", (actual_results.values != expect_results).values.sum())
        print("N diff values:", (actual_results == expect_results).sum())

        assert all(
            [
                actual_results.equals(expect_results),
                set(actual_files) == set(expect_files),
            ]
        ), "Loaded results differ from original data."
        mint_b.reset()

    def test__load_from_csv(self, tmp_path):
        fn = os.path.join(tmp_path, "results.csv")
        print("Filename:", fn)
        mint_c.load(fn)

        actual_results = mint_c.results.round(3)
        expect_results = mint_a.results.round(3)
        actual_files = mint_c.ms_files
        expect_files = mint_a.ms_files

        print(actual_results)
        print(expect_results)

        print(actual_files)
        print(expect_files)

        assert len(actual_results) == len(expect_results), pd.read_csv(fn).to_csv(
            "/home/swacker/test.csv"
        )

        print("Expected results:\n", expect_results)
        print("Actual results:\n", actual_results)
        print("N diff dtypes:", (actual_results.dtypes != expect_results.dtypes).sum())
        print(
            "N diff columns:", (actual_results.columns != expect_results.columns).sum()
        )
        print("N diff index:", (actual_results.columns != expect_results.columns).sum())
        print("N diff values:", (actual_results.values != expect_results).values.sum())
        print("N diff values:", (actual_results == expect_results).sum())

        ndx = actual_results.peak_rt_of_max != expect_results.peak_rt_of_max
        ndx = ndx[ndx].index

        print(ndx)
        print(actual_results.loc[ndx])
        print(expect_results.loc[ndx])

        print("Comparison:")
        print("Expected results:\n", expect_results.loc[ndx, "peak_rt_of_max"])
        print("Actual results:\n", actual_results.loc[ndx, "peak_rt_of_max"])

        print(
            expect_results.loc[ndx, "peak_rt_of_max"]
            == actual_results.loc[ndx, "peak_rt_of_max"]
        )
        print(
            (
                expect_results.loc[ndx, "peak_rt_of_max"]
                == actual_results.loc[ndx, "peak_rt_of_max"]
            ).sum()
        )

        assert all(
            [
                actual_results.equals(expect_results),
                set(actual_files) == set(expect_files),
            ]
        ), "Loaded results differ from original data."
