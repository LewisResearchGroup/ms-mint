import pandas as pd
import shutil
import os
import io

from ms_mint.Mint import Mint

from pathlib import Path as P

from ms_mint.io import (
    ms_file_to_df,
    mzml_to_pandas_df_pyteomics,
    convert_ms_file_to_feather,
    convert_ms_file_to_parquet,
    MZMLB_AVAILABLE,
)

from paths import (
    TEST_MZML,
    TEST_MZXML,
    TEST_PARQUET,
    TEST_MZMLB_POS,
    TEST_MZML_POS,
    TEST_MZML_NEG,
)


def test__ms_file_to_df__mzML():
    result = ms_file_to_df(TEST_MZML)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns


def test__ms_file_to_df__mzML_timeunit_minutes():
    result = ms_file_to_df(TEST_MZML, time_unit="minutes")
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns


def test__ms_file_to_df__mzXML():
    result = ms_file_to_df(TEST_MZXML)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns


def test__mzml_to_pandas_df_pyteomics_pos():
    result = mzml_to_pandas_df_pyteomics(TEST_MZML_POS)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns
    assert all(result.polarity == "+"), f'Polarity should be "+"\n{result}'


def test__mzml_to_pandas_df_pyteomics_neg():
    result = mzml_to_pandas_df_pyteomics(TEST_MZML_NEG)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns
    assert all(result.polarity == "-"), f'Polarity should be "-"\n{result}'


def test__read_parquet():
    result = ms_file_to_df(TEST_PARQUET)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns


def test__write_read_hdf(tmpdir):
    df = ms_file_to_df(TEST_PARQUET)
    fn = P(tmpdir) / "file.hdf"
    df.to_hdf(fn, key="data")
    result = ms_file_to_df(fn)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns


def test__read_mzMLb(tmpdir):
    if not MZMLB_AVAILABLE:
        return None
    result = ms_file_to_df(TEST_MZMLB_POS)
    expected_cols = [
        "scan_id",
        "ms_level",
        "polarity",
        "scan_time",
        "mz",
        "intensity",
    ]
    assert isinstance(result, pd.DataFrame), f"{type(result)} is not a dataframe"
    assert expected_cols == result.columns.to_list(), result.columns
    # assert all(result.polarity == '+'), f'Polarity should be "+"\n{result}'


def test__convert_ms_file_to_feather(tmpdir):
    print(tmpdir)
    shutil.copy(TEST_MZML, tmpdir)
    fn = P(tmpdir) / P(TEST_MZML).name
    fn_out = fn.with_suffix(".feather")
    print(fn, fn_out)
    convert_ms_file_to_feather(fn)
    assert fn_out.is_file(), f"File not generated {fn_out}"
    df = ms_file_to_df(fn)
    df_fea = ms_file_to_df(fn_out)
    assert df_fea.equals(df), "DataFrames not equal"


def test__convert_ms_file_to_parquet(tmpdir):
    print(tmpdir)
    shutil.copy(TEST_MZML, tmpdir)
    fn = P(tmpdir) / P(TEST_MZML).name
    fn_out = fn.with_suffix(".parquet")
    print(fn, fn_out)
    convert_ms_file_to_parquet(fn)
    assert fn_out.is_file(), f"File not generated {fn_out}"
    df = ms_file_to_df(fn)
    df_fea = ms_file_to_df(fn_out)
    assert df_fea.equals(df), "DataFrames not equal"


def test__export_to_excel(tmp_path):
    filename = os.path.join(tmp_path, "output.xlsx")
    mint = Mint(verbose=True)
    mint.ms_files = "tests/data/test.mzXML"
    mint.run()
    mint.export(filename)
    assert os.path.isfile(filename)


def test__export_to_excel_without_fn():
    mint = Mint(verbose=True)

    data = pd.DataFrame(
        {
            "peak_label": ["2-DEOXYADENOSINE"],
            "mz_mean": [250.094559],
            "mz_width": [10],
            "intensity_threshold": [0],
            "rt": [320],
            "rt_min": [300],
            "rt_max": [337],
            "rt_units": ["s"],
            "targets_filename": ["unknown"],
        }
    )

    mint.results = data

    buffer = mint.export()
    assert isinstance(buffer, io.BytesIO)
    result = pd.read_excel(buffer, sheet_name="Results")
    assert data.equals(result)
