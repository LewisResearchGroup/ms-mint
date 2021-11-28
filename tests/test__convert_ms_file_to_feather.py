import pandas as pd
from pathlib import Path as P
from ms_mint.io import convert_ms_file_to_feather, ms_file_to_df

import paths
from paths import TEST_MZML, TEST_MZXML

print(paths.__file__)


def test__read_mzXML(tmp_path):
    df = ms_file_to_df(TEST_MZXML)
    assert isinstance(df, pd.DataFrame), df


def test_convert_mzxml_to_feather(tmp_path):
    fn_out = P(TEST_MZXML).with_suffix(".feather")
    convert_ms_file_to_feather(TEST_MZXML, fn_out)
    assert fn_out.is_file()


def test_convert_mzml_to_feather(tmp_path):
    fn_out = P(TEST_MZML).with_suffix(".feather")
    convert_ms_file_to_feather(TEST_MZML, fn_out)
    assert fn_out.is_file()
