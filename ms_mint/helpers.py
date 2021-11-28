# ms_mint/helpers.py

import os


def df_diff(df1, df2, which="both"):
    """
    Find rows which are different between two dataframes.
    """
    _df = df1.merge(df2, indicator=True, how="outer")
    diff_df = _df[_df["_merge"] != which]
    return diff_df.reset_index(drop=True)


def is_ms_file(fn):
    if (
        (fn.lower().endswith(".mzxml"))
        or (fn.lower().endswith(".mzml"))
        or (fn.lower().endswith(".mzmlb"))
        or (fn.lower().endswith(".mzhdf"))
        or (fn.lower().endswith(".raw"))
        or (fn.lower().endswith(".parquet"))
        or (fn.lower().endswith(".feather"))
    ):
        return True
    else:
        return False


def get_ms_files_from_results(results):
    ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files
