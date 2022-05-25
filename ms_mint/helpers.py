# ms_mint/helpers.py

import os


def df_diff(df1, df2, which="both"):
    """_summary_

    :param df1: _description_
    :type df1: _type_
    :param df2: _description_
    :type df2: _type_
    :param which: _description_, defaults to "both"
    :type which: str, optional
    :return: _description_
    :rtype: _type_
    """
    _df = df1.merge(df2, indicator=True, how="outer")
    diff_df = _df[_df["_merge"] != which]
    return diff_df.reset_index(drop=True)


def is_ms_file(fn):
    """_summary_

    :param fn: _description_
    :type fn: function
    :return: _description_
    :rtype: _type_
    """
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
    """_summary_

    :param results: _description_
    :type results: _type_
    :return: _description_
    :rtype: _type_
    """
    ms_files = results[["ms_path", "ms_file"]].drop_duplicates()
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files
