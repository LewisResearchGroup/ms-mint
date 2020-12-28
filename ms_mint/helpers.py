# ms_mint/helpers.py

import os

def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def sort_columns_by_median(df):
    cols = df.median().sort_values(ascending=False).index
    return df[cols]


def remove_all_zero_columns(df):
    is_zero = df.max() != 0
    is_zero = is_zero[is_zero].index
    return df[is_zero]


def is_ms_file(fn):
    if (fn.lower().endswith('.mzxml')) or \
       (fn.lower().endswith('.mzml'))  or \
       (fn.lower().endswith('.mzhdf')) or \
       (fn.lower().endswith('.feather')):
        return True
    else:
        return False

def get_ms_files_from_results(results):
    ms_files = results[['ms_path', 'ms_file']].drop_duplicates()
    ms_files = [os.path.join(ms_path, ms_file) for ms_path, ms_file in ms_files.values]
    return ms_files