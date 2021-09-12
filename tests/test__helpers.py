import pandas as pd

from ms_mint.helpers import df_diff, is_ms_file


def test__df_diff():

    df1 = pd.DataFrame({'A':[1, 1, 0, 0, 2], 
                        'B':[1, 0, 0, 1, 2]},
                        index=[1, 2, 3, 4, 5])

    df2 = pd.DataFrame({'A':[1, 1, 1, 1, 3], 
                        'B':[1, 1, 1, 1, 3]},
                        index=[1, 2, 3, 4 ,6])

    result = df_diff(df1, df2)

    _merge = pd.CategoricalIndex(['left_only']*4+['right_only'],
                 categories=['left_only', 'right_only', 'both'], 
                 ordered=False, dtype='category')

    expected = pd.DataFrame({'A': [1, 0, 0, 2, 3],
                             'B': [0, 0, 1, 2, 3],
                             '_merge': _merge})

    assert result.equals(expected), result.values==expected.values


def test__is_ms_file():

    pos = ['file.mzML',
           'path/file.mzML',
           'file.mzXML', 
           'path/file.mzXML', 
           'file.RAW', 
           'path/file.RAW',
           'file.mzMLb',
           'file.mzhdf',
           'file.parquet']

    neg = ['file.txt', 'file.png', 'file.ML']

    res_pos = [ is_ms_file(fn) for fn in pos]
    res_neg = [ is_ms_file(fn) for fn in neg]

    assert all(res_pos), res_pos
    assert not any(res_neg), res_neg

