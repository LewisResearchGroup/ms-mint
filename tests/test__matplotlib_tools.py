import numpy as np
import pandas as pd

from ms_mint.matplotlib_tools import hierarchical_clustering


def test__plotly_heatmap__call_show():
    N = 10
    data = np.random.uniform(size=(N, N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    df.index.name = "INDEX"
    df.columns.name = "COLUMNS"

    clustered, fig, ndx_leaves, col_leaves = hierarchical_clustering(df)

    assert True