import pandas as pd
import numpy as np
import pytest

from plotly.graph_objs._figure import Figure
from ms_mint.plotly_tools import (
    set_template,
    plotly_heatmap,
)

@pytest.mark.parametrize("transposed,normed_by_cols,correlation,clustered,add_dendrogram", [
    (False, False, False, False, False),
    (True, False, False, False, False),
    (False, True, False, False, False),
    (False, False, True, False, False),
    (False, False, False, True, True),
    (False, False, True, True, False),
])
def test__plotly_heatmap(transposed, normed_by_cols, correlation, clustered, add_dendrogram):
    N = 10
    data = np.random.uniform(size=(N, N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, transposed=transposed, normed_by_cols=normed_by_cols, correlation=correlation, clustered=clustered, add_dendrogram=add_dendrogram)
    assert isinstance(img, Figure), type(img)

def test__set_template():
    set_template()
    assert True