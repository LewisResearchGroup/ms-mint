import pandas as pd
import numpy as np


from plotly.graph_objs._figure import Figure

from ms_mint import Mint
from ms_mint.vis.plotly import set_template, plotly_heatmap, plotly_peak_shapes, plotly_peak_shapes_3d

from paths import TEST_FEATHER, TEST_TARGETS_FN


def test__plotly_heatmap():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df)
    assert isinstance(img, Figure), type(img)


def test__plotly_heatmap__transposed():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, transposed=True)
    assert isinstance(img, Figure), type(img)


def test__plotly_heatmap__normed_by_cols():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, normed_by_cols=True)
    assert isinstance(img, Figure), type(img)


def test__plotly_heatmap__correlation():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, correlation=True)
    assert isinstance(img, Figure), type(img)


def test__plotly_heatmap__clustered_with_dendrogram():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, clustered=True, add_dendrogram=True)
    assert isinstance(img, Figure), type(img)


def test__plotly_heatmap__clustered_correlation():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    img = plotly_heatmap(df, clustered=True, add_dendrogram=False, correlation=True)
    assert isinstance(img, Figure), type(img)

'''
def test__plotly_heatmap__call_show():
    N = 10
    data = np.random.uniform(size=(N,N)) + np.arange(N) - N / 2
    df = pd.DataFrame(data)
    df.index.name = 'INDEX'
    df.columns.name = 'COLUMNS'
    img = plotly_heatmap(df, call_show=True, name='TEST')
    assert img is None, type(img)
'''

def test__plotly_peak_shapes():
    mint = Mint()
    mint.ms_files = TEST_FEATHER
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    img = plotly_peak_shapes(mint.results)
    assert isinstance(img, Figure), type(img)


def test__plotly_peak_shapes_3d():
    mint = Mint()
    mint.ms_files = TEST_FEATHER
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    img = plotly_peak_shapes_3d(mint.results, peak_label='11')
    assert isinstance(img, Figure), type(img)


def test__set_template():
    set_template()
    assert True