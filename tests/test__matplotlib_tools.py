import seaborn as sns

import matplotlib as mpl

from ms_mint import Mint

from paths import RESULTS_FN



def test__hierarchical_clustering():
    mint = Mint()
    mint.load(RESULTS_FN)
    fig = mint.plot.hierarchical_clustering()
    assert isinstance(fig, mpl.image.AxesImage), type(fig)    


def test__plot_peak_shapes():
    mint = Mint()
    mint.load(RESULTS_FN)
    fig = mint.plot.peak_shapes()
    assert isinstance(fig, sns.axisgrid.FacetGrid), type(fig)