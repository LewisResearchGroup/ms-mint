from ms_mint.Mint import Mint

from paths import RESULTS_FN


def test__run_pca():
    mint = Mint(verbose=False)
    mint.load(RESULTS_FN)
    mint.pca.run()
    assert mint.pca.results is not None