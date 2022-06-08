from ms_mint.Mint import Mint


def test__run_pca():
    mint = Mint(verbose=False)
    mint.load("tests/data/results/example_results.csv")
    mint.pca.run()
    assert mint.pca.results is not None