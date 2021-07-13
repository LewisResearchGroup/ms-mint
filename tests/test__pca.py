from ms_mint.Mint import Mint



def test__run_pca():
    mint = Mint()
    mint.load('tests/data/results/example_results.csv')
    mint.pca()

