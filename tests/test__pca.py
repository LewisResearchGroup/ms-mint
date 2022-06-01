from ms_mint.Mint import Mint


def test__run_pca():
    mint = Mint(verbose=True)
    mint.load("tests/data/results/example_results.csv")
    print(mint.ms_files)
    print(mint.targets)
    print(mint.results)
    print(mint.results.columns)
    mint.pca()
