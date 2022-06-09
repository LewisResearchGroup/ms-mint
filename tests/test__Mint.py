from ms_mint.Mint import Mint

from paths import TEST_TARGETS_FN_V2_CSV_SEC, TEST_MZML, TEST_MZXML


def test__switch_verbosity_off():
    mint = Mint(verbose=True)
    assert mint.verbose == True
    mint.verbose = False
    assert mint.verbose == False


def test__switch_verbosity_on():
    mint = Mint(verbose=False)
    assert mint.verbose == False
    mint.verbose = True
    assert mint.verbose == True


def test__clear_results():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    mint.run()
    assert len(mint.results) > 0
    mint.clear_results()
    assert len(mint.results) == 0


def test__clear_ms_files():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    assert len(mint.ms_files) == 1
    mint.clear_ms_files()
    assert len(mint.ms_files) == 0


def test__clear_targets():
    mint = Mint(verbose=False)
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    assert len(mint.targets) > 0
    mint.clear_targets()
    assert len(mint.targets) == 0


def test__heatmap():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    mint.run()
    mint.plot.heatmap()
    mint.plot.heatmap(transposed=True)


def test__hierarchical_clustering():
    mint = Mint(verbose=False)
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    mint.run()
    mint.plot.hierarchical_clustering()
    mint.plot.hierarchical_clustering(transposed=True)
    mint.plot.hierarchical_clustering(transform_func="log1p")
    mint.plot.hierarchical_clustering(transform_func="log2p1")
    mint.plot.hierarchical_clustering(transform_func="log10p1")
    mint.plot.hierarchical_clustering(transform_func="log10p1")
    mint.plot.hierarchical_clustering(transform_filenames_func=lambda x: x[:3])


def test__pca_plots_are_working():
    mint = Mint(verbose=False)
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    mint.run()
    mint.pca.run(fillna="mean")
    mint.pca.run(fillna="zero")
    mint.pca.run(fillna="median")
    mint.pca.plot.cumulative_variance()
    mint.pca.plot.pairplot()


def test__progress_callback():
    result = []
    callback_func = lambda x: result.append(True)
    mint = Mint(progress_callback=callback_func)
    mint.ms_files = TEST_MZML
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    mint.run()
    expected = [True, True]
    assert result == expected


def test__progress():
    mint = Mint()
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.load_targets(TEST_TARGETS_FN_V2_CSV_SEC)
    assert mint.progress == 0
    mint.run()
    print(mint.progress, type(mint.progress))
    assert mint.progress == 100
