
import pandas as pd

from ms_mint.Mint import Mint

from paths import TEST_TARGETS_FN, TEST_MZML, TEST_MZXML


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


def test__read_ms_file():
    mint = Mint(verbose=False)
    result = mint.ms_file_to_df(TEST_MZML)
    assert isinstance(result, pd.DataFrame)


def test__clear_results():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.targets_files = TEST_TARGETS_FN
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


def test__clear_results():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    assert len(mint.results) > 0
    mint.clear_results()
    assert len(mint.results) == 0


def test__clear_targets():
    mint = Mint(verbose=False)
    mint.targets_files = TEST_TARGETS_FN
    assert len(mint.targets) > 0
    mint.clear_targets()
    assert len(mint.targets) == 0


def test__optimize_rt():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.targets_files = TEST_TARGETS_FN
    mint.optimize_rt()


def test__heatmap():
    mint = Mint(verbose=False)
    mint.ms_files = TEST_MZML
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    mint.plot.heatmap()
    mint.plot.heatmap(transposed=True)


def test__hierarchical_clustering():
    mint = Mint(verbose=False)
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    mint.plot.hierarchical_clustering()
    mint.plot.hierarchical_clustering(transposed=True)
    mint.plot.hierarchical_clustering(transform_func='log1p')
    mint.plot.hierarchical_clustering(transform_func='log2p1')
    mint.plot.hierarchical_clustering(transform_func='log10p1')
    mint.plot.hierarchical_clustering(transform_func='log10p1')
    mint.plot.hierarchical_clustering(transform_filenames_func=lambda x: x[:3])
 

def test__pca_plots_are_running():
    mint = Mint(verbose=False)
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    mint.pca(fillna='mean')
    mint.pca(fillna='zero')
    mint.pca(fillna='median')
    mint.plot.pca_cumulative_variance()
    mint.plot.pca_scatter_matrix(color_groups='peak_label')


def test__plot_peak_shapes():
    mint = Mint(verbose=False)
    mint.ms_files = [TEST_MZML]
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    mint.plot.peak_shapes()



def test__progress_callback():
    result = []
    callback_func = lambda x: result.append(True)
    mint = Mint(progress_callback=callback_func)
    mint.ms_files = TEST_MZML
    mint.targets_files = TEST_TARGETS_FN
    mint.run()
    expected = [True, True]
    assert result == expected


def test__progress():
    mint = Mint()
    mint.ms_files = [TEST_MZML, TEST_MZXML]
    mint.targets_files = TEST_TARGETS_FN    
    assert mint.progress == 0
    mint.run()
    print(mint.progress, type(mint.progress))
    assert mint.progress == 100