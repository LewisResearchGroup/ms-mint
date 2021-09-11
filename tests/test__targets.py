import pandas as pd
from ms_mint import targets as T
from ms_mint.standards import TARGETS_COLUMNS

def test__read_targets():
    pass


def test__standardize_targets():
    targets = pd.DataFrame({
        'peakLabel': ['A'],
        'peakMz': [100],
        'rtmin': [1],
        'rtmax': [2],
        'peaklist': ['TEST']
    })
    
    expected = pd.DataFrame(
        {'peak_label': {0: 'A'},
         'mz_mean': {0: 100},
         'mz_width': {0: 10},
         'rt_min': {0: 1},
         'rt_max': {0: 2},
         'rt': {0: None},
         'intensity_threshold': {0: 0},
         'target_filename': {0: 'TEST'}}, index=range(0,1)
    )[TARGETS_COLUMNS]

    result = T.standardize_targets(targets)

    print(expected == result)

    assert result.equals(expected), result


def test__check_target():
    pass


def test__peak_window_from_target():
    pass


def test__gen_target_grid():
    target = T.gen_target_grid(
        [115], .1, intensity_threshold=1000
    )
    assert target is not None


def test__read_target__compound_formula():
    fn = 'tests/data/targets/compound-formula.csv'
    T.read_targets(fn)
