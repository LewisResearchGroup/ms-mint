import pandas as pd
from ms_mint import targets as T
from ms_mint.standards import TARGETS_COLUMNS

from paths import TEST_TARGETS_FN

def test__read_targets():
    pass


def test__standardize_targets():
    targets = pd.DataFrame(
        {
            "peakLabel": ["A"],
            "peakMz": [100],
            "rtmin": [1],
            "rtmax": [2],
            "peaklist": ["TEST"],
        }
    )

    expected = pd.DataFrame(
        {
            "peak_label": {0: "A"},
            "mz_mean": {0: 100},
            "mz_width": {0: 10},
            "rt_min": {0: 1},
            "rt_max": {0: 2},
            "rt": {0: None},
            "intensity_threshold": {0: 0},
            "target_filename": {0: "TEST"},
        },
        index=range(0, 1),
    )[TARGETS_COLUMNS]

    result = T.standardize_targets(targets)

    print(expected == result)

    assert result.equals(expected), result


def test__check_target__empty_list_ok():
    emtpy_target_list = pd.DataFrame(columns=TARGETS_COLUMNS)
    result = T.check_targets(emtpy_target_list)
    assert result is True


def test__check_targets_labels_not_string():
    targets = T.read_targets(TEST_TARGETS_FN)
    targets['peak_label'] = range(len(targets))
    result = T.check_targets(targets)
    assert result is False


def test__check_targets_labels_missing_rtmax():
    targets = T.read_targets(TEST_TARGETS_FN)
    targets.loc[0, 'rt_max'] = None
    result = T.check_targets(targets)
    assert result is False
    

def test__check_targets_labels_missing_rtmin():
    targets = T.read_targets(TEST_TARGETS_FN)
    targets.loc[0, 'rt_min'] = None
    result = T.check_targets(targets)
    assert result is False


def test__check_targets_labels_duplictated():
    targets = T.read_targets(TEST_TARGETS_FN)
    targets.loc[0, 'peak_label'] = "A"
    targets.loc[1, 'peak_label'] = "A"
    result = T.check_targets(targets)
    assert result is False


def test__check_targets_wrong_column_names():
    targets = T.read_targets(TEST_TARGETS_FN).rename(columns={'peak_label': 'paek_label'})
    targets.loc[0, 'peak_label'] = "A"
    targets.loc[1, 'peak_label'] = "A"
    result = T.check_targets(targets)
    assert result is False    