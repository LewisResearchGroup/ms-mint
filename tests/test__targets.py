import pandas as pd
from ms_mint.targets import (
    standardize_targets,
    check_targets,
    read_targets,
    gen_target_grid,
    diff_targets,
)
from ms_mint.standards import TARGETS_COLUMNS
from ms_mint import Mint

from paths import (
    #TEST_TARGETS_FN_V2_CSV_SEC,
    TEST_TARGETS_FN_V0_XLSX,
    TEST_TARGETS_FN_V0,
    TEST_TARGETS_FN_V1,
)


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
            "rt_min": {0: 1 * 60},
            "rt_max": {0: 2 * 60},
            "rt": {0: None},
            "rt_unit": {0: "s"},
            "intensity_threshold": {0: 0},
            "target_filename": {0: "TEST"},
        },
        index=range(0, 1),
    )[TARGETS_COLUMNS]

    result = standardize_targets(targets)

    print(expected.T)

    print(result.T)

    assert result.equals(expected), result


def test__check_target__empty_list_ok():
    emtpy_target_list = pd.DataFrame(columns=TARGETS_COLUMNS)
    result = check_targets(emtpy_target_list)
    assert result is True


def test__check_targets_labels_not_string():
    targets = read_targets(TEST_TARGETS_FN_V1)
    targets["peak_label"] = range(len(targets))
    result = check_targets(targets)
    assert result is False


def test__check_targets_labels_duplictated():
    targets = read_targets(TEST_TARGETS_FN_V1)
    targets.loc[0, "peak_label"] = "A"
    targets.loc[1, "peak_label"] = "A"
    result = check_targets(targets)
    assert result is False


def test__check_targets_wrong_column_names():
    targets = read_targets(TEST_TARGETS_FN_V1).rename(
        columns={"peak_label": "paek_label"}
    )
    targets.loc[0, "peak_label"] = "A"
    targets.loc[1, "peak_label"] = "A"
    result = check_targets(targets)
    assert result is False


def test__import_csv_and_xlsx():
    mint1 = Mint()
    mint2 = Mint()

    mint1.load_targets(TEST_TARGETS_FN_V0)
    mint2.load_targets(TEST_TARGETS_FN_V0_XLSX)

    mint1.targets.drop("target_filename", inplace=True, axis=1)
    mint2.targets.drop("target_filename", inplace=True, axis=1)

    print(mint1.targets)
    print(mint2.targets)

    assert mint1.targets.equals(mint2.targets)


def test__target_grid():
    targets = gen_target_grid([1, 2], dt=1, rt_max=1)

    assert all(targets.mz_mean == [1, 1, 2, 2])
    assert all(targets.rt_min == [0, 1, 0, 1])
    assert all(targets.rt_max == [1, 2, 1, 2])


def test__diff_targets():
    targets1 = read_targets(TEST_TARGETS_FN_V0)
    targets2 = targets1.copy()

    new_value = 10.23434
    selected_ndx = 1
    targets2.loc[selected_ndx, "rt_max"] = new_value

    result = diff_targets(targets1, targets2)

    print(result)

    assert len(result) == 1
    assert (
        result.at[selected_ndx, "peak_label"] == targets1.at[selected_ndx, "peak_label"]
    )
    assert result.at[selected_ndx, "rt_min"] == targets1.at[selected_ndx, "rt_min"]
    assert result.at[selected_ndx, "rt_max"] == new_value
