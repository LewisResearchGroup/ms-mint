import os
import pandas as pd
import numpy as np

from pathlib import Path as P
from multiprocessing import Pool
from ms_mint import processing
from ms_mint.io import ms_file_to_df
from ms_mint.standards import MINT_RESULTS_COLUMNS
from paths import TEST_MZXML


def test__find_test_mzxml():
    assert os.path.isfile(TEST_MZXML), "Test mzXML ({}) not found.".format(TEST_MZXML)


def test__process_ms1():
    ms_data = pd.DataFrame(
        {"scan_time": [1, 2, 3], "mz": [100, 200, 300], "intensity": [2, 3, 7]}
    )

    targets = pd.DataFrame(
        {
            "peak_label": ["A"],
            "mz_mean": [200],
            "mz_width": [10],
            "intensity_threshold": [0],
            "rt_min": [0],
            "rt_max": [10],
            "targets_filename": ["unknown"],
        }
    )

    result = processing.process_ms1(ms_data, targets)

    expect = pd.DataFrame(
        {
            "peak_label": {0: "A"},
            "mz_mean": {0: 200},
            "mz_width": {0: 10},
            "intensity_threshold": {0: 0},
            "rt_min": {0: 0},
            "rt_max": {0: 10},
            "targets_filename": {0: "unknown"},
            "peak_area": {0: 3},
            "peak_area_top3": {
                0: 1
            },  # The only value in the extraction window is 3, therefore the expected value is 1
            "peak_n_datapoints": {0: 1},
            "peak_max": {0: 3},
            "peak_rt_of_max": {0: 2},
            "peak_min": {0: 3},
            "peak_median": {0: 3.0},
            "peak_mean": {0: 3.0},
            "peak_delta_int": {0: 0},
            "peak_shape_rt": {0: "2"},
            "peak_shape_int": {0: "3"},
            "peak_mass_diff_25pc": {0: 0.0},
            "peak_mass_diff_50pc": {0: 0.0},
            "peak_mass_diff_75pc": {0: 0.0},
            "peak_score": {0: None},
        }
    )

    print("Result:")
    print(result.T)

    assert result.equals(expect), result


def test__process_ms1_from_df():
    df = pd.DataFrame(
        {"scan_time": [1, 2, 3], "mz": [100, 200, 300], "intensity": [2, 3, 7]}
    )
    peaklist = pd.DataFrame(
        {
            "peak_label": ["A"],
            "mz_mean": [200],
            "mz_width": [10],
            "intensity_threshold": [0],
            "rt_min": [0],
            "rt_max": [10],
        }
    )
    result = processing._process_ms1_from_df_(df, peaklist)
    expect = [["A", 3, 1, 1, 3, 2, 3, 3.0, 3.0, 0, "2", "3", 0.0, 0.0, 0.0, None]]
    print(result)
    print(expect)
    assert result == expect


def test__slice_ms1_array():
    array = np.array([[1, 100, 2], [2, 200, 3], [3, 300, 7]])
    result = processing.slice_ms1_array(
        array, rt_min=1.5, rt_max=2.5, mz_mean=200, mz_width=10, intensity_threshold=0
    )
    expect = np.array([[2, 200, 3]])
    print(expect)
    print(result)
    assert np.array_equal(result, expect)


def test_process_ms1_from_numpy():
    """
    Test the processing of MS1 data from a numpy array.

    The function processes an array of MS1 data and returns a list of
    tuples containing the peak label, maximum intensity, apex RT, start
    RT, end RT, intensity at start RT, intensity at end RT, mz, and
    other fields.

    :return: None
    :rtype: None
    """

    array = np.array([[1, 100, 2], [2, 200, 3], [3, 300, 7]])

    targets = [(100, 10, 0.5, 1.5, 0, "A"), (200, 10, 1.5, 2.5, 0, "B")]

    result = processing.process_ms1_from_numpy(array, targets)

    expect = [
        ["A", 2, 0, 1, 2, 1, 2, 2.0, 2.0, 0, "1", "2", 0.0, 0.0, 0.0, None],
        ["B", 3, 1, 1, 3, 2, 3, 3.0, 3.0, 0, "2", "3", 0.0, 0.0, 0.0, None],
    ]

    print("Expected:")
    _ = [print(i) for i in expect]
    print("Actual")
    _ = [print(i) for i in result]
    assert result == expect


def test__extract_chromatogram_from_ms1():
    df = pd.DataFrame(
        {"scan_time": [1, 2, 3], "mz": [100, 200, 300], "intensity": [2, 3, 7]}
    )
    result = processing.extract_chromatogram_from_ms1(df, 200, 10)
    expected = pd.Series([3], index=[2])
    assert result.equals(expected)


def test_run_parallel(tmp_path):
    """
    Test running parallel processing of multiple MS files and storing the results in a CSV file.

    :param tmp_path: A temporary directory created by pytest.
    :type tmp_path: py.path.local
    """

    # Define test targets
    targets = pd.DataFrame(
        {
            "peak_label": ["A"],
            "mz_mean": [128.034],
            "mz_width": [100],
            "intensity_threshold": [0],
            "rt_min": [0],
            "rt_max": [700],
            "rt": [150],
            "rt_unit": ["s"],
            "target_filename": ["unknown"],
        }
    )

    # Initialize multiprocessing pool
    pool = Pool(processes=2, maxtasksperchild=None)

    # Generate test MS files
    n_files = 4
    fns = [P(tmp_path) / f"File-{i}.mzXML" for i in range(n_files)]
    for fn in fns:
        os.symlink(TEST_MZXML, fn)

    # Generate list of arguments for parallel processing
    args_list = []
    for fn in fns:
        args = {
            "filename": fn,
            "targets": targets,
            "q": None,
            "mode": None,
            "output_fn": None,
        }
        args_list.append(args)

    # Run parallel processing
    results = pool.map_async(processing.process_ms1_files_in_parallel, args_list)
    pool.close()
    pool.join()

    # Concatenate results
    result = pd.concat(results.get())

    # Print info for debugging
    print(targets)
    print(fns)
    print(result)
    print(result.shape)
    print(ms_file_to_df(fn))

    # Assert that the results have the expected length
    assert len(result) == n_files * len(targets)


def test__run_parallel_with_output_filename(tmp_path):
    """
    Test for running parallel processing of multiple MS files and storing the results in a CSV file.

    :param tmp_path: A temporary directory created by pytest.
    :type tmp_path: py.path.local
    """

    # Create targets DataFrame
    targets = pd.DataFrame(
        {
            "peak_label": ["A"],
            "mz_mean": [128.034],
            "mz_width": [100],
            "intensity_threshold": [0],
            "rt_min": [286],
            "rt_max": [330],
            "rt": [300],
            "rt_unit": "s",
            "target_filename": ["unknown"],
        }
    )

    # Set up multiprocessing pool
    pool = Pool(processes=2, maxtasksperchild=None)

    N = 4
    fns = [P(tmp_path) / f"File-{i}.mzXML" for i in range(N)]
    output_fn = P(tmp_path) / "results.csv"

    # Create argument list and symlink test MS files
    args_list = []
    for fn in fns:
        os.symlink(TEST_MZXML, fn)

        args = {
            "filename": fn,
            "targets": targets,
            "q": None,
            "mode": None,
            "output_fn": output_fn,
        }

        # Print some info for debugging
        print(fn)
        print(ms_file_to_df(fn).head())
        print(ms_file_to_df(fn).tail())
        print("*" * 80)
        args_list.append(args)

    # Prepare output file (only headers)
    pd.DataFrame(columns=MINT_RESULTS_COLUMNS).to_csv(output_fn, index=False)

    # Run async processing
    results = pool.map_async(processing.process_ms1_files_in_parallel, args_list)

    pool.close()
    pool.join()

    returned_results = results.get()

    print(returned_results)

    # Expect all returned results to be None
    assert all(e is None for e in returned_results), returned_results

    # All results should be stored in the output file
    assert output_fn.is_file()

    stored_results = pd.read_csv(output_fn)
    print("=" * 80)
    print(output_fn)
    print(stored_results)
    assert len(stored_results) == N * len(targets)
