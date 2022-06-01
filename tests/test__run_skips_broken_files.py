from os.path import basename

from ms_mint.Mint import Mint

from paths import TEST_MZML, TEST_TARGETS_FN_V2_CSV_SEC, TEST_MZXML_BROKEN


def test__run_skips_broken_files():
    mint = Mint(verbose=True)
    mint.targets_files = TEST_TARGETS_FN_V2_CSV_SEC
    mint.ms_files = [TEST_MZML, TEST_MZXML_BROKEN]

    mint.run()

    broken_files_absent = all(mint.results.ms_file != basename(TEST_MZXML_BROKEN))
    good_files_present = all(mint.results.ms_file == basename(TEST_MZML))

    assert broken_files_absent, mint.results.ms_file
    assert good_files_present, mint.results.ms_file
