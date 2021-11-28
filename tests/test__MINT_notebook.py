from ms_mint.notebook import Mint

mint = Mint(verbose=True)


class TestClass:
    def test__mint_n_files(self):
        ms_files = [
            "tests/data/ms_files/fileA.mzXML",
            "tests/data/ms_files/fileB.mzxml",
            "tests/data/ms_files/fileC.mzML",
            "tests/data/ms_files/fileD.mzml",
        ]
        mint.ms_files = ms_files
        result = mint.n_files
        expect = len(ms_files)
        assert result == expect, f"Expected ({expect}) != result ({result})"
