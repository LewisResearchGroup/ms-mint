import os
from ms_mint.Mint import Mint

def test__export_to_excel(tmp_path):
    filename = os.path.join(tmp_path, 'output.xlsx')
    mint = Mint(verbose=True)
    mint.files = 'tests/data/test.mzXML'
    mint.run()
    mint.export(filename)
    assert os.path.isfile(filename)
