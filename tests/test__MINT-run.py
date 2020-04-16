from ms_mint import Mint

def test__run_skips_broken_files():
    mint = Mint()
    mint.peaklist_files = 'tests/data/peaklist_v0.csv'
    mint.files = ['tests/data/test.mzXML', 'tests/data/test-broken.mzXML']
    mint.run()
    print(mint.results)
    assert True == False, mint.results
