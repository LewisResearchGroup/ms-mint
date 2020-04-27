from ms_mint.Mint import Mint

def test__run_skips_broken_files():
    mint = Mint()
    mint.peaklist_files = 'tests/data/peaklist_v0.csv'
    mint.files = ['tests/data/test.mzXML', 
                  'tests/data/test-broken.mzXML']
    mint.run()
    
    broken_files_absent = all(mint.results.ms_file != 'test-broken.mzXML')
    good_files_present = all(mint.results.ms_file != 'test.mzXML')
    
    assert broken_files_absent, mint.results.ms_file
    assert good_files_present, mint.results.ms_file

