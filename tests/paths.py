
import os
from pathlib import Path as P

from ms_mint.standards import MINT_ROOT

TEST_MZXML = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'ms_files', 'test.mzXML'
    )
)


TEST_MZML = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'ms_files', 'test.mzML'
    )
)

TEST_PEAKLIST_FN = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'peaklists', 'peaklist_v1.csv'
    )
)

TEST_PEAKLIST_FN_V0 = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'peaklists', 'peaklist_v0.csv'
    )
)


TEST_PEAK_AREA_RESULTS = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'results', 'test_peak_area_results.csv'
    )
)


TEST_MZXML_BROKEN = os.path.abspath( 
    os.path.join( 
        MINT_ROOT ,'..' ,'tests' , 'data', 'broken', 'broken.mzXML'
    )
)


assert P( TEST_MZXML ).is_file()
assert P( TEST_MZML ).is_file()
assert P( TEST_MZXML_BROKEN ).is_file()
assert P( TEST_PEAKLIST_FN ).is_file()