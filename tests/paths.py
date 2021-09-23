
import os
from pathlib import Path as P


TEST_MZXML = os.path.abspath( 
    os.path.join( 
       'tests' , 'data', 'ms_files', 'test.mzXML'
    )
)

TEST_MZML = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'test.mzML'
    )
)


TEST_MZML_POS = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'example-pos.mzML'
    )
)

TEST_MZML_NEG = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'example-neg.mzML'
    )
)

TEST_FEATHER = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'test.feather'
    )
)

TEST_PARQUET = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'thermo-parser-pos-ion-example.parquet'
    )
)

TEST_MZMLB_POS = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'ms_files', 'test-pos.mzMLb'
    )
)


TEST_TARGETS_FN = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'targets', 'targets_v1.csv'
    )
)

TEST_TARGETS_FN_V0 = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'targets', 'targets_v0.csv'
    )
)


TEST_PEAK_AREA_RESULTS = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'results', 'test_peak_area_results.csv'
    )
)


TEST_MZXML_BROKEN = os.path.abspath( 
    os.path.join( 
        'tests' , 'data', 'broken', 'broken.mzXML'
    )
)


assert P( TEST_MZXML ).is_file(), TEST_MZXML
assert P( TEST_MZML ).is_file(), TEST_MZML
assert P( TEST_MZXML_BROKEN ).is_file(), TEST_MZXML_BROKEN
assert P( TEST_TARGETS_FN ).is_file(), TEST_TARGETS_FN
