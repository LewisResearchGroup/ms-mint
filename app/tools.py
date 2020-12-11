import base64
import os
from glob import glob

from ms_mint.io import mzml_to_pandas_df, mzxml_to_pandas_df


def parse_ms_files(contents, filename, date, target_dir):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    fn_abs = os.path.join(target_dir, filename)

    with open(fn_abs, 'wb') as file:
        file.write(decoded)


def get_dirnames(path):
    dirnames = [ f.name for f in os.scandir(path) if f.is_dir() ]
    return dirnames

