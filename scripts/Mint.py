#!/usr/bin/env python

import os, sys
import mint
from pathlib import Path as P 

print('Starting Mint')

if sys.platform == 'linux':
    wdir = os.path.abspath(P(os.path.dirname(mint.__file__))/P('../notebooks'))
    os.chdir(wdir)
    os.system('xdg-open "http://localhost:9999/apps/Mint.ipynb?appmode_scroll=0"')
    os.system('jupyter notebook --port 9999 --no-browser')

elif sys.platform == 'win32':
    wdir = os.path.abspath(P(os.path.dirname(mint.__file__))/P('../notebooks'))
    os.chdir(wdir)
    os.system('start "" http://localhost:9999/apps/Mint.ipynb?appmode_scroll=0')
    os.system('jupyter notebook --no-browser --port=9999')

else:
    print(sys.platform, 'unknown')
