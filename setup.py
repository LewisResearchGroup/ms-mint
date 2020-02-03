try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Metabolomics Integration Tool (Mint)',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip/ms-mint',
    'author_email': 'swacker@ucalgary.ca',
    'version': 0.0.25,
    'scripts': ['scripts/Mint.py', 'scripts/Mint.bat'],
    'packages': ['ms_mint'],
    'name': 'ms_mint',
    'data_files': [('scripts', ['scripts/Mint.py', 'scripts/Mint.bat']),
                   ('static', ['static/Standard_Peaklist.csv'])]
}

setup(**config)
