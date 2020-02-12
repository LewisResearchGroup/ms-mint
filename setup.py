try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'ms-mint-soerendip'
    'version': '0.0.25',
    'description': 'Metabolomics Integration Tool (Mint)',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip/ms-mint',
    'author_email': 'swacker@ucalgary.ca',
    'scripts': ['scripts/Mint.py', 'scripts/Mint.bat'],
    'packages': ['ms_mint'],
    'data_files': [('scripts', ['scripts/Mint.py', 'scripts/Mint.bat']),
                   ('static', ['static/Standard_Peaklist.csv'])]
}

setup(**config)
