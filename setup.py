try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

NAME = 'mint'
config = {
    'description': 'Metabolomics Integration Tool (Mint)',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip/mint',
    'author_email': 'swacker@ucalgary.ca',
    'version': 0.1,
    'scripts': ['scripts/Mint', 'scripts/Mint__windows.bat'],
    'packages': ['mint'],
    'name': 'mint',
    'data_files': [('static', ['static/Standard_Peaklist.csv']),
                   ('notebooks', ['notebooks/Mint.ipynb'])]
}

setup(**config)
