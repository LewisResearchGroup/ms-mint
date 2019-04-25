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
    'install_requires': [],
    'packages': [f'{NAME}'],
    'name': f'{NAME}',
    'data_files': [('static', ['static/Standard_Peaklist.csv'])]
}

setup(**config)
