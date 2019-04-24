try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

NAME = 'miiit'
config = {
    'description': 'Metabolomics INtegration Tool (Mint)',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip',
    'download_url': f'https://github.com/soerendip/{NAME}',
    'author_email': 'swacker@ucalgary.ca',
    'version': 0.1,
    'install_requires': [],
    'packages': [f'{NAME}'],
    'scripts': [],
    'name': f'{NAME}',
    'data_files': [('static', ['static/Standard_Peaklist.csv'])]
}

setup(**config)
