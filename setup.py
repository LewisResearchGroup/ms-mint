from setuptools import setup, find_packages

import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'numpy>=1.20'
    'pandas>=1',
    'plotly',
    'lxml',
    'matplotlib',
    'pandoc',
    'plotly',
    'plotly_express',
    'dash',
    'scipy==1.6.2',  #
    'setuptools',
    'statsmodels',
    'flask',
    'pyteomics',
    'openpyxl',
    'colorlover',
    'dash-bootstrap-components',
    'dash-extensions',
    'dash-tabulator>=0.4.2',
    'scikit-learn',
    'xlrd',
    'urllib3',
    'ipywidgets',
    'pyopenms',
    'pymzml',
    'tqdm',
    'seaborn',
    'ipyfilechooser',
    'waitress',
    'pyarrow>=3',
    'flask-compress',
    'molmass',
    'dash_uploader',
    'filelock',
    'wget',
    'bs4',
    'pytest-cov',
    'jsbeautifier']


config = {
    'name': 'ms-mint',
    'version': versioneer.get_version(),
    'cmdclass': versioneer.get_cmdclass(),
    'description': 'Metabolomics Integrator (Mint)',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip/ms-mint',
    'author_email': 'swacker@ucalgary.ca',
    'scripts': ['scripts/Mint.py'],
    'packages': find_packages(),
    'data_files': [('scripts', ['scripts/Mint.py'])],
 #                  ('static', ['static/Standard_Peaklist.csv',
 #                               'static/ChEBI-Chem.parquet',
 #                               'static/ChEBI-Groups.parquet']),
 #                  ],
    'classifiers': [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
   'python_requires': '>=3.7',
   'install_requires': install_requires,
   'include_package_data': True,
   'package_data': {'ms_mint.static': [
                        'Standard_Peaklist.csv',
                        'ChEBI-Chem.parquet',
                        'ChEBI-Groups.parquet']
                    }
}

setup(**config)
