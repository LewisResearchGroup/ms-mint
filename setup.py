from setuptools import setup, find_packages

import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()


install_reqs = [
    "six",
    "jsonschema",
    "entrypoints",
    "pygments",
    "pexpect",
    "decorator",
    "pillow",
    "lxml",
    "pandas",
    "matplotlib",
    "seaborn",
    "pyteomics",
    "scikit-learn",
    "molmass",
    "pymzml",
    "plotly",
    "colorlover",
    "tqdm",
    "ipywidgets",
    "ipyfilechooser",
    "openpyxl",
    "pyarrow",
    "tables",
]


config = {
    "name": "ms-mint",
    "version": versioneer.get_version(),
    "cmdclass": versioneer.get_cmdclass(),
    "description": "Metabolomics Integrator (Mint)",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "Soren Wacker",
    "url": "https://github.com/LewisResearchGroup/ms-mint",
    "author_email": "swacker@ucalgary.ca",
    "packages": find_packages(),
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    "python_requires": ">=3.8",
    "install_requires": install_reqs,
    "include_package_data": True,
}


setup(**config)
