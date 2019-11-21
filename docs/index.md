# Welcome to MINT (Metabolomics Integrator)

MINT is an app for summing up intensity values from consistent mass spectrometry (MS) files in `mzML` and `mzXML` format. The tool extracts data within specified windows of retention time and m/z-values using a pre-defined list of peaks that can be changed from project to project. Therefore, the tool allows to reproduciblibly extract data from large numbers of MS files.

The app's frontend is build using [Plotly-Dash](https://plot.ly/dash/) and runs locally in a browser. Thought, the Python functions provided can be imported and used in any Python project independently.


## Installation

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) (or alternatively `anaconda`) package to install dependencies in a virtual environment.

    git clone https://github.com/soerendip/ms-mint
    cd ms-mint

    conda create -n ms-mint -c plotly lxml matplotlib pandas pandoc pip plotly plotly_express dash dash-bootstrap-components pyqt python=3 scipy setuptools sqlite statsmodels flask

    conda activate ms-mint
    pip install pyteomics openpyxl colorlover
    python setup.py install  # Don't use pip to install it.


## Start the application

Start the app with

    conda activate ms-mint
    Mint.py  # Under Linux
    python scripts/Mint.py  # Under Windows

Then navigate to the following ULR with your browser: `http://localhost:9999/`

## The GUI
The graphical user interface is explained in more detail [here](gui.md)
