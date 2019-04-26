# Metabolomics Integration Tool (Mint)

A jupyter notebook based app for summing up intensity values in mass spectrometry mzXML files in specified windows of retention time and m/z.

## Installation

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) 
(or alternatively `anaconda`) package to install dependencies in a virtual environment.

    conda env create -f requirements.txt
    conda activate mint
    python setup.py install

## Start the application

Start the app with

    conda activate mint
    jupyter notebook --no-browser --port 9999

Then navigate to the following ULR with your browser:

    http://localhost:9999/apps/Mint.ipynb?appmode_scroll=0


![Demo Image](./image/mint.png "Demo image")
