# MIIIT
Metabolomics Interactive Intensity Integration Tool

A jupyter notebook based app for summing up intensity values in mass spectrometry mzXML files in specified windows of retention time and m/z.

## Installation

Here we use `conda` from the `miniconda` (or `anaconda`) package to install dependencies in a virtual environment.
[conda](https://conda.io/en/latest/miniconda.html)

    conda env create -f requirements.txt

Start the app with

    conda activate miiit
    jupyter notebook Metabolomics_Interactive_Intensity_Integration_Tool.ipynb --no-browser

Then navigate to 

    http://localhost:8888/apps/Metabolomics_Interactive_Intensity_Integration_Tool.ipynb?appmode_scroll=0
