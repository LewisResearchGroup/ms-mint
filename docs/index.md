# Welcome


MINT (Metabolomics Integrator) is an app for processing mzML and mzXML mass-spectrometry (MS) files in MS1 mode. Its main function is to sum up intensity values from usually pre-defined windows in m/z and scan-time (also called retention time) space. These windows can be provided in a `csv-file` or set programmatically. Therefore, a large number of MS-files can be processed in very standardized and reproducible manner.

The tool can be used with a browser based graphical user interface (GUI) implemented as interactive dashboard with [Plotly-Dash](https://plot.ly/dash/). A second (experimental) GUI is available that runs integrated in a `Jupyter Notebook`. Alternatively, the `ms_mint` package can be imported as python library to be integrated in any Python script and processing pipeline to automate MS-file processing.


## Installation

### With Anaconda (Miniconda)

   conda create -n ms-mint python=3.8
   conda activate ms-mint
   pip install ms-mint

The server to run the GUI can be started with 

    Mint.py

Then navigate to http://localhost:9999.

### With PIP

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint

The server to run the GUI can be started with 

    Mint.py

Then navigate to http://localhost:9999.

### From source

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) (or alternatively `anaconda`) package to install dependencies in a virtual environment.

    git clone https://github.com/soerendip/ms-mint
    cd ms-mint

    conda create -n ms-mint python=3.8
    conda activate ms-mint
    pip setup.py install  # for regular install
    pip install -e .  # for development

## Start the application

Start the app with

    conda activate ms-mint  # if you run MINT in an Anaconda environment
    Mint.py
    
Then navigate to the following ULR with your browser: `http://localhost:9999/`

## The GUI
The graphical user interface is explained in more detail [here](gui.md)
