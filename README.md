# MINT (Metabolomics Integrator)
 


![GUI](./docs/image/mint-overview.png "The GUI")

MINT is an app for processing mzML and mzXML mass-spectrometry (MS) files in MS1 mode. Its main function is to sum up intensity values from usually pre-defined windows in m/z and scan-time (also called retention time) space. These windows can be provided in a `csv-file` or set programmatically. Therefore, a large number of MS-files can be processed in very standardized and reproducible manner.

The tool can be used with a browser based graphical user interface (GUI) implemented as interactive dashboard with [Plotly-Dash](https://plot.ly/dash/). A second (experimental) GUI is available that runs integrated in a `Jupyter Notebook`. Alternatively, the `ms_mint` package can be imported as python library to be integrated in any Python script and processing pipeline to automate MS-file processing.

More information on how to install and run the program can be found in the [Documentation](https://soerendip.github.io/ms-mint/).



## Download Latest release
For Windows 10 a build is provided [here](https://github.com/soerendip/ms-mint/releases/latest)

Simply, unpack the zip-archive and start `Mint.exe`. Then navigate to http://localhost:9999 if the browser does not open automatically.


## Installation

### Installation with pip

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint

Then the GUI can be started with 

    Mint.py

Then navigate to http://localhost:9999 if the browser does not open automatically.


### From source and creating windows executable

    git clone ...
    pip install -e . 
    pyinstaller --one-dir specfiles\Mint__onedir__.spec


# Developer Notes

    python3 setup.py sdist bdist_wheel
    python3 -m twine upload --repository ms-mint dist/ms*mint-*

### On windows
    pyinstaller --onedir --noconfirm specfiles\Mint__onedir__.spec

