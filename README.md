# MINT (Metabolomics Integrator)

MINT is an app for processing mzML and mzXML mass-spectrometry (MS) files in MS1 mode. Its main function is to sum up intensity values from predefined windows in m/z and scan-time (also called retention time) space. These window can be provided a separate `csv-file` or set interactively in a Jupyter notebook. A large number of MS-files can be processed in a standardized and reproducible manner.

The tool can be used with a browser based GUI implemented as interactve dashboard with [Plotly-Dash](https://plot.ly/dash/). A second experimental GUI is available that runs integrated in a Jupyter Notebook. Alternatively, the `ms_mint` package can be imported as python library without any gui to be integrated in any Python script to automate MS-file processing.

More information on how to install and run the program can be found in the [Documentation](https://soerendip.github.io/ms-mint/).

### Installation with pip

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint
