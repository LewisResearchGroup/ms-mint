# MINT (Metabolomics Integrator)

MINT is an app for processing mzML and mzXML mass-spectrometry files in MS1 mode. It main function is to sum up intensity values from predifined windows in m/z and scan-time (also called retention time) windows. The tool extracts based on window definitions provided in a `csv-file`. A large number of files can be processed in express mode.

The tool can be used with a GUI that is implemented with Plotly-Dash or with the experimental Jupyter notebook gui. Alternatively, it can be imported in any python script to automate MS-file processing in a standardized and reproducible manner.

The app's frontend is build using [Plotly-Dash](https://plot.ly/dash/) and runs locally in a browser. Thought, the Python functions provided can be imported and used in any python project.

More information on how to install and run the program can be found here: [MINT Documentation](https://soerendip.github.io/ms-mint/)

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint
