# MINT (Metabolomics Integrator)

MINT is an app for summing up intensity values from consistent mass spectrometry (MS) files in `mzML` and `mzXML` format. The tool extracts data within specified windows of retention time and m/z-values using a pre-defined list of peaks that can be changed from project to project. Therefore, the tool allows to reproduciblibly extract data from large numbers of MS files.

The app's frontend is build using [Plotly-Dash](https://plot.ly/dash/) and runs locally in a browser. Thought, the Python functions provided can be imported and used in any python project.

[MINT Documentation](https://soerendip.github.io/ms-mint/)
