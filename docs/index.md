# Metabolomics INTegrator (MINT)

MINT (Metabolomics Integrator) is an app for a streamlined post-processing of files from  liquid chromatography-mass spectrometry (LCMS) files. MINT can read mzML and mzXML formats. Its main function is to sum up intensity values from pre-defined windows in mass over charge (m/z) and retention time (RT) space. These windows can be provided in from of a [peaklist](peaklists.md) or created interactively in the [GUI](gui.md). Whith this setup large numbers of LCMS-files can be processed automatically, standardized and in a highly reproducible manner.

The tool can be used with a browser based graphical user interface (GUI) implemented as interactive dashboard with [Plotly-Dash](https://plot.ly/dash/). Alternatively, the `ms_mint` package can be imported as python library to be integrated in any regular Python code as part of a larger processing pipeline.

![MINT](image/mint-overview.png)



