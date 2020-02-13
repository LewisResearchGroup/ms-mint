# Welcome


MINT (Metabolomics Integrator) is an app for processing mzML and mzXML mass-spectrometry (MS) files in MS1 mode. Its main function is to sum up intensity values from (usually) pre-defined windows in m/z and scan-time (also called retention time) space. These windows can be provided in from of a [peaklist-file](index.md#peaklists) or set programmatically. Therefore, a large number of MS-files can be processed in very standardized and reproducible manner.

The tool can be used with a browser based graphical user interface (GUI) implemented as interactive dashboard with [Plotly-Dash](https://plot.ly/dash/). A second (experimental) GUI is available that runs integrated in a `Jupyter Notebook`. Alternatively, the `ms_mint` package can be imported as python library to be integrated in any Python script and processing pipeline to automate MS-file processing.

## Installation

### With Anaconda (Miniconda) and PIP

    conda create -n ms-mint python=3.8
    conda activate ms-mint
    pip install ms-mint

### Just PIP

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint

### From source in Anaconda environment (recommended for Windows)

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) (or alternatively `anaconda`) package to install dependencies in a virtual environment.

    git clone https://github.com/soerendip/ms-mint
    cd ms-mint

    conda create -n ms-mint python=3.8
    conda activate ms-mint
    pip setup.py install  # for regular install
    pip install -e .  # for development

### Starting the browser based GUI

The browser based GUI can then be started with

    conda activate ms-mint  # if you run MINT in an Anaconda environment
    Mint.py

Then navigate to the following ULR with your browser: [http://localhost:9999/](http://localhost:9999/). The graphical user interface is explained in more detail [here](gui.md)

# Peaklists
A peaklist is the protocol that captures how data is going to be extracted from the individual MS-files. It is provided as `csv-file` and essentially contains the definitions of peaks to be extracted. A single peak is defined by five properties that need to be present as headers in the `csv-file` which will be explained in the following:

- **peak_label** : A __unique__ identifier such as the biomarker name or ID. Even if multiple peaklist files are used, the label have to be unique across all the files.
- **mz_mean** : The target mass (m/z-value) in [Da].
- **mz_width** : The width of the peak in the m/z-dimension in units of ppm. The window will be *mz_mean* +/- (mz_width * mz_mean * 1e-6). Usually, a values between 5 and 10 are used.
- **rt_min** : The start of the retention time for each peak in [min].
- **rt_max** : The end of the retention time for each peak in [min].
- **intensity_threshold** : A threshold that is applied to filter noise for each window individually. Can be set to 0 or any positive value.

## Example file
**peaklist.csv:**
```text
peak_label,mz_mean,mz_width,rt_min,rt_max,intensity_threshold
Biomarker-A,151.0605,10,4.65,5.2,0
Biomarker-B,151.02585,10,4.18,4.53,0
```