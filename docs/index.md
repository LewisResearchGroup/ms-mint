# Metabolomics Integrator
MINT (Metabolomics Integrator) is a post-processing tool for _liquid chromatography-mass spectrometry_ (LCMS) based _metabolomics_. 
Metabolomics is the study of all metabolites (small chemical compounds) in a biological sample e.g. from bacteria or a human blood sample. 
The metabolites can be used to define biomarkers used in medicine to find treatments for diseases or for the development of diagnostic tests 
or for the identification of pathogens such as methicillin resistant _Staphylococcus aureus_ (MRSA). 

## What is LCMS?
A typical biological sample, such as human blood or agar with some kind of bacteria, can contain thousands of [metabolites](https://en.wikipedia.org/wiki/Metabolite) such as 
sugars, alcohols, amino acids, nucleotides and more. To meassure the composition of such a sample [mass spectrometry](https://en.wikipedia.org/wiki/Mass_spectrometry) can be used. 

However, many metabolites share exact masses with other metabolites and therefore would be undistiguishable in the mass spectrometer. 
Therefore, compounds are sorted using [column chromatography](https://en.wikipedia.org/wiki/Column_chromatography) and spread out over time.
The metabolites that enter the column at the same time interact with the column in different ways based on their specific stereochemistry. 
These interactions let compounds move faster or slower through the column and therefore the compounds will elude at different times.
That way various metabolites can be analysed successively over certain timeframe rather than simultaneously.

The mass spectrometer that follows the chromatographic column meassures the masses given at each point in time and returns a time dependent spectrogram.
An example of a LSMS meassurement is visualized in the following figure:

![](image/demo_Saureus_sample_raw.png)
_**Figure 1:** test bla bla_

If we zoom into this figure to a very narrow band of masses the traces of individual metabolites can be observed. The
trace of succinate (or [succinic acid](https://en.wikipedia.org/wiki/Succinic_acid)) is shown here: 

![](image/demo_Saureus_sample_raw_succinate.png)
_**Figure 2:** test bla bla_

This illustrates how dense and precise the information in a LCMS messurement is. For comparison the M/Z value of an electron is 5.489e-4.


## Processing LCMS data
After the data has been collected on a mass spectrometer (MS) and stored in a (usually) vendor specific format the data can be subjected to analysis.
To process data with MINT the data has to be provided in an open format (mzML or mzXML).

Instead of analysing the raw LCMS data it is common practise to deconvolute the data and sum up the signal of individual metabolites.
The processed data should be proportional to the amount of metabolite in the sample. 
However, the meassured intensities will not reflect the relative concentrations between different compounds, only between different samples.
For example, due to different ion efficiences compound **A** might have a stronger signal than compound **B** even if the
compound **B** is present at higher concentration. Therefore, the intensities can only be use to compare relative amounts. 
To estimate absolute concentrations a calibration curve has to be created for every single metabolite.

The binning transforms the semi-structured data into a structured format where each column stands for one particular metabolite.
Often the data is normalized for each metabolite to reflect the relative intensities across multiple samples. 
The structured data can then be subjected to common data anayses such as dimensionality reduction, or clustering analysis.


![](image/demo_hierachical_clustering.png )
_**Figure 3:** Clustering analysis for a small set of metabolites across 12 different samples including 3 different pathogens (EC: _E. coli_, SA: _S. aureus_, CA: _C. albicans_).

## How to use MINT?
The tool can be used for targeted analysis where the m/z-values (mass to charge ratios) and chromatographic retention times are known. 
Alternatively, MINT can be used in an untargeted approach where new biomarkers can be explored without prior knowledge.

MINT currently supports the open data formats mzML and mzXML. The main function is to extract and characterise measured 
intensities in a given m/z and retention time (RT) window. These windows can be provided in form of a [peaklist](peaklists.md) 
or created interactively in the [GUI](gui.md). With this setup large numbers of LCMS-files can be processed automatically, 
standardized and perfectly reproducible.

The tool can be used with a browser based graphical user interface (GUI) implemented as interactive dashboard with 
[Plotly-Dash](https://plot.ly/dash/). Alternatively, the `ms_mint` package can be imported as python library to be 
integrated in any regular Python code as part of a larger processing pipeline or interacively in the [Jupyter Notebook](jupyter.md).


