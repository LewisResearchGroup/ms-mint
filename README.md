[![Python package](https://github.com/lewisresearchgroup/ms-mint/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/sorenwacker/ms-mint/actions/workflows/pythonpackage.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5178/badge)](https://bestpractices.coreinfrastructure.org/projects/5178)
![](images/coverage.svg)
[![CodeQL](https://github.com/lewisresearchgroup/ms-mint/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/lewisresearchgroup/ms-mint/actions/workflows/codeql-analysis.yml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/LewisResearchGroup/ms-mint.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/LewisResearchGroup/ms-mint/context:python)


# MINT (Metabolomics Integrator)

The Metabolomics Integrator (MINT) is a post-processing tool for liquid chromatography-mass spectrometry (LCMS) based metabolomics. 
Metabolomics is the study of all metabolites (small chemical compounds) in a biological sample e.g. from bacteria or a human blood sample. 
The metabolites can be used to define biomarkers used in medicine to find treatments for diseases or for the development of diagnostic tests 
or for the identification of pathogens such as methicillin resistant _Staphylococcus aureus_ (MRSA). 
More information on how to install and run the program can be found in the [Documentation](https://lewisresearchgroup.github.io/ms-mint/).

# News

MINT has been split into the Python library and the app. This repository contains the Python library. For the app follow [this link](https://github.com/LewisResearchGroup/ms-mint-app).

# Contributions

All contributions, bug reports, code reviews, bug fixes, documentation improvements, enhancements, and ideas are welcome.
Before you modify the code please reach out to us using the [issues](https://github.com/LewisResearchGroup/ms-mint/issues) page.


# Code standards

The project follows PEP8 standard and uses Black and Flake8 to ensure a consistent code format throughout the project.


# Change log

## v0.1.5

The way to load targets from files has been changed.

    mint.target_files = [...]  -> mint.load_targets([...])