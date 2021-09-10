# MINT (Metabolomics Integrator)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5178/badge)](https://bestpractices.coreinfrastructure.org/projects/5178)

The Metabolomics Integrator (MINT)) is a post-processing tool for liquid chromatography-mass spectrometry (LCMS) based metabolomics. 
Metabolomics is the study of all metabolites (small chemical compounds) in a biological sample e.g. from bacteria or a human blood sample. 
The metabolites can be used to define biomarkers used in medicine to find treatments for diseases or for the development of diagnostic tests 
or for the identification of pathogens such as methicillin resistant _Staphylococcus aureus_ (MRSA). 
More information on how to install and run the program can be found in the [Documentation](https://soerendip.github.io/ms-mint/).

![](./docs/image/distributions.png)

## Python API for metabolomics

    from ms_mint.notebook import Mint
    mint.ms_files = glob('/path/to/files/*mzML')
    mint.peaklist_files = '/path/to/peaklist/file/peaklist.csv'
    mint.run()
    mint.results

![Mint Jupyter Results](./docs/image/jupyter_results.png "Mint Jupyter Results")

More information in the documentation.

# Errors, Feedback, Feature Requests
If you encounter an error, if you have a request for a new feature, or for general feedback, please open a new ticket at the [issue tracker](https://github.com/soerendip/ms-mint/issues).

# Contributions are welcome
If you want to contribute to MINT please send me a notification. 

In general I will ask you to followowing steps:

1. fork the repository
1. implement the new feature or bug-fix
1. add corresponding tests
2. run `flake8`
3. submit a pull request

## Code standards
Before submitting a pull request please run `flake8`.
