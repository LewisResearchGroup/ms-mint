# Overview of user guide

# MS-MINT: Mass Spectrometry Metabolomics Integration Toolkit

## Introduction

MS-MINT is a powerful Python library designed for comprehensive analysis of mass spectrometry data in metabolomics research. It provides an integrated workflow for processing, analyzing, and visualizing metabolomic datasets.

## Key Features

### 1. Data Processing
- Support for multiple mass spectrometry file formats (mzXML, mzML, mzHDF)
- Automated peak detection and extraction
- Flexible target list management
- Advanced chromatogram analysis

### 2. Visualization
- Interactive and static plotting
- Heatmaps
- Peak shape visualization
- Chromatogram plotting
- 2D histogram analysis

### 3. Analysis Capabilities
- Peak integration
- Retention time alignment
- Intensity normalization
- Statistical comparisons

## Getting Started

### Installation

```bash
pip install ms-mint
```

### Basic Workflow

```python
from ms_mint import Mint

# Create a Mint instance
mint = Mint()

# Load target list
mint.load_targets('path/to/targets.csv')

# Load mass spectrometry files
mint.load_ms_files('path/to/ms/files/')

# Run analysis
mint.run()

# Visualize results
plotter = mint.plotter
plotter.heatmap()
plotter.peak_shapes()
```

## Core Concepts

### Target Lists
Target lists define the specific compounds or peaks of interest in your analysis. They include:

- Peak labels
- Theoretical m/z values
- Retention time windows
- Intensity thresholds

### File Formats
Supported input formats:

- CSV
- Excel (.xlsx)
- Mass spectrometry files:
  - mzXML
  - mzML
  - mzHDF

## Analysis Steps

1. **Target Definition**
   - Create a target list with compound information
   - Specify m/z values, retention times, and other parameters

2. **Data Loading**
   - Load mass spectrometry files
   - Load target list
   - Configure analysis parameters

3. **Peak Extraction**
   - Automated peak detection
   - Integration based on target specifications
   - Quality filtering

4. **Visualization**
   - Multiple visualization options
   - Interactive and static plots
   - Customizable color schemes and layouts

## Advanced Usage

### Customization

- Modify peak detection parameters
- Custom filtering
- Advanced visualization options

### Experimental Notebook Interface
```python
from ms_mint.notebook import Mint

# Interactive Jupyter Notebook mode
mint = Mint()
mint.display()
```

## Performance Considerations

- Optimized for large metabolomics datasets
- Supports parallel processing
- Memory-efficient data handling

## Best Practices

1. **Data Preparation**
   - Use high-quality, clean mass spectrometry data
   - Create precise target lists
   - Validate input files

2. **Parameter Tuning**
   - Adjust peak detection parameters
   - Validate results through visualization
   - Compare multiple analysis runs

3. **Reproducibility**
   - Document all analysis parameters
   - Use consistent target lists and settings
   - Export and share results

## Troubleshooting

### Common Issues
- Incorrect file formats
- Mismatched target list specifications
- Unexpected peak detection results

### Debugging Tips
- Check input file integrity
- Verify target list format
- Use visualization tools to inspect data
- Consult documentation and example datasets

## Contributing

MS-MINT is an open-source project. Contributions are welcome!

- Report issues on GitHub
- Submit pull requests
- Share improvements and extensions

## Citation

When using MS-MINT in your research, please cite the library in your publications DOI: 10.5281/zenodo.12733875

## Support

- GitHub Repository: [https://github.com/LewisResearchGroup/ms-mint](https://github.com/LewisResearchGroup/ms-mint)
- Issue Tracker: [https://github.com/LewisResearchGroup/ms-mint/issues](https://github.com/LewisResearchGroup/ms-mint/issues)

## Disclaimer

MS-MINT is provided as-is. Always validate results and consult domain experts.