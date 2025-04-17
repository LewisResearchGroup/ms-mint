# Target Lists User Manual

## Introduction

Target lists are essential tools in mass spectrometry data analysis, defining precise parameters for extracting and analyzing peaks of interest in chromatographic data.

## Example Target List

| peak_label | mz_mean | mz_width | rt | rt_min | rt_max | rt_unit | intensity_threshold | target_filename |
|------------|---------|----------|-----|--------|--------|---------|---------------------|-----------------|
| Caffeine | 195.0875 | 10 | 3.5 | 3.0 | 4.0 | min | 0 | caffeine_targets.csv |
| Glucose | 203.0794 | 10 | 2.7 | 2.3 | 3.1 | min | 0 | sugar_targets.csv |
| Acetaminophen | 152.0706 | 10 | 4.2 | 3.8 | 4.6 | min | 0 | drug_targets.csv |

## Target List Structure

A target list is a pandas DataFrame with nine key columns:

### 1. peak_label
- **Type**: String
- **Description**: Unique identifier for each peak
- **Key Features**:
  - Must be unique across the entire list
  - Automatically converted to string type
  - Default labels (e.g., "C_0", "C_1") generated if not provided

### 2. mz_mean
- **Type**: Numeric 
- **Description**: Theoretical m/z (mass-to-charge) value of the target ion
- **Capabilities**:
  - Can be calculated from chemical formula
  - Critical for precise ion extraction

### 3. mz_width
- **Type**: Numeric
- **Description**: Peak width in parts per million (ppm)
- **Details**:
  - Default: 10 ppm
  - Mass window calculated by: `m/z * 1e-6 * mz_width`

### 4. rt (Retention Time)
- **Type**: Numeric or None
- **Description**: Expected peak time
- **Notes**: 
  - Optional field
  - Informs peak optimization procedures
  - Not directly used in processing

### 5. rt_min
- **Type**: Numeric
- **Description**: Starting time for peak integration

### 6. rt_max
- **Type**: Numeric 
- **Description**: Ending time for peak integration

### 7. rt_unit
- **Allowed Values**: 
  - `s` (seconds)
  - `min` (minutes)
- **Behavior**:
  - Automatic conversions:
    * "m", "minute", "minutes" → "min"
    * "sec", "second", "seconds" → "s"
  - Standardizes to seconds internally

### 8. intensity_threshold
- **Type**: Numeric (≥ 0)
- **Description**: Minimum intensity for peak inclusion
- **Recommendation**: 0 (no filtering)

### 9. target_filename
- **Type**: String
- **Purpose**: Tracking origin of target list
- **Usage**: Informational only, not used in processing

## Working with Target Lists

### Supported File Formats
- CSV (.csv)
- Excel (.xlsx)

### Reading Target Lists
```python
from ms_mint.targets import read_targets

# Load a single file
targets = read_targets('your_target_list.csv')

# Load multiple files
targets = read_targets(['file1.csv', 'file2.xlsx'])
```

### Creating Target List Programmatically
```python
import pandas as pd

targets = pd.DataFrame({
    'peak_label': ['Caffeine', 'Glucose', 'Acetaminophen'],
    'mz_mean': [195.0875, 203.0794, 152.0706],
    'mz_width': [10, 10, 10],
    'rt': [3.5, 2.7, 4.2],
    'rt_min': [3.0, 2.3, 3.8],
    'rt_max': [4.0, 3.1, 4.6],
    'rt_unit': ['min', 'min', 'min'],
    'intensity_threshold': [0, 0, 0],
    'target_filename': ['caffeine_targets.csv', 'sugar_targets.csv', 'drug_targets.csv']
})
```

## Advanced Functionality

### Generating Target Grids
Create a comprehensive grid of targets across retention times:

```python
from ms_mint.targets import gen_target_grid

masses = [100.5, 200.7, 300.2]
targets = gen_target_grid(
    masses=masses,      # List of m/z values
    dt=0.5,             # Time window size [min]
    rt_max=10,          # Maximum retention time
    mz_ppm=10,          # Mass width [ppm]
    intensity_threshold=0
)
```

### Comparing Target Lists
Identify new or changed targets:

```python
from ms_mint.targets import diff_targets

differences = diff_targets(old_target_list, new_target_list)
```

## Validation and Troubleshooting

### Validate Target List
```python
from ms_mint.targets import check_targets

is_valid = check_targets(your_target_list)
```

Validation checks include:
- Correct DataFrame structure
- Unique string labels
- Proper column configuration

### Common Issues
- **Duplicate Labels**: Ensure unique `peak_label`
- **Unit Conversion**: Be careful with retention time units
- **Mass Calculations**: Verify m/z values carefully
