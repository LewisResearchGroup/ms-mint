# Interactive GUI

MS-MINT includes a Solara-based graphical interface for interactive analysis in Jupyter notebooks.

## Installation

Install with GUI support:

```bash
pip install ms-mint[gui]
```

## Usage

### In Jupyter Notebook

```python
from ms_mint.gui import MintGui

MintGui()
```

### Standalone Application

```bash
solara run ms_mint.gui.app:MintGui
```

## Interface Overview

The GUI is organized into five tabs:

### MS Files Tab

- **File Selection**: Load MS files using glob patterns (e.g., `./data/*.mzML`)
- **Metadata Panel**: View and edit sample metadata
- Supports mzML, mzXML, mzMLb, and parquet formats

### Targets Tab

- **Target Loading**: Upload target lists (CSV/Excel)
- **Target Management**: Reorder, activate/deactivate targets
- **RT Optimization**: Automatically optimize retention time windows based on actual chromatographic data

### Processing Tab

- **Run Analysis**: Process MS files with loaded targets
- **Progress Tracking**: Monitor processing status
- **Export Results**: Save results to CSV
- **Settings**:
    - Processing mode (standard/parallel)
    - RT margin adjustment
    - Thread count for parallel processing

### Results Tab

- **Results Table**: View peak integration results
- **Crosstab View**: Pivot table of peak areas by sample and metabolite

### Visualization Tab

- **Heatmaps**: Hierarchical clustering visualization
- **Peak Shapes**: Chromatographic peak profiles
- **2D Histograms**: Raw data exploration
- Export plots in multiple formats (PNG, PDF, SVG)

## Sidebar Settings

The sidebar provides global settings:

| Setting | Description |
|---------|-------------|
| Working Directory | Base path for file operations |
| RT Unit | Display retention times in seconds or minutes |
| Image Format | Export format for plots |
| Threads | Number of parallel processing threads |
| Session | Save/load analysis state |

## Workflow Example

1. **Load MS Files**: Navigate to MS Files tab, enter glob pattern, click Load
2. **Load Targets**: Switch to Targets tab, upload target list CSV
3. **Optimize RT** (optional): Click "Optimize RT" to refine retention time windows
4. **Run Processing**: Go to Processing tab, click Run
5. **View Results**: Check Results tab for peak areas
6. **Visualize**: Use Visualization tab for heatmaps and peak shapes
7. **Export**: Save results and session for later use

## Session Management

Save your analysis state:

- **Save Session**: Preserves loaded files, targets, results, and settings
- **Load Session**: Restore a previous analysis state

Sessions are saved to the working directory as `mint_session.pkl`.

## See Also

- [Quickstart](../quickstart.md)
- [Visualization Guide](visualization.md)
- [Target Lists](target_lists.md)
