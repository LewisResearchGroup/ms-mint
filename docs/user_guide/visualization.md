# Visualization

## Overview

The MS-MINT visualization tools provide powerful and flexible ways to explore mass spectrometry data, offering both static (Matplotlib) and interactive (Plotly) visualization options.

## Visualization Types

### 1. Heatmaps

#### Static Heatmaps
```python
# Hierarchical clustering heatmap
plotter.hierarchical_clustering(
    data=your_dataframe,      # Data to visualize
    vmin=-3,                  # Minimum value for color scaling
    vmax=3,                   # Maximum value for color scaling
    metric='cosine',          # Distance metric for clustering
    figsize=(8, 8),           # Figure size
    xmaxticks=10,             # Maximum x-axis ticks
    ymaxticks=10              # Maximum y-axis ticks
)
```

#### Interactive Heatmaps
```python
# Create an interactive heatmap
plotter.heatmap(
    col_name='peak_max',      # Column to visualize
    normed_by_cols=True,      # Normalize by column
    transposed=False,         # Transpose matrix
    clustered=True,           # Apply hierarchical clustering
    correlation=False         # Convert to correlation matrix
)
```

#### Heatmap Options
- **Normalization**: Scale data by column maximum
- **Clustering**: Hierarchical clustering of rows/columns
- **Correlation**: Convert to correlation matrix
- **Color Scaling**: Customize color range and palette

### 2. Peak Shapes

#### Static Peak Shape Plots
```python
# Plot peak shapes from results
plotter.peak_shapes(
    fns=['file1.mzXML', 'file2.mzXML'],  # Specific files
    peak_labels=['Caffeine', 'Glucose'],  # Specific peaks
    height=3,                 # Facet height
    aspect=1.5,               # Facet aspect ratio
    col_wrap=4,               # Maximum columns
    legend=True               # Show legend
)
```

#### Interactive Peak Shape Plots
```python
# Interactive peak shape visualization
plotter.peak_shapes(
    fns=['file1.mzXML', 'file2.mzXML'],  # Specific files
    peak_labels=['Caffeine', 'Glucose'],  # Specific peaks
    interactive=True,         # Use Plotly interactive mode
    color='ms_file_label'     # Color by file label
)
```

### 3. Chromatograms

```python
# Plot chromatograms
plotter.chromatogram(
    fns=['file1.mzXML', 'file2.mzXML'],  # Specific files
    peak_labels=['Caffeine', 'Glucose'],  # Specific peaks
    interactive=False,        # Static plot
    filters=None              # Optional data filters
)
```

### 4. 2D Histograms

```python
# Create 2D histogram of MS file
plotter.histogram_2d(
    fn='your_ms_file.mzXML',  # MS file to visualize
    peak_label='Caffeine',    # Optional peak label to highlight
    rt_margin=0,              # Retention time margin
    mz_margin=0               # M/Z margin
)
```

## Advanced Visualization Techniques

### Customization Options
- **Color Palettes**: Choose from various color schemes
- **Interactive vs. Static**: Switch between Plotly and Matplotlib
- **Filtering**: Select specific files or peaks
- **Scaling**: Normalize and transform data

### Color Customization

```python
# Use different color palettes
plotter.peak_shapes(
    palette='Plasma',         # Color palette
    color='ms_file_label'     # Color grouping
)
```

## Best Practices

1. **Data Preprocessing**: Ensure data is clean and standardized
2. **Choose Appropriate Visualization**: Match plot type to your analysis goals
3. **Interactive vs. Static**: 
   - Use interactive for detailed exploration
   - Use static for publications/reports
4. **Color Choices**: Select palettes that are color-blind friendly

## Troubleshooting

- **No Data Displayed**: 
  - Check data filtering
  - Verify peak labels and file names
- **Performance Issues**: 
  - Reduce data size
  - Use interactive mode for large datasets
- **Color Mapping**: Ensure unique color mapping for different groups

## Performance Tips

- For large datasets, use interactive Plotly visualizations
- Limit the number of peaks and files in a single plot
- Use normalization and scaling to improve visualization clarity

## Example Workflow

```python
from ms_mint import Mint

# Create Mint instance
mint = Mint()

# Load data
mint.load_targets('targets.csv')
mint.load_ms_files('data_directory/*')

# Run analysis
mint.run()

# Create plotter
plotter = mint.plot

# Visualize results
plotter.heatmap(col_name='peak_max')
plotter.peak_shapes(interactive=True)
plotter.chromatogram()
```

## Notes

- Visualization tools are designed to be flexible and intuitive
- Always verify data and visualization parameters
- Experiment with different visualization techniques