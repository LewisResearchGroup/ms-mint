# Troubleshooting

Common issues and solutions when using MS-MINT.

## Installation Issues

### ImportError: solara not found

**Cause:** GUI dependencies not installed.

**Solution:**
```bash
pip install ms-mint[gui]
```

### mzMLb files not supported

**Cause:** Optional HDF5 dependencies missing.

**Solution:**
```bash
pip install h5py hdf5plugin
```

## File Loading Issues

### "File not found" errors

**Cause:** Incorrect file path or working directory.

**Solution:**
1. Use absolute paths or verify relative paths
2. Check the working directory: `mint.wdir`
3. Verify files exist: `ls your_path/*.mzML`

### Empty DataFrame after loading

**Cause:** File format not recognized or corrupted file.

**Solution:**
1. Verify file extension is lowercase (`.mzml` not `.MZML`)
2. Try converting with MSConvert
3. Check file integrity by opening in another tool

### Memory errors with large files

**Cause:** Loading too many files simultaneously.

**Solution:**
```python
# Process files and save results incrementally
mint.run(fn='results.csv')
```

## Target List Issues

### "Duplicate peak_label" error

**Cause:** Non-unique labels in target list.

**Solution:**
```python
# Check for duplicates
targets[targets.duplicated('peak_label')]

# Auto-generate unique labels
targets['peak_label'] = [f'peak_{i}' for i in range(len(targets))]
```

### No peaks detected

**Cause:** RT windows or m/z values don't match actual data.

**Solution:**
1. Check RT units (seconds vs minutes)
2. Verify m/z values match ionization mode
3. Use RT optimization:
```python
mint.opt.rt_min_max(plot=True)
```

### Wrong RT unit conversion

**Cause:** Inconsistent `rt_unit` column values.

**Solution:**
Standardize units before loading:
```python
targets['rt_unit'] = 's'  # or 'min'
```

## Processing Issues

### Processing hangs or is slow

**Cause:** Large number of files or targets.

**Solution:**
```python
# Use parallel processing
mint.run(nthreads=4)

# Or process in batches
for batch in file_batches:
    mint.ms_files = batch
    mint.run(fn='results.csv')
```

### Zero peak areas

**Cause:** No data points within the specified RT/m/z window.

**Solution:**
1. Widen RT windows: increase `rt_min`/`rt_max` range
2. Increase m/z width: set `mz_width` to 15-20 ppm
3. Lower intensity threshold: set `intensity_threshold` to 0

### Results differ between runs

**Cause:** Floating-point precision or file order differences.

**Solution:**
```python
# Sort results for consistent comparison
results = results.sort_values(['ms_file', 'peak_label']).reset_index(drop=True)
```

## Visualization Issues

### Plots not displaying in Jupyter

**Cause:** Matplotlib backend issue.

**Solution:**
```python
%matplotlib inline
# or
%pylab inline
```

### Interactive plots not working

**Cause:** Plotly not installed or notebook extension missing.

**Solution:**
```bash
pip install plotly
jupyter labextension install jupyterlab-plotly  # for JupyterLab
```

### Heatmap clustering fails

**Cause:** NaN or infinite values in data.

**Solution:**
```python
# Fill missing values
crosstab = mint.crosstab().fillna(0)

# Or remove rows/columns with missing data
crosstab = crosstab.dropna()
```

## GUI Issues

### GUI not rendering

**Cause:** Solara not properly installed or kernel mismatch.

**Solution:**
```bash
pip install ms-mint[gui] --force-reinstall
# Restart Jupyter kernel
```

### Session load fails

**Cause:** Incompatible session file from different version.

**Solution:**
Delete the session file and start fresh:
```bash
rm mint_session.pkl
```

## Performance Tips

| Issue | Solution |
|-------|----------|
| Slow file loading | Convert to parquet format |
| Memory issues | Process files in batches |
| Slow processing | Increase thread count |
| Large result files | Use `fn` parameter to stream to disk |

## Debug Mode

Enable verbose output for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

mint = Mint(verbose=True)
```

## Getting Help

- [GitHub Issues](https://github.com/LewisResearchGroup/ms-mint/issues)
- Check existing issues before creating new ones
- Include: Python version, ms-mint version, error traceback
