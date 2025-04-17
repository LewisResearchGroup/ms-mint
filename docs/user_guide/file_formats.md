# Supported file formats

MS-MINT is designed to support a variety of mass spectrometry (MS) data formats by converting them into a standardized tabular format for downstream analysis. Here's a breakdown of the file types you can use with MS-MINT, based on your code:


### Supported File Formats in MS-MINT

| Format     | Extension     | Description                                                                 | Read By Function            |
|------------|---------------|-----------------------------------------------------------------------------|-----------------------------|
| **mzXML**  | `.mzxml`      | An older open XML-based format for MS data.                                 | `mzxml_to_df()`             |
| **mzML**   | `.mzml`       | Widely used open standard for MS data (XML-based).                          | `mzml_to_df()`              |
| **mzMLb**  | `.mzmlb`      | A binary-efficient variant of mzML (faster, smaller).                       | `mzmlb_to_df__pyteomics()`  |
| **HDF5**   | `.hdf`, `.h5` | Hierarchical format often used for storing large numerical datasets.        | `pd.read_hdf()`             |
| **Feather**| `.feather`    | Fast, lightweight binary format for DataFrames (used with Arrow).           | `pd.read_feather()`         |
| **Parquet**| `.parquet`    | Columnar data format for fast read and compression.                         | `pd.read_parquet()`         |

---

### How It Works

The function `ms_file_to_df()` acts as a universal file loader. It:

1. **Detects the file extension** (e.g., `.mzML`, `.hdf`, etc.)
2. **Dispatches** to the appropriate reader (e.g., `mzml_to_df`, `read_parquet`)
3. **Normalizes the schema** to include the following standard columns:

```python
["scan_id", "ms_level", "polarity", "scan_time", "mz", "intensity"]
```

These columns are crucial for MS-MINT processing and analysis.

---

### Special Cases and Notes

- **Time Unit Handling**:
  - mzXML and mzML files may report scan times in **minutes**, but MS-MINT normalizes this to **seconds**.
  
- **Thermo RAW Parquet Files**:
  - If you load a `.parquet` file not already in MS-MINT format, MS-MINT attempts to reformat it using `format_thermo_raw_file_reader_parquet()`.

- **mzMLb Support**:
  - Only works if the optional dependency `pyteomics.mzmlb` is available.
  - If not installed, MS-MINT will log a warning but still function for other formats.

