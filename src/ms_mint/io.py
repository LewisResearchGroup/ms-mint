"""Functions to read and write MS-MINT files."""

import pandas as pd
import numpy as np
import io
import logging
import pathlib
from typing import Union, Optional, List, Dict, Any, Callable, Tuple, Literal, cast
from pathlib import Path as P
from datetime import date
from pyteomics import mzxml, mzml

try:
    from pyteomics import mzmlb

    MZMLB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import pyteomics.mzmlb:\n{e}")  # Fixed typo: Cound → Could
    MZMLB_AVAILABLE = False


MS_FILE_COLUMNS = [
    "scan_id",
    "ms_level",
    "polarity",
    "scan_time",
    "mz",
    "intensity",
]


def ms_file_to_df(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read MS file and convert it to a pandas DataFrame.

    Args:
        fn: Filename or path to the MS file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if the file cannot be read.
    """
    fn = str(fn)

    try:
        if fn.lower().endswith(".mzxml"):
            df = mzxml_to_df(fn, read_only=read_only)
        elif fn.lower().endswith(".mzml"):
            df = mzml_to_df(fn, read_only=read_only)
        elif fn.lower().endswith("hdf"):
            df = pd.read_hdf(fn)
        elif fn.lower().endswith(".feather"):
            df = pd.read_feather(fn)
        elif fn.lower().endswith(".parquet"):
            df = read_parquet(fn, read_only=read_only)
        elif fn.lower().endswith(".mzmlb"):
            df = mzmlb_to_df__pyteomics(fn, read_only=read_only)
        else:
            logging.error(f"Cannot read file {fn} of type {type(fn)}")
            return None
    except IndexError as e:
        logging.warning(f"{e}: {fn}")
        return None

    if read_only:
        return df
    else:
        # Compatibility with old schema
        df = df.rename(
            columns={
                "retentionTime": "scan_time",
                "intensity array": "intensity",
                "m/z array": "mz",
            }
        )
        if "scan_id" not in df.columns:
            df["scan_id"] = 0
        if "ms_level" not in df.columns:
            df["ms_level"] = 1
        # Set datatypes
        set_dtypes(df)
    return df


def mzxml_to_df(
    fn: Union[str, pathlib.Path],
    read_only: bool = False,
    time_unit_in_file: Literal["min", "sec"] = "min",
) -> Optional[pd.DataFrame]:
    """Read mzXML file and convert it to pandas DataFrame.

    Args:
        fn: Filename or path to the mzXML file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.
        time_unit_in_file: The time unit used in the mzXML file.
            Must be either 'sec' or 'min'. Default is 'min'.

    Returns:
        DataFrame containing MS data, or None if read_only is True.

    Raises:
        AssertionError: If the filename does not end with '.mzxml'.
    """
    assert str(fn).lower().endswith(".mzxml"), fn

    with mzxml.MzXML(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = [_extract_mzxml(x) for x in data]
    df = pd.json_normalize(data, sep="_")

    # Convert retention time to seconds
    if time_unit_in_file == "min":
        df["scan_time"] = df["scan_time"].astype(np.float64) * 60.0

    df = df.explode(["mz", "intensity"])
    set_dtypes(df)
    return df.reset_index(drop=True)[MS_FILE_COLUMNS]


def _extract_mzxml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant data from mzXML spectrum.

    Args:
        data: Dictionary containing mzXML spectrum data.

    Returns:
        Dictionary with extracted scan information.
    """
    return {
        "scan_id": data["num"],
        "ms_level": data["msLevel"],
        "polarity": data.get("polarity", None),
        "scan_time": data["retentionTime"],
        "mz": np.array(data["m/z array"]),
        "intensity": np.array(data["intensity array"]),
    }


def mzml_to_pandas_df_pyteomics(fn: Union[str, P], **kwargs) -> Optional[pd.DataFrame]:
    """Deprecated function to read mzML files.

    Args:
        fn: Filename or path to the mzML file.
        **kwargs: Additional arguments passed to mzml_to_df.

    Returns:
        DataFrame containing MS data, or None if read_only is True.
    """
    # warnings.warn("mzml_to_pandas_df_pyteomics() is deprecated use mzxml_to_df() instead", DeprecationWarning)
    return mzml_to_df(fn, **kwargs)


def mzml_to_df(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read mzML file and convert it to pandas DataFrame using the mzML library.

    Args:
        fn: Filename or path to the mzML file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if read_only is True.

    Raises:
        AssertionError: If the filename does not end with '.mzml'.
    """
    assert str(fn).lower().endswith(".mzml"), fn

    # Read mzML file using pyteomics
    with mzml.read(str(fn)) as reader:
        # Initialize empty lists for the slices and the attributes
        slices = []
        # Loop through the spectra and extract the data
        for spectrum in reader:
            time_unit = spectrum["scanList"]["scan"][0]["scan start time"].unit_info
            if read_only:
                continue
            # Extract the scan ID, retention time, m/z values, and intensity values
            scan_id = int(spectrum["id"].split("=")[-1])
            rt = spectrum["scanList"]["scan"][0]["scan start time"]
            if time_unit == "minute":
                rt = rt * 60.0
            mz = np.array(spectrum["m/z array"], dtype=np.float64)
            intensity = np.array(spectrum["intensity array"], dtype=np.float64)
            if "positive scan" in spectrum.keys():
                polarity = "+"
            elif "negative scan" in spectrum.keys():
                polarity = "-"
            else:
                polarity = None
            ms_level = spectrum["ms level"]
            slices.append(
                pd.DataFrame(
                    {
                        "scan_id": scan_id,
                        "mz": mz,
                        "intensity": intensity,
                        "polarity": polarity,
                        "ms_level": ms_level,
                        "scan_time": rt,
                    }
                )
            )
    if read_only:
        return None
    df = pd.concat(slices)
    df["intensity"] = df["intensity"].astype(int)
    df = df[MS_FILE_COLUMNS].reset_index(drop=True)
    set_dtypes(df)
    return df


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set appropriate data types for MS data columns.

    Args:
        df: DataFrame containing MS data.

    Returns:
        DataFrame with appropriate data types.
    """
    dtypes = dict(
        mz=np.float32,
        scan_id=np.int64,
        ms_level=np.int8,
        scan_time=np.float32,
        intensity=np.int64,
    )

    for var, dtype in dtypes.items():
        if var in df.columns and not df[var].dtype == dtype:
            df[var] = df[var].astype(dtype)

    return df


def _extract_mzml(data: Any, time_unit: str) -> Dict[str, Any]:
    """Extract relevant data from mzML spectrum.

    Args:
        data: Object containing mzML spectrum data.
        time_unit: Time unit used in the mzML file.

    Returns:
        Dictionary with extracted scan information.
    """
    RT = data.scan_time_in_minutes() * 60
    peaks = data.peaks("centroided")
    return {
        "scan_id": data["id"],
        "ms_level": data.ms_level,
        "polarity": "+" if data["positive scan"] else "-",
        "scan_time": RT,
        "mz": peaks[:, 0].astype("float64"),
        "intensity": peaks[:, 1].astype("int64"),
    }


extract_mzml = np.vectorize(_extract_mzml)


def read_parquet(fn: Union[str, P], read_only: bool = False) -> pd.DataFrame:
    """Read parquet file and return a pandas DataFrame.

    Args:
        fn: Filename or path to the parquet file.
        read_only: Whether to return the DataFrame as-is without formatting.
            Default is False.

    Returns:
        DataFrame containing MS data.
    """
    df = pd.read_parquet(fn)
    if read_only or (
        len(df.columns) == len(MS_FILE_COLUMNS) and all(df.columns == MS_FILE_COLUMNS)
    ):
        return df
    else:
        return format_thermo_raw_file_reader_parquet(df)


def format_thermo_raw_file_reader_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame from Thermo Raw File Reader to MS-MINT standard format.

    Args:
        df: DataFrame from Thermo Raw File Reader.

    Returns:
        Formatted DataFrame in MS-MINT standard format.
    """
    df = (
        df[["ScanNumber", "MsOrder", "RetentionTime", "Intensities", "Masses"]]
        .set_index(
            [
                "ScanNumber",
                "MsOrder",
                "RetentionTime",
            ]
        )
        .apply(pd.Series.explode)
        .reset_index()
        .rename(
            columns={
                "ScanNumber": "scan_id",
                "MsOrder": "ms_level",
                "RetentionTime": "scan_time",
                "Masses": "mz",
                "Intensities": "intensity",
            }
        )
    )
    df["scan_time"] = df["scan_time"] * 60
    df["polarity"] = None
    df["intensity"] = df.intensity.astype(np.float64)
    df = df[MS_FILE_COLUMNS]
    return df


def mzmlb_to_df__pyteomics(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read mzMLb file and convert it to pandas DataFrame using the pyteomics library.

    Args:
        fn: Filename or path to the mzMLb file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if read_only is True.
    """
    if not MZMLB_AVAILABLE:
        logging.error("mzmlb support is not available")
        return None

    with mzmlb.MzMLb(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzmlb(data))
    df = (
        pd.DataFrame.from_dict(data)
        .set_index(["index", "retentionTime", "polarity"])
        .apply(pd.Series.explode)
        .reset_index()
        .rename(
            columns={
                "index": "scan_id",
                "retentionTime": "scan_time",
                "m/z array": "mz",
                "ms level": "ms_level",
                "intensity array": "intensity",
            }
        )
    )

    # mzMLb starts scan index with 0
    df["scan_id"] = df["scan_id"] + 1

    df = df[MS_FILE_COLUMNS]
    return df


def _extract_mzmlb(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant data from mzMLb spectrum.

    Args:
        data: Dictionary containing mzMLb spectrum data.

    Returns:
        Dictionary with extracted scan information.
    """
    cols = ["index", "ms level", "polarity", "retentionTime", "m/z array", "intensity array"]
    data["retentionTime"] = data["scanList"]["scan"][0]["scan start time"] * 60
    if "positive scan" in data.keys():
        data["polarity"] = "+"
    elif "negative scan" in data.keys():
        data["polarity"] = "-"
    else:
        data["polarity"] = None
    return {c: data[c] for c in cols}


extract_mzmlb = np.vectorize(_extract_mzmlb)


def df_to_numeric(df: pd.DataFrame) -> None:
    """Convert dataframe columns to numeric types where possible.

    Args:
        df: DataFrame to convert. Modified in-place.
    """
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors="ignore")


def export_to_excel(
    mint: "ms_mint.Mint.Mint", fn: Optional[Union[str, P]] = None
) -> Optional[io.BytesIO]:
    """Export MINT state to Excel file.

    Args:
        mint: Mint instance containing data to export.
        fn: Output filename. If None, returns a file buffer instead of writing to disk.

    Returns:
        BytesIO buffer if fn is None, otherwise None.
    """
    date_string = str(date.today())
    if fn is None:
        file_buffer = io.BytesIO()
        writer = pd.ExcelWriter(file_buffer)
    else:
        writer = pd.ExcelWriter(fn)
    # Write into file
    mint.targets.to_excel(writer, "Targets", index=False)
    mint.results.to_excel(writer, "Results", index=False)
    meta = pd.DataFrame({"MINT_version": [mint.version], "Date": [date_string]}).T[0]
    meta.to_excel(writer, "Metadata", index=True, header=False)
    # Close writer and maybe return file buffer
    writer.close()
    if fn is None:
        file_buffer.seek(0)
        return file_buffer
    return None


def convert_ms_file_to_feather(fn: Union[str, P], fn_out: Optional[Union[str, P]] = None) -> str:
    """Convert MS file to feather format.

    Args:
        fn: Filename or path to the MS file to convert.
        fn_out: Output filename or path. If None, uses the same path with '.feather' extension.

    Returns:
        Path to the generated feather file.
    """
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".feather")
    df = ms_file_to_df(fn)
    if df is not None:
        df = df.reset_index(drop=True)
        df.to_feather(fn_out)
    return str(fn_out)


def convert_ms_file_to_parquet(fn: Union[str, P], fn_out: Optional[Union[str, P]] = None) -> str:
    """Convert MS file to parquet format.

    Args:
        fn: Filename or path to the MS file to convert.
        fn_out: Output filename or path. If None, uses the same path with '.parquet' extension.

    Returns:
        Path to the generated parquet file.
    """
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".parquet")
    df = ms_file_to_df(fn)
    if df is not None:
        df = df.reset_index(drop=True)
        df.to_parquet(fn_out)
    return str(fn_out)
