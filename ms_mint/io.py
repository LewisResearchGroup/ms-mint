"""
Funtions to read and write MINT files.
"""

import pandas as pd
import numpy as np
import io
import logging
import pymzml

from pathlib import Path as P
from datetime import date
from pyteomics import mzxml, mzml

try:
    from pyteomics import mzmlb

    MZMLB_AVAILABLE = True
except ImportError:
    logging.warning("Cound not import pyteomics.mzmlb")
    MZMLB_AVAILABLE = False


MS_FILE_COLUMNS = [
    "scan_id",
    "ms_level",
    "polarity",
    "scan_time",
    "mz",
    "intensity",
]


def ms_file_to_df(fn, read_only: bool = False, time_unit="seconds"):
    """
    Read MS file and convert it to a pandas.DataFrame.

    :param fn: Filename
    :type fn: str or PosixPath
    :param read_only: Whether or not to apply convert to dataframe (for testing purposes), defaults to False
    :type read_only: bool, optional
    :param time_unit: Time unit used in file, defaults to "seconds"
    :type time_unit: str, ['seconds', 'minutes']
    :return: MS data as DataFrame
    :rtype: pandas.DataFrame
    """

    assert time_unit in ["minutes", "seconds"]
    fn = str(fn)

    try:
        if fn.lower().endswith(".mzxml"):
            df = mzxml_to_df(fn, read_only=read_only)
        elif fn.lower().endswith(".mzml"):
            df = mzml_to_pandas_df_pyteomics(fn, time_unit=time_unit)
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
    except IndexError as e:
        logging.warning(f"{e}: {fn}")
        return None
    # Compatibility with old schema
    if not read_only:
        df = df.rename(
            columns={
                "retentionTime": "scan_time",
                "intensity array": "intensity",
                "m/z array": "mz",
            }
        )
    return df


def mzxml_to_df(fn, read_only=False):
    """
    Read mzXML file and convert it to pandas.DataFrame.

    :param fn: Filename
    :type fn: str or PosixPath
    :param read_only: Whether or not to convert to dataframe (for testing purposes), defaults to False
    :type read_only: bool, optional
    :return: MS data
    :rtype: pandas.DataFrame
    """

    with mzxml.MzXML(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzxml(data))

    df = (
        pd.DataFrame.from_dict(data)
        .set_index("retentionTime")
        .apply(pd.Series.explode)
        .reset_index()
    )

    df["retentionTime"] = df["retentionTime"].astype(np.float64) * 60.0
    df["m/z array"] = df["m/z array"].astype(np.float64)
    df["intensity array"] = df["intensity array"].astype(np.float64)

    df = df.rename(
        columns={
            "num": "scan_id",
            "msLevel": "ms_level",
            "retentionTime": "scan_time",
            "m/z array": "mz",
            "intensity array": "intensity",
        }
    )

    df = df.reset_index(drop=True)
    df = df[MS_FILE_COLUMNS]
    return df


def _extract_mzxml(data):
    cols = [
        "num",
        "msLevel",
        "polarity",
        "retentionTime",
        "m/z array",
        "intensity array",
    ]
    return {c: data[c] for c in cols}


extract_mzxml = np.vectorize(_extract_mzxml)


def mzml_to_pandas_df_pyteomics(fn):
    """
    Reads mzML file and returns a pandas.DataFrame using the pyteomics library.
    :param fn: Filename
    :type fn: str or PosixPath
    :return: MS data
    :rtype: pandas.DataFrame
    """
    # Read mzML file using pyteomics
    with mzml.read(str(fn)) as reader:

        # Initialize empty lists for the slices and the attributes
        slices = []

        # Loop through the spectra and extract the data
        for spectrum in reader:

            # Extract the scan ID, retention time, m/z values, and intensity values
            scan_id = int(spectrum["id"].split("scan=")[-1])
            rt = spectrum["scanList"]["scan"][0]["scan start time"]
            mz = np.array(spectrum["m/z array"], dtype=np.float64)
            intensity = np.array(spectrum["intensity array"], dtype=np.float64)
            if "positive scan" in spectrum.keys():
                polarity = "+"
            elif "negative scan" in spectrum.keys():
                polarity = "-"
            else:
                polarity = None
            ms_level = spectrum["ms level"]
            slices.append(pd.DataFrame({"scan_id": scan_id, "mz": mz, "intensity": intensity, "polarity": polarity, "ms_level": ms_level, "scan_time": rt}))

    df = pd.concat(slices)
    df["intensity"] = df["intensity"].astype(int)
    df = df[MS_FILE_COLUMNS]

    return df.reset_index(drop=True)


def mzml_to_df(fn, time_unit="seconds", read_only=False):
    """
    Reads mzML file and returns a pandas.DataFrame
    using the mzML library.

    :param fn: Filename
    :type fn: str or PosixPath
    :param read_only: Whether or not to convert to dataframe, defaults to False
    :type read_only: bool, optional
    :return: MS data
    :rtype: pandas.DataFrame
    """
    with pymzml.run.Reader(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzml(data, time_unit=time_unit))

    df = (
        pd.DataFrame.from_dict(data)
        .set_index(["scan_id", "ms_level", "polarity", "scan_time"])
        .apply(pd.Series.explode)
        .reset_index()
    )

    df["mz"] = df["mz"].astype("float64")
    df["intensity"] = df["intensity"].astype("float64")
    df["scan_time"] = df["scan_time"] * 60.0
    return df


def _extract_mzml(data, time_unit):
    try:
        RT = data.scan_time_in_minutes()
    except Exception:
        if time_unit == "seconds":
            RT = data.scan_time[0]
        elif time_unit == "minutes":
            RT = data.scan_time[0] * 60.0
    peaks = data.peaks("centroided")
    return {
        "scan_id": data["id"],
        "ms_level": data.ms_level,
        "polarity": "+" if data["positive scan"] else "-",
        "scan_time": RT,
        "mz": peaks[:, 0].astype("float64"),
        "intensity": peaks[:, 1].astype("float64"),
    }


extract_mzml = np.vectorize(_extract_mzml)


def read_parquet(fn, read_only=False):
    """
    Reads parquet file and returns a pandas.DataFrame.

    :param fn: Filename
    :type fn: str or PosixPath
    :param read_only: Whether or not to convert to dataframe, defaults to False
    :type read_only: bool, optional
    :return: MS data
    :rtype: pandas.DataFrame
    """
    df = pd.read_parquet(fn)
    if read_only or (
        len(df.columns) == len(MS_FILE_COLUMNS) and all(df.columns == MS_FILE_COLUMNS)
    ):
        return df
    else:
        return format_thermo_raw_file_reader_parquet(df)


def format_thermo_raw_file_reader_parquet(df):
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
    df["polarity"] = None
    df["intensity"] = df.intensity.astype(np.float64)
    df = df[MS_FILE_COLUMNS]
    return df


def mzmlb_to_df__pyteomics(fn, read_only=False):
    """
    Reads mzMLb file and returns a pandas.DataFrame
    using the pyteomics library.

    :param fn: Filename
    :type fn: str or PosixPath
    :param read_only: Whether or not to convert to dataframe, defaults to False
    :type read_only: bool, optional
    :return: MS data
    :rtype: pandas.DataFrame
    """
    with mzmlb.MzMLb(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzmlb(data))

    df = (
        pd.DataFrame.from_dict(data)
        .set_index("retentionTime")
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

    df["polarity"] = None
    df = df[MS_FILE_COLUMNS]
    return df


def _extract_mzmlb(data):
    cols = ["index", "ms level", "retentionTime", "m/z array", "intensity array"]
    data["retentionTime"] = data["scanList"]["scan"][0]["scan start time"] / 60
    return {c: data[c] for c in cols}


extract_mzmlb = np.vectorize(_extract_mzmlb)


def df_to_numeric(df):
    """
    Converts dataframe to numeric types if possible.
    """
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors="ignore")


def export_to_excel(mint, fn=None):
    """
    Export MINT state to Excel file.

    :param mint: Mint instance
    :type mint: ms_mint.Mint.Mint
    :return: None, or file buffer (if fn is None)
    :rtype: None or io.BytesIO
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


def convert_ms_file_to_feather(fn, fn_out=None):
    """
    Convert MS file to feather format.

    :param fn: Filename to convert
    :type fn: str or PosixPath
    :param fn_out: Output filename, defaults to None
    :type fn_out: str or PosixPath, optional
    :return: Filename of generated file
    :rtype: str
    """
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".feather")
    df = ms_file_to_df(fn).reset_index(drop=True)
    df.to_feather(fn_out)
    return fn_out


def convert_ms_file_to_parquet(fn, fn_out=None):
    """
    Convert MS file to parquet format.

    :param fn: Filename to convert
    :type fn: str or PosixPath
    :param fn_out: Output filename, defaults to None
    :type fn_out: str or PosixPath, optional
    :return: Filename of generated file
    :rtype: str
    """
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".parquet")
    df = ms_file_to_df(fn).reset_index(drop=True)
    df.to_parquet(fn_out)
    return fn_out
