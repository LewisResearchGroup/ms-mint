# ms_mint/io.py

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
except:
    logging.warning("Cound not import pyteomics.mzmlb")
    MZMLB_AVAILABLE = False


MS_FILE_COLUMNS = [
    "scan_id",
    "ms_level",
    "polarity",
    "scan_time_min",
    "mz",
    "intensity",
]


def ms_file_to_df(fn, read_only: bool = False, time_unit="seconds"):
    assert time_unit in ["minutes", "seconds"]
    fn = str(fn)
    if fn.lower().endswith(".mzxml"):
        df = mzxml_to_df(fn, read_only=read_only)
    elif fn.lower().endswith(".mzml"):
        df = mzml_to_df(fn, read_only=read_only, time_unit=time_unit)
    elif fn.lower().endswith("hdf"):
        df = pd.read_hdf(fn)
    elif fn.lower().endswith(".feather"):
        df = pd.read_feather(fn)
    elif fn.lower().endswith(".parquet"):
        df = read_parquet(fn, read_only=read_only)
    elif fn.lower().endswith(".mzmlb"):
        df = mzmlb_to_df__pyteomics(fn, read_only=read_only)
    else:
        logging.error(f'Cannot read file {fn} of type {type(fn)}')

    # Compatibility with old schema
    if not read_only:
        df = df.rename(
            columns={
                "retentionTime": "scan_time_min",
                "intensity array": "intensity",
                "m/z array": "mz",
            }
        )
    return df


def mzxml_to_df(fn, read_only=False):
    """
    Reads mzXML file and returns a pandas.DataFrame
    using pyteomics library.
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

    df["retentionTime"] = df["retentionTime"].astype(np.float64)
    df["m/z array"] = df["m/z array"].astype(np.float64)
    df["intensity array"] = df["intensity array"].astype(np.float64)

    df = df.rename(
        columns={
            "num": "scan_id",
            "msLevel": "ms_level",
            "retentionTime": "scan_time_min",
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


def mzml_to_pandas_df_pyteomics(fn, read_only=False):
    """
    Reads mzML file and returns a pandas.DataFrame
    using the pyteomics library. Needs refactoring.
    """
    slices = []
    with mzml.MzML(fn) as ms_data:

        while True:

            try:
                data = ms_data.next()
            except Exception as e:
                logging.warning(e)
                break

            scan = data["scanList"]["scan"][0]

            if "positive scan" in data.keys():
                data["polarity"] = "+"

            elif "negative scan" in data.keys():
                data["polarity"] = "-"
            else:
                data["polarity"] = None

            if "scan start time" in scan.keys():
                data["scan_time_min"] = scan["scan start time"] / 60
            elif "scan time" in scan.keys():
                data["scan_time_min"] = scan["scan time"] / 60
            else:
                logging.error(f"No scan time identified \n{scan}")
                break

            del data["scanList"]
            slices.append(pd.DataFrame(data))

    df = pd.concat(slices)

    df["scan_id"] = df["id"].apply(lambda x: int(x.split("scan=")[1].split(" ")[0]))
    df_to_numeric(df)
    df["intensity array"] = df["intensity array"].astype(int)

    mapping = {
        "m/z array": "mz",
        "intensity array": "intensity",
        "ms level": "ms_level",
    }

    df = df.rename(columns=mapping)
    df = df.reset_index(drop=True)
    df = df[MS_FILE_COLUMNS]
    return df


def mzml_to_df(fn, time_unit="seconds", read_only=False):

    with pymzml.run.Reader(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzml(data, time_unit=time_unit))

    df = (
        pd.DataFrame.from_dict(data)
        .set_index(["scan_id", "ms_level", "polarity", "scan_time_min"])
        .apply(pd.Series.explode)
        .reset_index()
    )

    df["mz"] = df["mz"].astype("float64")
    df["intensity"] = df["intensity"].astype("float64")
    return df


def _extract_mzml(data, time_unit):
    try:
        RT = data.scan_time_in_minutes()
    except:
        if time_unit == "seconds":
            RT = data.scan_time[0] / 60.0
        elif time_unit == "minutes":
            RT = data.scan_time[0]
    peaks = data.peaks("centroided")
    return {
        "scan_id": data["id"],
        "ms_level": data.ms_level,
        "polarity": "+" if data["positive scan"] else "-",
        "scan_time_min": RT,
        "mz": peaks[:, 0].astype("float64"),
        "intensity": peaks[:, 1].astype("float64"),
    }


extract_mzml = np.vectorize(_extract_mzml)


def read_parquet(fn, read_only=False):
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
                "RetentionTime": "scan_time_min",
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
                "retentionTime": "scan_time_min",
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
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".feather")
    df = ms_file_to_df(fn).reset_index(drop=True)
    df.to_feather(fn_out)
    return fn_out


def convert_ms_file_to_parquet(fn, fn_out=None):
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".parquet")
    df = ms_file_to_df(fn).reset_index(drop=True)
    df.to_parquet(fn_out)
    return fn_out
