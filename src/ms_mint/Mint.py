"""Main module of the ms-mint library."""

import os
import numpy as np
import pandas as pd
import time
import logging

from pathlib import Path as P
from multiprocessing import Pool, Manager, cpu_count
from glob import glob

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from .standards import MINT_RESULTS_COLUMNS, TARGETS_COLUMNS, DEPRECATED_LABELS
from .processing import process_ms1_files_in_parallel, extract_chromatogram_from_ms1
from .io import export_to_excel, ms_file_to_df
from .TargetOptimizer import TargetOptimizer
from .targets import read_targets, check_targets, standardize_targets
from .tools import (
    is_ms_file,
    get_ms_files_from_results,
    get_targets_from_results,
    scale_dataframe,
    init_metadata,
    fn_to_label,
    log2p1
)
from .pca import PrincipalComponentsAnalyser
from .MintPlotter import MintPlotter
from .Chromatogram import Chromatogram

import ms_mint

from tqdm import tqdm

from typing import Callable
from functools import lru_cache

METADATA_DEFAUT_FN = 'metadata.parquet'

class Mint(object):
    """
    Main class of the ms_mint package, which processes metabolomics files.

    :param verbose: Sets verbosity of the instance.
    :type verbose: bool

    :param progress_callback: A callback for a progress bar.
    :type progress_callback: Callable[]

    :parm wdir: Working directory
    :type wdir: str
    """

    def __init__(
        self,
        verbose: bool = False,
        progress_callback: Callable = None,
        time_unit: str = "s",
        wdir: str = None
    ):
        self.verbose = verbose
        self._version = ms_mint.__version__
        if verbose: 
            print("Mint version:", self.version, "\n")
        self.progress_callback = progress_callback
        self.reset()
        self.plot = MintPlotter(mint=self)
        self.opt = TargetOptimizer(mint=self)
        self.pca = PrincipalComponentsAnalyser(self)
        self.tqdm = tqdm

        # Setup working directory as pathlib.Path
        self.wdir = os.getcwd() if wdir is None else wdir
        self.wdir = P(self.wdir)

    @property
    def version(self):
        """
        ms-mint version number.

        :return: Version string.
        :rtype: str
        """
        return self._version

    def reset(self):
        """
        Reset Mint instance. Removes targets, MS-files and results.

        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        self._files = []
        self._targets_files = []
        self._targets = pd.DataFrame(columns=TARGETS_COLUMNS)
        self._results = pd.DataFrame({i: [] for i in MINT_RESULTS_COLUMNS})
        self._all_df = None
        self._progress = 0
        self.runtime = None
        self._status = "waiting"
        self._messages = []
        self.meta = init_metadata()
        return self

    def clear_targets(self):
        """
        Reset target list.
        """
        self.targets = pd.DataFrame(columns=TARGETS_COLUMNS)

    def clear_results(self):
        """
        Reset results.
        """
        self.results = pd.DataFrame(columns=MINT_RESULTS_COLUMNS)

    def clear_ms_files(self):
        """
        Reset ms files.
        """
        self.ms_files = []

    def run(self, nthreads=None, rt_margin=0.5, mode="standard", fn=None, **kwargs):
        """
        Main routine to run MINT and process MS-files with current target list.

        :param nthreads: Number of cores to use, defaults to None
        :type nthreads: int
                * None - Run with min(n_cpus, c_files) CPUs
                * 1: Run without multiprocessing on one CPU
                * >1: Run with multiprocessing enabled using nthreads threads.
        :param mode: Compute mode ('standard' or 'express'), defaults to 'standard'
        :type mode: str
                * 'standard': calculates peak shaped projected to RT dimension
                * 'express': omits calculation of other features, only peak_areas
        :param fn: Output filename to not keep results in memory.
        :type fn: str
        :param kwargs: Arguments passed to the procesing function.
        """
        self._status = "running"

        if (self.n_files == 0) or (len(self.targets) == 0):
            return None

        targets = self.targets.reset_index()
        self._set_rt_min_max(targets, rt_margin)

        nthreads = self._determine_nthreads(nthreads)

        if self.verbose:
            print(f"Run MINT with {nthreads} processes:")

        start = time.time()
        if nthreads > 1:
            self._run_parallel(nthreads=nthreads, mode=mode, fn=fn, **kwargs)
        else:
            self._run_sequential(mode=mode, fn=fn, targets=targets)

        self.progress = 100
        self._report_runtime(start)

        self._status = "done"
        assert self.progress == 100
        return self

    def _set_rt_min_max(self, targets, rt_margin):
        if "rt" in targets.columns:
            update_rt_min = (targets.rt_min.isna()) & (~targets.rt.isna())
            targets.loc[update_rt_min, "rt_min"] = (
                targets.loc[update_rt_min, "rt"] - rt_margin
            )
            update_rt_max = (targets.rt_max.isna()) & (~targets.rt.isna())
            targets.loc[update_rt_max, "rt_max"] = (
                targets.loc[update_rt_max, "rt"] + rt_margin
            )

    def _determine_nthreads(self, nthreads):
        if nthreads is None:
            nthreads = min(cpu_count(), self.n_files)
        return nthreads

    def _run_sequential(self, mode, fn, targets):
        results = []
        for i, filename in enumerate(self.ms_files):
            args = {
                "filename": filename,
                "targets": targets,
                "q": None,
                "mode": mode,
                "output_fn": None,
            }
            results.append(process_ms1_files_in_parallel(args))
            self.progress = int(100 * (i / self.n_files))
        self.results = pd.concat(results).reset_index(drop=True)

    def _report_runtime(self, start):
        end = time.time()
        self.runtime = end - start
        self.runtime_per_file = self.runtime / self.n_files
        self.runtime_per_peak = self.runtime / self.n_files / len(self.targets)

        if self.verbose:
            print(f"Total runtime: {self.runtime:.2f}s")
            print(f"Runtime per file: {self.runtime_per_file:.2f}s")
            print(
                f"Runtime per peak ({len(self.targets)}): {self.runtime_per_peak:.2f}s\n"
            )
            print("Results:", self.results)

    def _run_parallel(
        self, nthreads=1, mode="standard", maxtasksperchild=None, fn=None
    ):
        pool = Pool(processes=nthreads, maxtasksperchild=maxtasksperchild)
        m = Manager()
        q = m.Queue()
        args = []

        if fn is not None:
            # Prepare output file (only headers)
            pd.DataFrame(columns=MINT_RESULTS_COLUMNS).to_csv(fn, index=False)

        for filename in self.ms_files:
            args.append(
                {
                    "filename": filename,
                    "targets": self.targets.reset_index(),
                    "queue": q,
                    "mode": mode,
                    "output_fn": fn,
                }
            )

        results = pool.map_async(process_ms1_files_in_parallel, args)
        self._monitor_progress(results, q)

        pool.close()
        pool.join()

        if fn is None:
            results = results.get()
            self.results = pd.concat(results).reset_index(drop=True)

    def _monitor_progress(self, results, q):
        while not results.ready():
            size = q.qsize()
            self.progress = 100 * size / self.n_files
            time.sleep(1)
        self.progress = 100

    @property
    def status(self):
        """
        Returns current status of Mint instance.

        :return: ['waiting', 'running', 'done']
        :rtype: str
        """
        return self._status

    @property
    def ms_files(self):
        """
        Get/set ms-files to process.

        :getter:
        :return: List of filenames.
        :rtype: list[str]

        :setter:
        :param list_of_files: Filename or list of file names of MS-files.
        :type list_of_files: str or list[str]
        """
        return self._files

    @ms_files.setter
    def ms_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        list_of_files = [str(P(i)) for i in list_of_files if is_ms_file(i)]
        for f in list_of_files:
            if not os.path.isfile(f):
                logging.warning(f"File not found ({f})")
        self._files = list_of_files
        if self.verbose:
            print("Set files to:\n" + "\n".join(self.ms_files) + "\n")
        self.meta = self.meta.reindex([fn_to_label(fn) for fn in list_of_files])

    @property
    def n_files(self):
        """
        Number of currently stored ms filenames.

        :return: Number of files stored in self.ms_files
        :rtype: int
        """
        return len(self.ms_files)

    def load_files(self, obj):
        """
        Load ms_files as a function that returns the Mint instance for chaining.

        :param list_of_files: Filename or list of file names.
        :type list_of_files: str or list[str]
        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        if isinstance(obj, str):
            self.ms_files = glob(obj, recursive=True)
        elif isinstance(obj, list):
            self.ms_files = obj
        return self

    def load_targets(self, list_of_files):
        """
        Load targets from a file (csv, xslx)

        :param list_of_files: Filename or list of file names.
        :type list_of_files: str or list[str]
        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        if isinstance(list_of_files, str) or isinstance(list_of_files, P):
            list_of_files = [list_of_files]
        if not isinstance(list_of_files, list):
            raise ValueError("Input should be a list of files.")
        for f in list_of_files:
            assert os.path.isfile(f), f"File not found ({f})"
        self._targets_files = list_of_files
        if self.verbose:
            print("Set targets files to:\n".join(self._targets_files) + "\n")
        self.targets = read_targets(list_of_files)
        return self

    @property
    def targets(self):
        """
        Set/get target list.

        :getter:
        :return: Target list
        :rtype: pandas.DataFrame

        :setter:
        :param targets: Sets the target list of the instance.
        :type targets: pandas.DataFrame
        """
        return self._targets

    @targets.setter
    def targets(self, targets):
        targets = standardize_targets(targets)
        assert check_targets(targets), check_targets(targets)
        self._targets = targets.set_index("peak_label")
        if self.verbose:
            print("Set targets to:\n", self.targets.to_string(), "\n")

    def get_target_params(self, peak_label):
        target_data = self.targets.loc[peak_label]
        mz_mean, mz_width, rt_min, rt_max = target_data[
            ["mz_mean", "mz_width", "rt_min", "rt_max"]
        ]
        return mz_mean, mz_width, rt_min, rt_max

    @property
    def peak_labels(self):
        return self.targets.index.to_list()

    @property
    def results(self):
        """
        Get/Set the Mint results.

        :getter:
        :return: Results
        :rtype: pandas.DataFrame
        :setter:
        :param df: DataFrame with MINT results.
        :type df: pandas.DataFrame
        """
        return self._results

    @results.setter
    def results(self, df):
        self._results = df

    def crosstab(self, var_name: str = None, index: str = None, column: str = None, aggfunc: str = 'mean', 
                 apply: Callable = None, scaler: Callable = None, groupby: str = None):
        """
        Create condensed representation of the results.
        More specifically, a cross-table with filenames as index and target labels.
        The values in the cells are determined by *col_name*.

        :param var_name: Name of the column from *mint.results* table that is used for the cell values. If None, defaults to 'peak_area_top3'.
        :type var_name: str, optional

        :param index: Name of the column to be used as index in the resulting cross-tabulation. If None, defaults to 'ms_file_label'.
        :type index: str, optional

        :param column: Name of the column to be used as columns in the resulting cross-tabulation. If None, defaults to 'peak_label'.
        :type column: str, optional

        :param aggfunc: Aggregation function to be used for aggregating values. Defaults to 'mean'.
        :type aggfunc: str, optional

        :param apply: Function to be applied to the resulting cross-tabulation. If None, no function is applied.
        :type apply: Callable, optional

        :param scaler: Function to scale the data in the resulting cross-tabulation. If None, no scaling is performed.
        :type scaler: Callable, optional

        :param groupby: Name of the column to group data before scaling. If None, scaling is applied to the whole data, not group-wise.
        :type groupby: str, optional

        :return: DataFrame representing the cross-tabulation.
        :rtype: pandas.DataFrame
        """
        df_meta = pd.merge(self.meta, self.results, left_index=True, right_on='ms_file_label')
        # Remove None if in index
        if isinstance(index, list):
            if None in index:
                index.remove(None)
        if isinstance(groupby, str):
            groupby = [groupby]
            
        if index is None:
            index = 'ms_file_label'
        if column is None:
            column = 'peak_label'
        if var_name is None:
            var_name = 'peak_area_top3'
        if apply:
            if apply == 'log2p1':
                apply = log2p1
            if apply == 'logp1':
                apply = np.log1p
            df_meta[var_name] = df_meta[var_name].apply(apply)
        if isinstance(scaler, str):
            scaler_dict = {'standard': StandardScaler(),
                           'robust': RobustScaler(),
                           'minmax': MinMaxScaler()}

            if scaler not in scaler_dict:
                raise ValueError(f"Unsupported scaler: {scaler}")

            scaler = scaler_dict[scaler]
            
        if scaler:
            if groupby:          
                groupby_cols = groupby + [column]
                df_meta[var_name] = df_meta.groupby(groupby_cols)[var_name].transform(lambda x: self._scale_group(x, scaler))
            else:
                df_meta[var_name] = df_meta.groupby(column)[var_name].transform(lambda x: self._scale_group(x, scaler))
                
        df = pd.pivot_table(
            df_meta,
            index=index,
            columns=column,
            values=var_name,
            aggfunc=aggfunc,
        ).astype(np.float64)
        return df
    
    @property
    def progress(self):
        """
        Shows the current progress.

        :getter: Returns the current progress value.
        :setter: Set the progress to a value between 0 and 100 and calls the progress callback function.
        """
        return self._progress

    @progress.setter
    def progress(self, value):
        assert value >= 0, value
        assert value <= 100, value
        self._progress = value
        if self.progress_callback is not None:
            self.progress_callback(value)

    def export(self, fn=None):
        """
        Export current results to file.

        :param fn: Filename, defaults to None
        :type fn: str, optional
        :param filename: **deprecated**
        :type filename: str, optional
        :return: file buffer if *filename* is None otherwise returns None
        :rtype: io.BytesIO
        """
        if fn is None:
            buffer = export_to_excel(self, fn=fn)
            return buffer
        elif fn.endswith(".xlsx"):
            export_to_excel(self, fn=fn)
        elif fn.endswith(".csv"):
            self.results.to_csv(fn, index=False)
        elif fn.endswith(".parquet"):
            self.results.to_parquet(fn, index=False)

    def load(self, fn):
        """
        Load results into Mint instance.

        :param fn: Filename (csv, xlsx)
        :type fn: str
        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        if self.verbose:
            print(f"Loading MINT results from {fn}")

        if isinstance(fn, str):
            if fn.endswith("xlsx"):
                results = pd.read_excel(fn, sheet_name="Results")
                self.results = results

            elif fn.endswith(".csv"):
                results = pd.read_csv(fn)
                results["peak_shape_rt"] = results["peak_shape_rt"].fillna("")
                results["peak_shape_int"] = results["peak_shape_int"].fillna("")
                self.results = results

            elif fn.endswith(".parquet"):
                results = pd.read_parquet(fn)
        else:
            results = pd.read_csv(fn)

        # Add file labels if not present already    
        if 'ms_file_label' not in results.columns:
            results['ms_file_label'] = [fn_to_label(fn) for fn in results.ms_file]

        self.results = results.rename(columns=DEPRECATED_LABELS)
        self.digest_results()
        return self

    def digest_results(self):
        self.ms_files = get_ms_files_from_results(self.results)
        self.targets = get_targets_from_results(self.results)


    def get_chromatograms(self, fns=None, peak_labels=None, filters=None, **kwargs):
        if fns is None:
            fns = self.ms_files
        if peak_labels is None:
            peak_labels = self.peak_labels
        return self._get_chromatograms(fns=tuple(fns), peak_labels=tuple(peak_labels), 
                                       filters=tuple(filters) if filters is not None else None, **kwargs)

    @lru_cache(1)
    def _get_chromatograms(self, fns=None, peak_labels=None, filters=None, **kwargs):

        if isinstance(fns, tuple):
            fns = list(fns)     

        if not isinstance(fns, list):
            fns = [fns]

        labels = [fn_to_label(fn) for fn in fns]

        # Need to get the actual file names with get_chromatogramsath
        # in case only ms_file_labels are provided
        fns = [fn for fn in self.ms_files if fn_to_label(fn) in labels]

        data = []

        for fn in self.tqdm(fns, desc="Loading chromatograms"):
            df = ms_file_to_df(fn)
            for label in peak_labels:
                mz_mean, mz_width, rt_min, rt_max = self.get_target_params(label)
                chrom_raw = extract_chromatogram_from_ms1(
                    df, mz_mean=mz_mean, mz_width=mz_width
                ).to_frame()
                if len(chrom_raw) == 0:
                    continue
                chrom = Chromatogram(chrom_raw.index, chrom_raw.values, filters=filters, **kwargs)
                if filters is not None:
                    chrom.apply_filters()
                chrom_data = chrom.data
                chrom_data["ms_file"] = fn
                chrom_data["ms_file_label"] = fn_to_label(fn)
                chrom_data["peak_label"] = label
                chrom_data["rt_min"] = rt_min
                chrom_data["rt_max"] = rt_max
                data.append(chrom_data)

        data = pd.concat(data).reset_index()

        data["ms_file"] = data["ms_file"].apply(lambda x: P(x).with_suffix("").name)
        return data

    def load_metadata(self, fn=None):
        if fn is None:
            fn = self.wdir/METADATA_DEFAUT_FN
        if str(fn).endswith('.csv'):
            self.meta = pd.read_csv(fn, index_col=0)
        elif str(fn).endswith('.parquet'):
            self.meta = pd.read_parquet(fn)
        if 'ms_file_label' in self.meta.columns:
            self.meta = self.meta.set_index('ms_file_label')
        return self

    def save_metadata(self, fn=None):
        if fn is None:
            fn = self.wdir/METADATA_DEFAUT_FN
        if str(fn).endswith('.csv'):
            self.meta.to_csv(fn, na_filter=False)
        elif str(fn).endswith('.parquet'):
            self.meta.to_parquet(fn)
        return self

    def _scale_group(self, group, scaler):
        """
        Helper function to scale groups individually.
        """
        return scaler.fit_transform(group.to_numpy().reshape(-1, 1)).flatten()
