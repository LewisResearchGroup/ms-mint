"""
Main module of the ms-mint library.
"""

import os
import numpy as np
import pandas as pd
import time
import logging

from pathlib import Path as P

from sklearn.decomposition import PCA

from multiprocessing import Pool, Manager, cpu_count

from ms_mint.PlotGenerator import PlotGenerator

from .standards import MINT_RESULTS_COLUMNS, TARGETS_COLUMNS, DEPRECATED_LABELS
from .processing import process_ms1_files_in_parallel
from .io import export_to_excel
from .targets import read_targets, check_targets, standardize_targets
from .tools import scale_dataframe, is_ms_file, get_ms_files_from_results


import ms_mint


from typing import Callable


class Mint(object):
    """Main class of the ms_mint package, which processes metabolomics files.

    :param verbose: Sets verbosity of the instance.
    :type verbose: bool

    :param progress_callback: A callback for a progress bar.
    :type progress_callback: Callable[]

    """

    def __init__(self, verbose: bool = False, progress_callback=None):

        self._verbose = verbose
        self._version = ms_mint.__version__
        self._progress_callback = progress_callback
        self.reset()
        if self.verbose:
            print("Mint Version:", self.version, "\n")
        self.plot = PlotGenerator(self)

    @property
    def verbose(self):
        """Get/set verbosity.

        :getter: Get current verbosity.
        :return: True or False
        :rtype: bool
        :setter: Sets verbosity.
        :param value: True or False
        :type value: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = value

    @property
    def version(self):
        """ms-mint version number.

        :return: Version string.
        :rtype: str
        """
        return self._version

    def reset(self):
        """Reset Mint instance. Removes targets, MS-files and results.

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

    def run(self, nthreads=None, rt_margin=0.5, mode="standard", **kwargs):
        """
        Main routine to run MINT and process MS-files with current target list.

        :param nthreads: Number of cores to use, defaults to None
        :type nthreads: int
                * None - Run with min(n_cpus, c_files) CPUs
                * 1: Run without multiprocessing on one CPU
                * >1: Run with multiprocessing enabled using nthreads threads.
        :param mode: Compute mode ('standard' or 'express'), defaults to 'standard'
                * 'standard': calculates peak shaped projected to RT dimension
                * 'express': omits calculation of other features, only peak_areas
        :type mode: str
        :param kwargs: Arguments passed to the procesing function.
        """
        self._status = "running"

        if (self.n_files == 0) or (len(self.targets) == 0):
            return None

        targets = self.targets
        if "rt" in targets.columns:
            ndx = (targets.rt_min.isna()) & (~targets.rt.isna())
            targets.loc[ndx, "rt_min"] = targets.loc[ndx, "rt"] - rt_margin
            ndx = (targets.rt_max.isna()) & (~targets.rt.isna())
            targets.loc[ndx, "rt_max"] = targets.loc[ndx, "rt"] + rt_margin
            del ndx

        if nthreads is None:
            nthreads = min(cpu_count(), self.n_files)

        if self.verbose:
            print(f"Run MINT with {nthreads} processes:")

        start = time.time()
        if nthreads > 1:
            self.__run_parallel__(nthreads=nthreads, mode=mode, **kwargs)
        else:
            results = []
            for i, filename in enumerate(self.ms_files):
                args = {
                    "filename": filename,
                    "targets": self.targets,
                    "q": None,
                    "mode": mode,
                    "output_fn": None,
                }
                results.append(process_ms1_files_in_parallel(args))
                self.progress = int(100 * (i / self.n_files))
            self.results = pd.concat(results).reset_index(drop=True)

        self.progress = 100

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

        self._status = "done"
        assert self.progress == 100
        return self

    def __run_parallel__(
        self, nthreads=1, mode="standard", maxtasksperchild=None, output_fn=None
    ):
        print(f"maxtasksperchild: {maxtasksperchild}")
        pool = Pool(processes=nthreads, maxtasksperchild=maxtasksperchild)
        m = Manager()
        q = m.Queue()
        args = []

        if output_fn is not None:
            # Prepare output file (only headers)
            pd.DataFrame(columns=MINT_RESULTS_COLUMNS).to_csv(output_fn, index=False)

        for i, filename in enumerate(self.ms_files):
            args.append(
                {
                    "filename": filename,
                    "targets": self.targets,
                    "queue": q,
                    "mode": mode,
                    "output_fn": output_fn,
                }
            )

        results = pool.map_async(process_ms1_files_in_parallel, args)

        # monitor progress
        while True:
            if results.ready():
                break
            else:
                size = q.qsize()
                self.progress = 100 * size / self.n_files
                time.sleep(1)

        self.progress = 100

        pool.close()
        pool.join()

        if output_fn is None:
            results = results.get()
            self.results = pd.concat(results).reset_index(drop=True)

    @property
    def status(self):
        """Returns current status of Mint instance.

        :return: ['waiting', 'running', 'done']
        :rtype: str
        """
        return self._status

    @property
    def ms_files(self):
        """Get/set ms-files to process.

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

    @property
    def n_files(self):
        """Number of currently stored ms filenames.

        :return: Number of files stored in self.ms_files
        :rtype: int
        """
        return len(self.ms_files)

    def load_targets(self, list_of_files):
        """Load targets from a file (csv, xslx)

        :param list_of_files: Filename or list of file names.
        :type list_of_files: str or list[str]
        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        if not isinstance(list_of_files, list):
            raise ValueError("Input should be a list of files.")
        for f in list_of_files:
            assert os.path.isfile(f), f"File not found ({f})"
        self._targets_files = list_of_files
        if self.verbose:
            print("Set targets files to:\n".join(self.targets_files) + "\n")
        self.targets = read_targets(list_of_files)
        return self

    @property
    def targets(self):
        """Set/get target list.

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
        assert check_targets(targets)
        self._targets = targets
        if self.verbose:
            print("Set targets to:\n", self.targets.to_string(), "\n")

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

    def crosstab(self, col_name="peak_area"):
        """
        Create condensed representation of the results.
        More specifically, a cross-table with filenames as index and target labels.
        The values in the cells are determined by *col_name*.


        :param col_name: Name of the column from *mint.results* table that is used for the cell values.
        :type col_name: str

        cells of the returned table.
        """
        return pd.crosstab(
            self.results.ms_file,
            self.results.peak_label,
            self.results[col_name],
            aggfunc=sum,
        ).astype(np.float64)

    @property
    def progress_callback(self):
        """Assigns a callback function to update a progress bar.

        :getter: Returns the current callback function.
        :setter: Sets the callback function.
        """
        return self._progress_callback

    @progress_callback.setter
    def progress_callback(self, func: Callable = None):
        self._progress_callback = func

    @property
    def progress(self):
        """Shows the current progress.

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

    def export(self, fn=None, filename=None):
        """Export current results to file.

        :param fn: Filename, defaults to None
        :type fn: str, optional
        :param filename: **deprecated**
        :type filename: str, optional
        :return: file buffer if *filename* is None otherwise returns None
        :rtype: io.BytesIO
        """

        if filename is not None:
            fn = filename
            raise DeprecationWarning("'filename' is deprecated use 'fn' instead")
        if fn is None:
            buffer = export_to_excel(self, fn=fn)
            return buffer
        elif fn.endswith(".xlsx"):
            export_to_excel(self, fn=fn)
        elif fn.endswith(".csv"):
            self.results.to_csv(fn, index=False)

    def load(self, fn):
        """Load results into Mint instance.

        :param fn: Filename (csv, xlsx)
        :type fn: str
        :return: self
        :rtype: ms_mint.Mint.Mint
        """
        if self.verbose:
            print("Loading MINT state")
        if isinstance(fn, str):
            if fn.endswith("xlsx"):
                results = pd.read_excel(fn, sheet_name="Results").rename(
                    columns=DEPRECATED_LABELS
                )
                self.results = results
                self.targets = pd.read_excel(fn, sheet_name="Peaklist")
                self.ms_files = get_ms_files_from_results(results)

            elif fn.endswith(".csv"):
                results = pd.read_csv(fn).rename(columns=DEPRECATED_LABELS)
                results["peak_shape_rt"] = results["peak_shape_rt"].fillna("")
                results["peak_shape_int"] = results["peak_shape_int"].fillna("")
                ms_files = get_ms_files_from_results(results)
                targets = results[
                    [col for col in TARGETS_COLUMNS if col in results.columns]
                ].drop_duplicates()
                self.results = results
                self.ms_files = ms_files
                self.targets = targets
                return None
        else:
            results = pd.read_csv(fn).rename(columns=DEPRECATED_LABELS)
            if "ms_file" in results.columns:
                ms_files = get_ms_files_from_results(results)
                self.results = results
                self.ms_files = ms_files
            targets = results[
                [col for col in TARGETS_COLUMNS if col in results.columns]
            ].drop_duplicates()
            self.targets = targets
            return self

    def pca(
        self, var_name="peak_max", n_components=3, fillna="median", scaler="standard"
    ):
        """Run Principal Component Analysis on current results. Results are stored in
        self.decomposition_results.

        :param var_name: Column name to use for pca, defaults to "peak_max"
        :type var_name: str, optional
        :param n_components: Number of PCA components to return, defaults to 3
        :type n_components: int, optional
        :param fillna: Method to fill missing values, defaults to "median"
        :type fillna: str, optional
        :param scaler: Method to scale the columns, defaults to "standard"
        :type scaler: str, optional
        """

        df = self.crosstab(var_name).fillna(fillna)

        if fillna == "median":
            fillna = df.median()
        elif fillna == "mean":
            fillna = df.mean()
        elif fillna == "zero":
            fillna = 0

        df = df.fillna(fillna)
        if scaler is not None:
            df = scale_dataframe(df, scaler)

        min_dim = min(df.shape)
        n_components = min(n_components, min_dim)
        pca = PCA(n_components)
        X_projected = pca.fit_transform(df)
        # Convert to dataframe
        df_projected = pd.DataFrame(X_projected, index=df.index.get_level_values(0))
        # Set columns to PC-1, PC-2, ...
        df_projected.columns = [f"PC-{int(i)+1}" for i in df_projected.columns]

        # Calculate cumulative explained variance in percent
        explained_variance = pca.explained_variance_ratio_ * 100
        cum_expl_var = np.cumsum(explained_variance)

        # Create feature contributions
        a = np.zeros((n_components, n_components), int)
        np.fill_diagonal(a, 1)
        dfc = pd.DataFrame(pca.inverse_transform(a))
        dfc.columns = df.columns
        dfc.index = [f"PC-{i+1}" for i in range(n_components)]
        dfc.index.name = "PC"
        # convert to long format
        dfc = dfc.stack().reset_index().rename(columns={0: "Coefficient"})

        self.decomposition_results = {
            "df_projected": df_projected,
            "cum_expl_var": cum_expl_var,
            "n_components": n_components,
            "type": "PCA",
            "feature_contributions": dfc,
            "class": pca,
        }
