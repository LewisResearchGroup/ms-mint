#src/ms_mint/Mint.py
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
from io import BytesIO
from tqdm import tqdm
from typing import Callable, Optional, Union, List, Dict, Tuple, Any, Set, Iterable
from functools import lru_cache

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
    log2p1,
)
from .pca import PrincipalComponentsAnalyser
from .MintPlotter import MintPlotter
from .Chromatogram import Chromatogram

import ms_mint

METADATA_DEFAUT_FN = "metadata.parquet"


class Mint:
    """Main class of the ms_mint package for processing metabolomics files.

    This class provides the primary interface for extracting, processing, and
    analyzing mass spectrometry data for metabolomics analysis.

    Attributes:
        verbose: Controls the verbosity level of the instance.
        version: The version of the ms_mint package being used.
        progress_callback: Function to update progress information.
        plot: Instance of MintPlotter for visualization.
        opt: Instance of TargetOptimizer for target optimization.
        pca: Instance of PrincipalComponentsAnalyser for PCA analysis.
        tqdm: Progress bar utility.
        wdir: Working directory for input/output operations.
        status: Current status of processing ('waiting', 'running', 'done').
        ms_files: List of MS files to be processed.
        n_files: Number of MS files currently loaded.
        targets: DataFrame with target compounds information.
        results: DataFrame with analysis results.
        progress: Current progress of processing (0-100).
    """

    def __init__(
        self,
        verbose: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
        time_unit: str = "s",
        wdir: Optional[Union[str, P]] = None,
    ) -> None:
        """Initialize a Mint instance.

        Args:
            verbose: Sets verbosity of the instance.
            progress_callback: A callback function for reporting progress (0-100).
            time_unit: Unit for time measurements.
            wdir: Working directory. If None, uses current directory.
        """
        self.verbose = verbose
        self._version = ms_mint.__version__
        if verbose:
            print(f"Mint version: {self.version}\n")
        self.progress_callback = progress_callback
        self.reset()
        self.plot = MintPlotter(mint=self)
        self.opt = TargetOptimizer(mint=self)
        self.pca = PrincipalComponentsAnalyser(self)
        self.tqdm = tqdm

        # Setup working directory as pathlib.Path
        self.wdir = P(os.getcwd() if wdir is None else wdir)

    @property
    def version(self) -> str:
        """Get the ms-mint version number.

        Returns:
            Version string.
        """
        return self._version

    def reset(self) -> "Mint":
        """Reset Mint instance by removing targets, MS-files and results.

        Returns:
            Self for method chaining.
        """
        self._files: List[str] = []
        self._targets_files: List[str] = []
        self._targets: pd.DataFrame = pd.DataFrame(columns=TARGETS_COLUMNS)
        self._results: pd.DataFrame = pd.DataFrame({i: [] for i in MINT_RESULTS_COLUMNS})
        self._all_df: Optional[pd.DataFrame] = None
        self._progress: float = 0
        self.runtime: Optional[float] = None
        self._status: str = "waiting"
        self._messages: List[str] = []
        self.meta: pd.DataFrame = init_metadata()
        return self

    def clear_targets(self) -> None:
        """Reset target list."""
        self.targets = pd.DataFrame(columns=TARGETS_COLUMNS)

    def clear_results(self) -> None:
        """Reset results."""
        self.results = pd.DataFrame(columns=MINT_RESULTS_COLUMNS)

    def clear_ms_files(self) -> None:
        """Reset MS files."""
        self.ms_files = []

    def run(
        self,
        nthreads: Optional[int] = None,
        rt_margin: float = 0.5,
        mode: str = "standard",
        fn: Optional[str] = None,
        **kwargs,
    ) -> Optional["Mint"]:
        """Run MINT and process MS-files with current target list.

        Args:
            nthreads: Number of cores to use. Options:
                * None - Run with min(n_cpus, n_files) CPUs
                * 1: Run without multiprocessing on one CPU
                * >1: Run with multiprocessing using specified threads
            rt_margin: Margin to add to rt values when rt_min/rt_max not specified.
            mode: Compute mode, one of:
                * 'standard': calculates peak shapes projected to RT dimension
                * 'express': omits calculation of other features, only peak_areas
            fn: Output filename to save results directly to disk instead of memory.
            **kwargs: Additional arguments passed to the processing function.

        Returns:
            Self for method chaining, or None if no files or targets loaded.
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

    def _set_rt_min_max(self, targets: pd.DataFrame, rt_margin: float) -> None:
        """Set retention time min/max values based on rt and margin.

        Args:
            targets: DataFrame containing target information.
            rt_margin: Margin to add/subtract from rt for min/max.
        """
        if "rt" in targets.columns:
            update_rt_min = (targets.rt_min.isna()) & (~targets.rt.isna())
            targets.loc[update_rt_min, "rt_min"] = targets.loc[update_rt_min, "rt"] - rt_margin
            update_rt_max = (targets.rt_max.isna()) & (~targets.rt.isna())
            targets.loc[update_rt_max, "rt_max"] = targets.loc[update_rt_max, "rt"] + rt_margin

    def _determine_nthreads(self, nthreads: Optional[int]) -> int:
        """Determine number of threads to use for parallel processing.

        Args:
            nthreads: Requested number of threads, or None for automatic.

        Returns:
            Number of threads to use.
        """
        if nthreads is None:
            nthreads = min(cpu_count(), self.n_files)
        return nthreads

    def _run_sequential(self, mode: str, fn: Optional[str], targets: pd.DataFrame) -> None:
        """Run processing sequentially (single-threaded).

        Args:
            mode: Processing mode ('standard' or 'express').
            fn: Output filename or None.
            targets: DataFrame of targets to process.
        """
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

    def _report_runtime(self, start: float) -> None:
        """Report runtime statistics after processing.

        Args:
            start: Start time of processing in seconds.
        """
        end = time.time()
        self.runtime = end - start
        self.runtime_per_file = self.runtime / self.n_files
        self.runtime_per_peak = self.runtime / self.n_files / len(self.targets)

        if self.verbose:
            print(f"Total runtime: {self.runtime:.2f}s")
            print(f"Runtime per file: {self.runtime_per_file:.2f}s")
            print(f"Runtime per peak ({len(self.targets)}): {self.runtime_per_peak:.2f}s\n")
            print("Results:", self.results)

    def _run_parallel(
        self,
        nthreads: int = 1,
        mode: str = "standard",
        maxtasksperchild: Optional[int] = None,
        fn: Optional[str] = None,
    ) -> None:
        """Run processing in parallel using multiple threads.

        Args:
            nthreads: Number of threads to use.
            mode: Processing mode ('standard' or 'express').
            maxtasksperchild: Maximum number of tasks per child process.
            fn: Output filename or None.
        """
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

    def _monitor_progress(self, results: Any, q: Any) -> None:
        """Monitor progress of parallel processing.

        Args:
            results: AsyncResult object from parallel processing.
            q: Queue for tracking progress.
        """
        while not results.ready():
            size = q.qsize()
            self.progress = 100 * size / self.n_files
            time.sleep(1)
        self.progress = 100

    @property
    def status(self) -> str:
        """Get current status of Mint instance.

        Returns:
            Status string, one of: 'waiting', 'running', 'done'
        """
        return self._status

    @property
    def ms_files(self) -> List[str]:
        """Get list of MS files to process.

        Returns:
            List of filenames.
        """
        return self._files

    @ms_files.setter
    def ms_files(self, list_of_files: Union[str, List[str]]) -> None:
        """Set MS files to process.

        Args:
            list_of_files: Filename or list of file names of MS-files.
        """
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
    def n_files(self) -> int:
        """Get number of currently stored MS filenames.

        Returns:
            Number of files stored in self.ms_files
        """
        return len(self.ms_files)

    def load_files(self, obj: Union[str, List[str]]) -> "Mint":
        """Load MS files and return self for chaining.

        Args:
            obj: Filename pattern (for glob) or list of file names.

        Returns:
            Self for method chaining.
        """
        if isinstance(obj, str):
            self.ms_files = glob(obj, recursive=True)
        elif isinstance(obj, list):
            self.ms_files = obj
        return self

    def load_targets(self, list_of_files: Union[str, P, List[Union[str, P]]]) -> "Mint":
        """Load targets from file(s) (csv, xlsx).

        Args:
            list_of_files: Filename or list of file names.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input is not a list of files.
            AssertionError: If a file is not found.
        """
        if isinstance(list_of_files, str) or isinstance(list_of_files, P):
            list_of_files = [list_of_files]
        if not isinstance(list_of_files, list):
            raise ValueError("Input should be a list of files.")
        for f in list_of_files:
            assert os.path.isfile(f), f"File not found ({f})"
        self._targets_files = list_of_files
        if self.verbose:
            print("Set targets files to:\n" + "\n".join(str(f) for f in self._targets_files) + "\n")
        self.targets = read_targets(list_of_files)
        return self

    @property
    def targets(self) -> pd.DataFrame:
        """Get target list.

        Returns:
            Target list DataFrame.
        """
        return self._targets

    @targets.setter
    def targets(self, targets: pd.DataFrame) -> None:
        """Set target list.

        Args:
            targets: DataFrame containing target information.

        Raises:
            AssertionError: If targets validation fails.
        """
        targets = standardize_targets(targets)
        assert check_targets(targets), check_targets(targets)
        self._targets = targets.set_index("peak_label")
        if self.verbose:
            print("Set targets to:\n", self.targets.to_string(), "\n")

    def get_target_params(self, peak_label: str) -> Tuple[float, float, float, float]:
        """Get target parameters for a specific peak label.

        Args:
            peak_label: Label of the target peak.

        Returns:
            Tuple of (mz_mean, mz_width, rt_min, rt_max).
        """
        target_data = self.targets.loc[peak_label]
        mz_mean, mz_width, rt_min, rt_max = target_data[["mz_mean", "mz_width", "rt_min", "rt_max"]]
        return mz_mean, mz_width, rt_min, rt_max

    @property
    def peak_labels(self) -> List[str]:
        """Get list of peak labels from targets.

        Returns:
            List of peak label strings.
        """
        return self.targets.index.to_list()

    @property
    def results(self) -> pd.DataFrame:
        """Get results DataFrame.

        Returns:
            DataFrame containing analysis results.
        """
        return self._results

    @results.setter
    def results(self, df: pd.DataFrame) -> None:
        """Set results DataFrame.

        Args:
            df: DataFrame with MINT results.
        """
        self._results = df

    def crosstab(
        self,
        var_name: Optional[str] = None,
        index: Optional[Union[str, List[str]]] = None,
        column: Optional[str] = None,
        aggfunc: str = "mean",
        apply: Optional[Callable] = None,
        scaler: Optional[Union[str, Any]] = None,
        groupby: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """Create condensed representation of the results.

        Creates a cross-table with filenames as index and target labels as columns.
        The values in the cells are determined by var_name.

        Args:
            var_name: Name of the column from results table for cell values.
                Defaults to 'peak_area_top3'.
            index: Column(s) to use as index in the resulting cross-tabulation.
                Defaults to 'ms_file_label'.
            column: Column to use as columns in the resulting cross-tabulation.
                Defaults to 'peak_label'.
            aggfunc: Aggregation function for aggregating values. Defaults to 'mean'.
            apply: Function to apply to the resulting cross-tabulation.
                Options include 'log2p1', 'logp1', or a custom function.
            scaler: Function or name of scaler to scale the data.
                Options include 'standard', 'robust', 'minmax', or a scikit-learn scaler.
            groupby: Column(s) to group data before scaling.

        Returns:
            DataFrame representing the cross-tabulation.

        Raises:
            ValueError: If an unsupported scaler is specified.
        """
        df_meta = pd.merge(self.meta, self.results, left_index=True, right_on="ms_file_label")
        # Remove None if in index
        if isinstance(index, list):
            if None in index:
                index.remove(None)
        if isinstance(groupby, str):
            groupby = [groupby]

        if index is None:
            index = "ms_file_label"
        if column is None:
            column = "peak_label"
        if var_name is None:
            var_name = "peak_area_top3"
        if apply:
            if apply == "log2p1":
                apply = log2p1
            if apply == "logp1":
                apply = np.log1p
            df_meta[var_name] = df_meta[var_name].apply(apply)
        if isinstance(scaler, str):
            scaler_dict = {
                "standard": StandardScaler(),
                "robust": RobustScaler(),
                "minmax": MinMaxScaler(),
            }

            if scaler not in scaler_dict:
                raise ValueError(f"Unsupported scaler: {scaler}")

            scaler = scaler_dict[scaler]

        if scaler:
            if groupby:
                groupby_cols = groupby + [column]
                df_meta[var_name] = df_meta.groupby(groupby_cols)[var_name].transform(
                    lambda x: self._scale_group(x, scaler)
                )
            else:
                df_meta[var_name] = df_meta.groupby(column)[var_name].transform(
                    lambda x: self._scale_group(x, scaler)
                )

        df = pd.pivot_table(
            df_meta,
            index=index,
            columns=column,
            values=var_name,
            aggfunc=aggfunc,
        ).astype(np.float64)
        return df

    @property
    def progress(self) -> float:
        """Get current progress value.

        Returns:
            Current progress value (0-100).
        """
        return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        """Set progress and call progress callback function.

        Args:
            value: Progress value between 0 and 100.

        Raises:
            AssertionError: If value is outside the range 0-100.
        """
        assert value >= 0, value
        assert value <= 100, value
        self._progress = value
        if self.progress_callback is not None:
            self.progress_callback(value)

    def export(self, fn: Optional[str] = None) -> Optional[BytesIO]:
        """Export current results to file.

        Args:
            fn: Filename to export to. If None, returns file buffer.
                Supported formats: .xlsx, .csv, .parquet

        Returns:
            BytesIO buffer if fn is None, otherwise None.
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
        return None

    def load(self, fn: Union[str, BytesIO]) -> "Mint":
        """Load results into Mint instance.

        Args:
            fn: Filename (csv, xlsx, parquet) or file-like object.

        Returns:
            Self for method chaining.
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
        if "ms_file_label" not in results.columns:
            results["ms_file_label"] = [fn_to_label(fn) for fn in results.ms_file]

        self.results = results.rename(columns=DEPRECATED_LABELS)
        self.digest_results()
        return self

    def digest_results(self) -> None:
        """Extract MS files and targets from results and set them in the instance."""
        self.ms_files = get_ms_files_from_results(self.results)
        self.targets = get_targets_from_results(self.results)

    def get_chromatograms(
        self,
        fns: Optional[List[str]] = None,
        peak_labels: Optional[List[str]] = None,
        filters: Optional[List[Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get chromatograms for specified files and peak labels.

        Args:
            fns: List of filenames to extract chromatograms from. Defaults to all MS files.
            peak_labels: List of peak labels to extract. Defaults to all peak labels.
            filters: List of filters to apply to the chromatograms.
            **kwargs: Additional arguments to pass to the Chromatogram constructor.

        Returns:
            DataFrame containing chromatogram data.
        """
        if fns is None:
            fns = self.ms_files
        if peak_labels is None:
            peak_labels = self.peak_labels
        return self._get_chromatograms(
            fns=tuple(fns),
            peak_labels=tuple(peak_labels),
            filters=tuple(filters) if filters is not None else None,
            **kwargs,
        )

    @lru_cache(1)
    def _get_chromatograms(
        self,
        fns: Optional[Tuple[str, ...]] = None,
        peak_labels: Optional[Tuple[str, ...]] = None,
        filters: Optional[Tuple[Any, ...]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Cached implementation of get_chromatograms.

        Args:
            fns: Tuple of filenames to extract chromatograms from.
            peak_labels: Tuple of peak labels to extract.
            filters: Tuple of filters to apply to the chromatograms.
            **kwargs: Additional arguments to pass to the Chromatogram constructor.

        Returns:
            DataFrame containing chromatogram data.
        """
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

    def load_metadata(self, fn: Optional[Union[str, P]] = None) -> "Mint":
        """Load metadata from file.

        Args:
            fn: Filename to load metadata from. Defaults to metadata.parquet in working directory.

        Returns:
            Self for method chaining.
        """
        if fn is None:
            fn = self.wdir / METADATA_DEFAUT_FN
        if str(fn).endswith(".csv"):
            self.meta = pd.read_csv(fn, index_col=0)
        elif str(fn).endswith(".parquet"):
            self.meta = pd.read_parquet(fn)
        if "ms_file_label" in self.meta.columns:
            self.meta = self.meta.set_index("ms_file_label")
        return self

    def save_metadata(self, fn: Optional[Union[str, P]] = None) -> "Mint":
        """Save metadata to file.

        Args:
            fn: Filename to save metadata to. Defaults to metadata.parquet in working directory.

        Returns:
            Self for method chaining.
        """
        if fn is None:
            fn = self.wdir / METADATA_DEFAUT_FN
        if str(fn).endswith(".csv"):
            self.meta.to_csv(fn, na_filter=False)
        elif str(fn).endswith(".parquet"):
            self.meta.to_parquet(fn)
        return self

    def _scale_group(self, group: pd.Series, scaler: Any) -> np.ndarray:
        """Scale a group of values using a scaler.

        Args:
            group: Series of values to scale.
            scaler: Scikit-learn scaler with fit_transform method.

        Returns:
            Scaled values as a numpy array.
        """
        return scaler.fit_transform(group.to_numpy().reshape(-1, 1)).flatten()
