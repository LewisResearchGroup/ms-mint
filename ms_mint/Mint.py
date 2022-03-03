# ms_mint/Mint.py

import os
import numpy as np
import pandas as pd
import time
import logging

from warnings import simplefilter
from pathlib import Path as P
from functools import lru_cache

from matplotlib import pyplot as plt
import seaborn as sns

from multiprocessing import Pool, Manager, cpu_count
from scipy.cluster.hierarchy import ClusterWarning
from sklearn.decomposition import PCA

from .standards import MINT_RESULTS_COLUMNS, TARGETS_COLUMNS, DEPRECATED_LABELS
from .processing import process_ms1_files_in_parallel, extract_chromatogram_from_ms1
from .io import export_to_excel, ms_file_to_df
from .targets import read_targets, check_targets, standardize_targets
from .helpers import is_ms_file, get_ms_files_from_results
from .vis.plotly import plotly_heatmap, plotly_peak_shapes
from .vis.mpl import plot_peak_shapes, hierarchical_clustering
from .peak_optimization.RetentionTimeOptimizer import RetentionTimeOptimizer
from .tools import scale_dataframe

from tqdm import tqdm

import ms_mint


class Mint(object):
    def __init__(self, verbose: bool = False, progress_callback=None):
        self._verbose = verbose
        self._version = ms_mint.__version__
        self._progress_callback = progress_callback
        self.reset()
        if self.verbose:
            print("Mint Version:", self.version, "\n")

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = value

    @property
    def version(self):
        return self._version

    def reset(self):
        self._files = []
        self._targets_files = []
        self._targets = pd.DataFrame(columns=TARGETS_COLUMNS)
        self._results = pd.DataFrame({i: [] for i in MINT_RESULTS_COLUMNS})
        self._all_df = None
        self._progress = 0
        self.runtime = None
        self._status = "waiting"
        self._messages = []

    def optimize_rt(self, ms_files=None, peak_labels=None, rt_margin=0.5, **kwargs):
        chromatograms = []
        if ms_files is None:
            ms_files = self.ms_files
        if peak_labels is None:
            peak_labels = self.targets.peak_label.values
        targets = self.targets.copy()
        targets = targets[targets.peak_label.isin(peak_labels)]
        n_peaks = len(targets)
        for i, (ndx, row) in tqdm(enumerate(targets.iterrows()), total=n_peaks):
            progress = int(100 * (i + 1) / n_peaks)
            if self.progress_callback is not None:
                self.progress_callback(progress)
            peak_label = row["peak_label"]
            if peak_label not in peak_labels:
                continue
            chromatograms = []
            mz_mean, mz_width, rt, rt_min, rt_max = row[
                ["mz_mean", "mz_width", "rt", "rt_min", "rt_max"]
            ]
            for fn in ms_files:
                df = self.ms_file_to_df(fn)
                chrom = extract_chromatogram_from_ms1(
                    df, mz_mean=mz_mean, mz_width=mz_width
                )
                chromatograms.append(chrom)
            params = dict(rt=rt, rt_min=rt_min, rt_max=rt_max, rt_margin=rt_margin)
            rtopt = RetentionTimeOptimizer(**params, **kwargs)
            rt_min, rt_max = rtopt.find_largest_peak(chromatograms)
            self.targets.loc[ndx, ["rt_min", "rt_max"]] = rt_min, rt_max

    def ms_file_to_df(self, fn):
        return ms_file_to_df(fn)

    def clear_targets(self):
        self.targets = pd.DataFrame(columns=TARGETS_COLUMNS)

    def clear_results(self):
        self.results = pd.DataFrame(columns=MINT_RESULTS_COLUMNS)

    def clear_ms_files(self):
        self.ms_files = []

    def run(self, nthreads=None, rt_margin=0.5, mode="standard", **kwargs):
        """
        Run MINT with set up targets and ms-files.
        ----
        Args
            - nthreads: int or None, default = None
                * None: Run with min(n_cpus, c_files) CPUs
                * 1: Run without multiprocessing on one CPU
                * >1: Run with multiprocessing enabled using nthreads threads.
            - mode: str, default = 'standard'
                * 'standard': calculates peak shaped projected to RT dimension
                * 'express': omits calculation of other features, only peak_areas
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
            self.run_parallel(nthreads=nthreads, mode=mode, **kwargs)
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

    def run_parallel(self, nthreads=1, mode="standard", maxtasksperchild=None, output_fn=None):
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
                    "output_fn": output_fn
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
    def messages(self):
        return self._messages

    @property
    def status(self):
        return self._status

    @property
    def files(self):
        return self._files

    @property
    def ms_files(self):
        return self._files

    @files.setter
    def files(self, list_of_files):
        print("Mint.files is deprecated, please use Mint.ms_files instead!")
        self.ms_files = list_of_files

    @property
    def n_files(self):
        return len(self.ms_files)

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
    def targets_files(self):
        return self._targets_files

    @targets_files.setter
    def targets_files(self, list_of_files):
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

    @property
    def n_targets_files(self):
        return len(self.targets_files)

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, targets):
        targets = standardize_targets(targets)
        check_targets(targets)
        self._targets = targets
        if self.verbose:
            print("Set targets to:\n", self.targets.to_string(), "\n")

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, df):
        self._results = df

    @property
    def rt_projections(self):
        return DeprecationWarning(
            "rt_projections is deprecated. Peak shapes are now "
            "directly stored in the results table (mint.results)."
        )

    def crosstab(self, col_name="peak_area"):
        return pd.crosstab(
            self.results.ms_file,
            self.results.peak_label,
            self.results[col_name],
            aggfunc=sum,
        ).astype(np.float64)

    @property
    def progress_callback(self):
        return self._progress_callback

    @progress_callback.setter
    def progress_callback(self, func=None):
        self._progress_callback = func

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        assert value >= 0, value
        assert value <= 100, value
        self._progress = value
        if self.progress_callback is not None:
            self.progress_callback(value)

    def export(self, filename=None):
        fn = filename
        if fn is None:
            buffer = export_to_excel(self, fn=fn)
            return buffer
        elif fn.endswith(".xlsx"):
            export_to_excel(self, fn=fn)
        elif fn.endswith(".csv"):
            self.results.to_csv(fn, index=False)

    def load(self, fn):
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

    def plot_clustering(
        self,
        data=None,
        title=None,
        figsize=(8, 8),
        targets_var="peak_max",
        vmin=-3,
        vmax=3,
        xmaxticks=None,
        ymaxticks=None,
        transform_func="log2p1",
        scaler_ms_file=None,
        scaler_peak_label="standard",
        metric="euclidean",
        transform_filenames_func="basename",
        transpose=False,
        **kwargs,
    ):
        """
        Performs a cluster analysis and plots a heatmap. If no data is provided,
        data is taken form self.crosstab(targets_var).
        The clustered non-transformed non-scaled data is stored in `self.clustered`.

        -----
        Args:

        transform_func: default 'log2p1', values: [None, 'log1p', 'log2p1', 'log10p1']
            - None: no transformation
            - log1p: tranform data with lambda x: np.log1p(x)
            - log2p1: transform data with lambda x: log2(x+1)
            - log10p1: transform data with lambda x: log10(x+1)

        scaler_ms_file: default None, values: [None, 'standard', 'robust']
            - scaler used to scale along ms_file axis
            - if None no scaling is applied
            - if 'standard' use scikit learn StandardScaler()
            - if 'robust' use scikit learn RobustScaler()

        scaler_peak_label: default 'standard'
            - like scaler_ms_file, but scaling along peak_label axis

        metric: default 'euclidean', can be string or a list of two values:
            if two values are provided e.g. ('cosine', 'euclidean') the first
            will be used to cluster the x-axis and the second for the y-axis.

            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’,
            ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
            ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
            More information:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        transpose: bool, default False
            - True: transpose the figure
        """

        if len(self.results) == 0:
            return None

        simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.crosstab(targets_var).copy()

        tmp_data = data.copy()

        if transform_func == "log1p":
            transform_func = np.log1p
        if transform_func == "log2p1":
            transform_func = lambda x: np.log2(x + 1)
        if transform_func == "log10p1":
            transform_func = lambda x: np.log10(x + 1)

        if transform_func is not None:
            tmp_data = tmp_data.apply(transform_func)

        if transform_filenames_func == "basename":
            transform_filenames_func = lambda x: P(x).with_suffix("").name
        elif transform_filenames_func is not None:
            tmp_data.columns = [transform_filenames_func(i) for i in tmp_data.columns]

        # Scale along ms-files
        if scaler_ms_file is not None:
            tmp_data = scale_dataframe(tmp_data.T, scaler_ms_file).T

        # Scale along peak_labels
        if scaler_peak_label is not None:
            tmp_data = scale_dataframe(tmp_data, scaler_peak_label)

        if transpose:
            tmp_data = tmp_data.T

        clustered, fig, ndx_x, ndx_y = hierarchical_clustering(
            tmp_data,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            xmaxticks=xmaxticks,
            ymaxticks=ymaxticks,
            metric=metric,
            **kwargs,
        )

        if not transpose:
            self.clustered = data.iloc[ndx_x, ndx_y]
        else:
            self.clustered = data.iloc[ndx_y, ndx_x]
        return fig

    def plot_peak_shapes(self, **kwargs):
        if len(self.results) > 0:
            return plot_peak_shapes(self.results, **kwargs)

    def plotly_heatmap(
        self,
        col_name="peak_max",
        normed_by_cols=False,
        transposed=False,
        clustered=False,
        add_dendrogram=False,
        name="",
        correlation=False,
    ):
        """Creates an interactive heatmap
        that can be used to explore the data interactively.
        `mint.crosstab()` is called and then subjected to
        the `mint.vis.plotly.plotly_tools.plot_heatmap()`.

        Arguments
        ---------
        col_name: str, default='peak_max'
            Name of the column in `mint.results` to be analysed.
        normed_by_cols: bool, default=True
            Whether or not to normalize the columns in the crosstab.
        clustered: bool, default=False
            Whether or not to cluster the rows.
        add_dendrogram: bool, default=False
            Whether or not to replace row labels with a dendrogram.
        transposed: bool, default=False
            If True transpose matrix before plotting.
        correlation: bool, default=False
            If True convert data to correlation matrix before plotting.

        """
        if len(self.results) > 0:
            return plotly_heatmap(
                self.crosstab(col_name),
                normed_by_cols=normed_by_cols,
                transposed=transposed,
                clustered=clustered,
                add_dendrogram=add_dendrogram,
                name=col_name,
                correlation=correlation,
            )

    def pca(
        self, var_name="peak_max", n_components=3, fillna="median", scaler="standard"
    ):

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

    def pca_plot_cumulative_variance(self):
        n_vars = self.decomposition_results["n_components"]
        fig = plt.figure(figsize=(7, 3), dpi=300)
        cum_expl_var = self.decomposition_results["cum_expl_var"]
        plt.bar(np.arange(n_vars) + 1, cum_expl_var, facecolor="grey", edgecolor="none")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained variance [%]")
        plt.title("Cumulative explained variance")
        plt.grid()
        plt.xticks(range(1, len(cum_expl_var) + 1))
        return fig

    def plot_pair_plot(
        self, n_vars=3, color_groups=None, group_name=None, marker=None, **kwargs
    ):
        df = self.decomposition_results["df_projected"]
        cols = df.columns.to_list()[:n_vars]
        df = df[cols]

        if color_groups is not None:
            if group_name is None:
                group_name = "Group"
            df[group_name] = color_groups
            df[group_name] = df[group_name].astype(str)

        fig = plt.figure(dpi=300)

        if marker is None and len(df) > 20:
            marker = "+"

        g = sns.pairplot(
            df, plot_kws={"s": 50, "marker": marker}, hue=group_name, **kwargs
        )

        if color_groups is not None:
            leg = g._legend
            # leg.set_bbox_to_anchor([1.05, 0.5])

        return fig
