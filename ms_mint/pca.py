
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from .tools import scale_dataframe


class PrincipalComponentsAnalyser():

    def __init__(self, mint=None):
        self.mint = mint
        self.results = None
        self.plot = PCA_Plotter(self)
    

    def run(self, on="peak_max", n_components=3, fillna="median", scaler="standard"):
        """
        Run Principal Component Analysis on current results. Results are stored in
        self.decomposition_results.

        :param on: Column name to use for pca, defaults to "peak_max"
        :type on: str, optional
        :param n_components: Number of PCA components to return, defaults to 3
        :type n_components: int, optional
        :param fillna: Method to fill missing values, defaults to "median"
        :type fillna: str, optional
        :param scaler: Method to scale the columns, defaults to "standard"
        :type scaler: str, optional
        """

        df = self.mint.crosstab(on).fillna(fillna)

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

        self.results = {
            "df_projected": df_projected,
            "cum_expl_var": cum_expl_var,
            "n_components": n_components,
            "type": "PCA",
            "feature_contributions": dfc,
            "class": pca,
        }


class PCA_Plotter():

    def __init__(self, pca):
        self.pca = pca

    def cumulative_variance(self):
        """After running mint.pca() this function can be used to plot the cumulative variance of the
        principal components.

        :return: Returns a matplotlib figure.
        :rtype: plt.figure
        """
        n_vars = self.pca.results["n_components"]
        fig = plt.figure(figsize=(7, 3), dpi=300)
        cum_expl_var = self.pca.results["cum_expl_var"]
        plt.bar(np.arange(n_vars) + 1, cum_expl_var, facecolor="grey", edgecolor="none")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained variance [%]")
        plt.title("Cumulative explained variance")
        plt.grid()
        plt.xticks(range(1, len(cum_expl_var) + 1))
        return fig

    def scatter_matrix(
        self, n_vars=3, color_groups=None, group_name=None, marker=None, **kwargs
    ):
        """After running mint.pca() this function can be used to plot a scatter matrix of the
        principal components.

        """
        df = self.pca.results["df_projected"]
        cols = df.columns.to_list()[:n_vars]
        df = df[cols]

        if color_groups is not None:
            if group_name is None:
                group_name = "Group"
            df[group_name] = color_groups
            df[group_name] = df[group_name].astype(str)

        plt.figure(dpi=300)

        if marker is None and len(df) > 20:
            marker = "+"

        g = sns.pairplot(
            df, plot_kws={"s": 50, "marker": marker}, hue=group_name, **kwargs
        )

        return g        