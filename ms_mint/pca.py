import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import plotly.figure_factory as ff
from plotly import express as px

from sklearn.decomposition import PCA
from .tools import scale_dataframe


class PrincipalComponentsAnalyser:
    """
    Class for applying PCA to Mint instance.
    """

    def __init__(self, mint=None):
        """
        Class for applying PCA to Mint instance.

        :param mint: Mint instance, defaults to None
        :type mint: ms_mint.Mint.Mint, optional
        """
        self.mint = mint
        self.results = None
        self.plot = PCA_Plotter(self)

    def run(self, n_components=3, on=None, var_name="peak_max", fillna="median", apply=None, groupby=None, scaler="standard"):
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
        
        if on is not None:
            warnings.warn("on is depricated use var_name instead", DeprecationWarning)            
            var_name = on            
        
        df = self.mint.crosstab(var_name=var_name, apply=apply, scaler=scaler, groupby=groupby)

        if fillna == "median":
            fillna = df.median()
        elif fillna == "mean":
            fillna = df.mean()
        elif fillna == "zero":
            fillna = 0

        df = df.fillna(fillna)

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


class PCA_Plotter:
    """
    Class for plotting Mint PCA results.
    """

    def __init__(self, pca):
        """
        Class for plotting Mint PCA results.

        :param pca: PrincipalComponentsAnalyser instance
        :type pca: ms_mint.pca.PrincipalComponentsAnalyser
        """
        self.pca = pca

        
    def cumulative_variance(self, interactive=False, **kwargs):
        if interactive:
            return self.cumulative_variance_px(**kwargs)
        else:
            return self.cumulative_variance_sns(**kwargs) 
        
        
    def cumulative_variance_px(self, **kwargs):
        """
        After running mint.pca() this function can be used to plot the cumulative variance of the
        principal components.

        :return: Returns a plotly express figure.
        :rtype: plotly.graph_objs._figure.Figure
        """
        n_components = self.pca.results["n_components"]
        cum_expl_var = self.pca.results["cum_expl_var"]
        df = pd.DataFrame({'Principal Component': np.arange(n_components) + 1, 'Explained variance [%]': cum_expl_var})
        fig = px.bar(df, x='Principal Component', y='Explained variance [%]', 
                     title="Cumulative explained variance",
                     labels={'Principal Component':'Principal Component', 'Explained variance [%]':'Explained variance [%]'},
                     **kwargs)
        fig.update_layout(autosize=True, showlegend=False)
        return fig

    def cumulative_variance_sns(self, **kwargs):
        """
        After running mint.pca() this function can be used to plot the cumulative variance of the
        principal components.

        :return: Returns a matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        # Set default values for aspect and height
        aspect = kwargs.get('aspect', 1)
        height = kwargs.get('height', 5)

        n_components = self.pca.results["n_components"]
        cum_expl_var = self.pca.results["cum_expl_var"]

        # Calculate width based on aspect ratio and number of components
        width = height * aspect

        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(
            np.arange(n_components) + 1,
            cum_expl_var,
            facecolor="grey",
            edgecolor="none",
        )
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained variance [%]")
        ax.set_title("Cumulative explained variance")
        #ax.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        ax.set_xticks(range(1, len(cum_expl_var) + 1))
        return fig

    def _prepare_data(self, n_components=3, hue=None):
        df = self.pca.results["df_projected"].copy()
        cols = df.columns.to_list()[:n_components]
        df = df[cols]

        df = pd.merge(df, self.pca.mint.meta.dropna(axis=1, how='all'), left_index=True, right_index=True)

        if hue and (not isinstance(hue, str)):
            df['Label'] = hue
            df['Label'] = df['Label'].astype(str)

        return df

    def pairplot(
        self, n_components=3, hue=None, fig_kws=None, interactive=False, **kwargs
    ):
        """
        After running mint.pca() this function can be used to plot a scatter matrix of the
        principal components.

        :param n_components: Number of principal components to plot, defaults to 3.
        :type n_components: int, optional
        :param hue: Labels used for hue. If string, the data will be taken from the mint.meta dataframe.
        :type hue: List[str] or str, optional
        :return: Returns a matplotlib figure.
        :rtype: seaborn.axisgrid.PairGrid
        """

        df = self._prepare_data(n_components=n_components, hue=hue)

        if isinstance(hue, list):
            hue = 'label'

        if interactive:
            return self.pairplot_plotly(df, color_col=hue, **kwargs)
        else:
            return self.pairplot_sns(df, fig_kws=fig_kws, hue=hue, **kwargs)

    def pairplot_sns(self, df, fig_kws=None, **kwargs):
        if fig_kws is None:
            fig_kws = {}
        plt.figure(**fig_kws)
        g = sns.pairplot(df, **kwargs)
        return g

    def pairplot_plotly(self, df, color_col=None, **kwargs):
        columns = df.filter(regex=f'PC|^{color_col}$').columns
        fig = ff.create_scatterplotmatrix(df[columns], index=color_col, hovertext=df.index, **kwargs)
        # set the legendgroup equal to the marker color
        for t in fig.data:
            t.legendgroup = t.marker.color
        return fig

    def loadings(self, interactive=False, **kwargs):
        if interactive:
            return self.loadings_plotly(**kwargs)
        else:    
            return self.loadings_sns(**kwargs)

    def loadings_sns(self, **kwargs):
        if 'row' not in kwargs:
            kwargs['row'] = 'PC'
        g = sns.catplot(data=self.pca.results['feature_contributions'], x='peak_label', y='Coefficient', kind='bar', **kwargs)
        plt.tight_layout()
        return g
    
    def loadings_plotly(self, **kwargs):
        if 'facet_row' not in kwargs:
            kwargs['facet_row'] = 'PC'
        fig = px.bar(self.pca.results['feature_contributions'], x='peak_label', y='Coefficient', barmode='group', **kwargs)
        return  fig