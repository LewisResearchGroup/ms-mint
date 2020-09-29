import numpy as np
import matplotlib as mpl    
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy


def hierarchical_clustering(df, vmin=None, vmax=None, figsize=(8,8), 
                            xmaxticks=None, ymaxticks=None, metric='euclidean'):
    '''based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    '''

    df = df.copy()

    cm = plt.cm
    cmap = cm.rainbow(np.linspace(0, 0, 1))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    ydim, xdim = df.shape

    if xmaxticks is None:
        xmaxticks = min(20, xdim)
    if ymaxticks is None:
        ymaxticks = min(20, ydim)

    dm = df.fillna(0).values
    D1 = squareform(pdist(dm, metric=metric))
    D2 = squareform(pdist(dm.T, metric=metric))
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(False)
    # add first dendrogram

    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6], frameon=False)
    Y = linkage(D1, method='complete')
    Z1 = dendrogram(Y, orientation='left', color_threshold=0, above_threshold_color='k')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add second dendrogram
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2], frameon=False)
    Y = linkage(D2, method='complete')
    Z2 = dendrogram(Y, color_threshold=0, above_threshold_color='k')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = dm[idx1, :]
    D = D[:, idx2]

    fig = axmatrix.matshow(D[::-1], aspect='auto', cmap='hot',
                           vmin=vmin, vmax=vmax)

    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()

    clustered = df.iloc[Z1['leaves'][::-1], Z2['leaves']]

    ndx_y = np.linspace(0,len(clustered.index)-1, ymaxticks)
    ndx_x = np.linspace(0,len(clustered.columns)-1, xmaxticks)
    ndx_y = [int(i) for i in ndx_y]
    ndx_x = [int(i) for i in ndx_x]

    _ = plt.yticks(ndx_y, clustered.iloc[ndx_y].index)
    _ = plt.xticks(ndx_x, clustered.columns[ndx_x], rotation=90)

    return clustered, fig