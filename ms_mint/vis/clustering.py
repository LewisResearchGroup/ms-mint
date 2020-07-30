import numpy as np
import matplotlib as mpl    
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy


def hierarchical_clustering(df, vmin=0, vmax=1, figsize=(8,8)):
    '''based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    ''' 
    cm = plt.cm
    cmap = cm.rainbow(np.linspace(0, 0, 1))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    dm = df.values
    D1 = squareform(pdist(dm, metric='euclidean'))
    D2 = squareform(pdist(dm.T, metric='euclidean'))
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
    return df.iloc[Z1['leaves'][::-1], Z2['leaves']], fig