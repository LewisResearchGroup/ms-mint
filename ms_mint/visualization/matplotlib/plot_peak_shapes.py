import seaborn as sns
import pandas as pd

# from matplotlib import pyplot as plt


def plot_peak_shapes(
    mint_results,
    ms_files=None,
    peak_labels=None,
    height=4,
    aspect=1,
    legend=False,
    n_cols=None,
    col_wrap=4,
    hue="ms_file",
    top=None,
    title=None,
    dpi=None,
    sharex=False,
    sharey=False,
    **kwargs
):

    # fig = plt.figure(dpi=dpi)

    R = mint_results.copy()

    if peak_labels is not None:
        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]
        R = R[R.peak_label.isin(peak_labels)]
    else:
        peak_labels = R.peak_label.drop_duplicates().values

    if ms_files is not None:
        R = R[R.ms_file.isin(ms_files)]

    dfs = []
    for peak_label in peak_labels:
        for ndx, row in R[
            (R.peak_label == peak_label) & (R.peak_n_datapoints > 1)
        ].iterrows():
            peak_rt = [float(i) for i in row.peak_shape_rt.split(",")]
            peak_int = [float(i) for i in row.peak_shape_int.split(",")]
            ms_file = row.ms_file
            df = pd.DataFrame(
                {
                    "Retention Time [min]": peak_rt,
                    "MS-Intensity": peak_int,
                    "ms_file": ms_file,
                    "peak_label": peak_label,
                }
            )
            dfs.append(df)

    df = pd.concat(dfs)

    if n_cols is not None:
        col_wrap = n_cols

    g = sns.relplot(
        data=df,
        x="Retention Time [min]",
        y="MS-Intensity",
        hue=hue,
        col="peak_label",
        kind="line",
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        facet_kws=dict(sharex=sharex, sharey=sharey),
        legend=legend,
        **kwargs
    )

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flatten():
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    if title is not None:
        g.fig.suptitle(title, y=1.01)

    return g
