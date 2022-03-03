import pandas as pd
from matplotlib import pyplot as plt

# from scipy.signal import find_peaks, find_peaks_cwt


def plot_peak_properties(peak_data, show_legend=True, find_peaks=False):
    peak_shape_int = [float(i) for i in peak_data.peak_shape_int.split(",")]
    peak_shape_rt = [float(i) for i in peak_data.peak_shape_rt.split(",")]

    shape = pd.Series(peak_shape_int, index=peak_shape_rt)

    if peak_shape_rt is None:
        print("Nothing to plot.")
        return None

    plt.plot(peak_shape_rt, peak_shape_int, color="k", label="Signal")

    plt.vlines(
        peak_data.peak_rt_of_max,
        0,
        peak_data.peak_max,
        lw=1,
        colors="k",
        linestyle="--",
        label="RT of max",
        color="grey",
    )

    plt.hlines(
        [peak_data.peak_max, peak_data.peak_min],
        peak_data.rt_min,
        peak_data.rt_max,
        color="C0",
        linewidth=2,
        label="Min/Max",
    )

    plt.hlines(
        [peak_data.peak_median],
        peak_data.rt_min,
        peak_data.rt_max,
        colors="C1",
        label="Median",
    )

    plt.hlines(
        [peak_data.peak_mean],
        peak_data.rt_min,
        peak_data.rt_max,
        colors="C2",
        label="Mean",
    )

    plt.hlines(
        [peak_data.peak_int_first, peak_data.peak_int_last],
        peak_data.rt_min,
        peak_data.rt_max,
        colors="y",
        linestyle="--",
        linewidth=2,
        label="Last/first intensity",
    )

    # Find peaks
    if find_peaks:
        n_rolling = 1
        if len(shape) > 25:
            n_rolling = max(1, int(len(shape) / 5))

        r_shape = (
            3 * shape.rolling(n_rolling, center=True).max()
            + shape.rolling(n_rolling, center=True).mean()
        )

        a, h = find_peaks(r_shape, height=1e5, rel_height=1e4)
        print(f"Found {len(a)} peaks.")

        shape.iloc[a].plot(
            lw=0, marker="x", ms=5, mew=1, label="Detected Peaks", color="green"
        )

    if show_legend:
        plt.legend(loc=0, bbox_to_anchor=(1, 1), fontsize=8)

    # Formating the figure
    plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(peak_data.peak_label, size=10)
    plt.xlabel("Retention Time [min]")
    plt.ylabel("Intensity")
    if show_legend:
        plt.legend(loc=0, bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
