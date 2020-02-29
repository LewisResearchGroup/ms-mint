from matplotlib.pyplot import vlines, hlines, tight_layout, title, xlabel, ylabel, legend
from scipy.signal import find_peaks, find_peaks_cwt

def plot_peak_prop(peak_data, show_legend=False):
    shape = peak_data.peak_shape
    if shape is None:
        print('Nothing to plot.')
        return None
    shape.plot(color='k', label='Signal')
    vlines(peak_data.peak_rt_of_max, 0, peak_data.peak_max, lw=1, colors='k', linestyle='--', label='RT of max', color='grey')
    hlines([peak_data.peak_max, peak_data.peak_min], peak_data.rt_min, peak_data.rt_max, color='C0', linewidth=2, label='Min/Max')
    hlines([peak_data.peak_median], peak_data.rt_min, peak_data.rt_max, colors='C1', label='Median')
    hlines([peak_data.peak_mean], peak_data.rt_min, peak_data.rt_max, colors='C2', label='Mean')
    hlines([peak_data.peak_int_first, peak_data.peak_int_last], peak_data.rt_min, peak_data.rt_max, colors='y', linestyle='--', linewidth=2, label='Last/First')

    # Find peaks
    n_rolling = 1
    if len(shape) > 25:
        n_rolling = max(1, int(len(shape) / 5))
    print(len(shape), n_rolling)
    r_shape = shape.rolling(n_rolling, center=True).max() + shape.rolling(n_rolling, center=True).mean()
    a, h = find_peaks(r_shape, height=1e5, rel_height=1e4)
    print(f'Found {len(a)} peaks.')
    shape.iloc[a].plot(lw=0, marker='x', ms=5, mew=1, label='Detected Peaks', color='green')
    
    # Find peaks with wavelets
    a = find_peaks_cwt(r_shape, widths=[5])
    print(f'Found {len(a)} peaks (cwt).')
    shape.iloc[a].plot(lw=0, marker='+', ms=5, mew=1, label='Detected Peaks (cwt)', color='cyan')
    
    if show_legend:
        legend(loc=0, bbox_to_anchor=(1,1), fontsize=8)
        
    title(peak_data.peak_label, size=10)
    xlabel('Retention Time')
    ylabel('Intensity')
    tight_layout()
    