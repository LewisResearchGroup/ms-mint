
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import KernelDensity

from ..Resampler import Resampler

class RetentionTimeOptimizer():
    def __init__(self, rt_min=None, rt_max=None, rt_expected=None, 
                 how='largest', dt=0.01, show_figure=False, precission=3
                 
                 
                 ):
        self.t_peaks = None
        self.x_peaks = None
        self.peak_width_bottom = None
        self.rt_max = rt_max
        self.rt_min = rt_min
        self.rt_span = (0,0)
        self.rt_expected = rt_expected
        self.how = how
        self.dt = dt
        self.resampler = Resampler()
        self.show_figure = show_figure
        self.precission = precission

    def find_peaks_and_widths(self, chrom, prominence=1000):
        x = chrom.values
        ndx_peaks, _ = find_peaks(x, prominence=(prominence, None))
        t_peaks = chrom.iloc[ndx_peaks].index
        x_peaks = chrom.iloc[ndx_peaks].values
        results_bottom  = peak_widths(x, ndx_peaks, rel_height=0.90)
        self.t_peaks = t_peaks
        self.x_peaks = x_peaks
        self.peak_width_bottom = results_bottom
        
    def ndx_largest_peak(self):
        x_peaks = self.x_peaks
        if len(x_peaks) == 0:
            return None
        return np.argmax(x_peaks)
        
    def resample(self, chrom):
        return self.resampler.resample(chrom)
    
    def get_chrom_properties(self, chrom):
        self.t_min = chrom.index.min()
        self.t_max = chrom.index.max()
        self.t_span = self.t_max - self.t_min
        self.len_chrom = len(chrom)
    
    def x_to_t(self, x):
        return ((self.t_span * np.array(x) / self.len_chrom) + self.t_min).flatten()
    
    def find_largest_peak_in_chromatogram(self, chrom):
        chrom = chrom.copy()
        if self.show_figure: chrom.plot(color='grey', alpha=0.5)
        chrom = self.resample(chrom)
        self.get_chrom_properties(chrom)        
        if self.show_figure: chrom.plot(ax=plt.gca(), color='darkblue', alpha=0.5)
        self.find_peaks_and_widths(chrom)
        ndx = self.ndx_largest_peak()
        if ndx is None: return None
        prop = self.get_peak_properties(ndx)
        if self.show_figure: 
            plt.xlim(prop[2]-0.1, prop[3]+0.1)
            plt.hlines(-0.1, prop[2], prop[3], lw=2)
        return prop
    
    def find_largest_peak(self, chromatograms):
        props = []
        for chrom in chromatograms:
            prop = self.find_largest_peak_in_chromatogram(chrom)
            if prop is not None: props.append( prop )
        if len(props) == 0: return (None, None)
        df = pd.DataFrame( props, columns=['rt', 'max_intensity', 'rt_min', 'rt_max'] )

        rt_of_largest_peak_max = df.sort_values('max_intensity', ascending=False).iloc[0]['rt'] 
        # Filter out far away peaks
        df = df[(df.rt - rt_of_largest_peak_max).abs()<0.3]
        rt_min = estimate_expectation_value(df.rt_min)
        rt_max = estimate_expectation_value(df.rt_max)
        if self.show_figure:
            plt.vlines([rt_min, rt_max], 0, np.max(df.max_intensity), label='Selected RT range', color='k', ls='--')
        return np.round(rt_min, self.precission), np.round(rt_max, self.precission)
    
    def get_peak_start_timee(self, ndx):
        return self.x_to_t(self.peak_width_bottom[2][ndx])[0]
 
    def get_peak_end_timee(self, ndx):
        return self.x_to_t(self.peak_width_bottom[3][ndx])[0]
    
    def get_peak_properties(self, ndx):
        return (self.t_peaks[ndx], 
                self.x_peaks[ndx], 
                self.get_peak_start_timee(ndx),
                self.get_peak_end_timee(ndx))


def estimate_expectation_value(values, kernel='gaussian'):
    values = np.array(values)
    assert kernel in ['gaussian', 'tophat', 'epanechnikov']
    x = np.linspace(np.min(values)-1, np.max(values)+1, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(values[:, np.newaxis])
    log_dens = kde.score_samples(x)
    ndx = np.argmax(log_dens)
    return x[ndx][0]
