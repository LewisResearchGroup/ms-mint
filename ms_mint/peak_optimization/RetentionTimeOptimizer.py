
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, argrelmin, peak_widths



class RetentionTimeOptimizer():
    def __init__(self, mint):
        self._mint = mint
        self._interim_peaklist = self._mint.peaklist.copy()
        self._ms_files = mint.ms_files
        
    def init_rt_minmax(self, peaklist, margin=None):
        '''
        '''
        peaklist = peaklist.copy()
        cols = peaklist.columns
        for ndx, row in peaklist.iterrows():
            if margin is None:
                margin = 0.2
            if (row['rt_min'] is not None) and (row['rt_max'] is not None):
                peaklist.loc[ndx, 'rt_min'] = max( 0, row['rt_min'] - margin)
                peaklist.loc[ndx, 'rt_max'] = row['rt_max'] + margin
            else: 
                assert 'rt' in cols, 'Missing column for retention time (rt)'
                if margin is None:
                    margin = 0
                peaklist.loc[ndx, 'rt_min'] = max( 0, row['rt'] - margin)
                peaklist.loc[ndx, 'rt_max'] = row['rt'] + margin
            # If rt is none set it to the mean of rt_min and rt_max
            peaklist.loc[peaklist['rt'].isna(), 'rt'] = \
                peaklist.loc[peaklist['rt'].isna(), ['rt_min', 'rt_max']].mean(axis=1)
        self._interim_peaklist = peaklist
        return peaklist

    def fit_transform(self, margin=None, **kwargs):
        self.init_rt_minmax(self._mint.peaklist, margin=margin)
        self._mint.peaklist = self._interim_peaklist
        self._mint.run()
        peaklist, figures = self.fit(**kwargs)
        self._mint.peaklist = peaklist
        self._mint.clear_results()
        return peaklist
        
    def fit(self, show_plots=False, **kwargs):
        return optimize_retention_times(self._mint.results, self._interim_peaklist, **kwargs)

    def get_peaklist(self):
        return self._mint.peaklist


def optimize_retention_times(results, peaklist, create_plots=False, 
        show_plots=True, how='closest', prominence=5e4, verbose=False, 
        **kwargs):

    if verbose:
        print('Optimize RT')
        print('Prominence:', prominence)

    if show_plots: create_plots = True
    assert how in ['closest', 'max']

    interim_peaklist = peaklist.set_index('peak_label').copy()
    figures = {}

    for label, grp in results.groupby('peak_label'):
        rt = grp['rt'].values[0]
        rt_min = grp['rt_min'].values[0]
        rt_max = grp['rt_max'].values[0]
        
        grp = grp[grp.peak_n_datapoints > 20]
        
        df = pd.DataFrame({'rt': grp['peak_shape_rt'].apply(lambda x: x.split(',')).explode('peak_shape_rt').astype('float'),
                           'intensity': grp['peak_shape_int'].apply(lambda x: x.split(',')).explode('peak_shape_int').astype('int')})
        if len(df)<20:
            if verbose: print('Smoothed to small')
            continue

        smoothed = df.groupby('rt').max().rolling(8, center=True).mean().dropna().reset_index()

        t_min = smoothed.rt.min()
        t_max = smoothed.rt.max()
        t_span = t_max - t_min
        
        dt = 0.01
        interpolated = smoothed.set_index('rt')\
                .reindex(np.arange(0, t_max, dt))\
                .interpolate().rolling(4, center=True)\
                .mean().reset_index().dropna() 

        interpolated = interpolated[interpolated.rt >= t_min]
        interpolated = interpolated[interpolated.rt <= t_max]
        
        t_min = interpolated.rt.min()
        t_max = interpolated.rt.max()
        t_span = t_max - t_min
        
        if (t_min is np.NaN) or (t_max is np.NaN):
            if verbose: print('Times are NaN', t_min, t_max)
            continue
        
        x_to_t = lambda x: ((t_span * np.array(x) / len(interpolated)) + t_min).flatten()
        
        assert np.abs(x_to_t(0) - t_min) <= dt, (x_to_t(0), t_min)
        assert np.abs(x_to_t(len(interpolated)) - t_max) <= dt, (x_to_t(len(interpolated)), t_max)
        
        t = interpolated['rt']
        x = interpolated['intensity'].values
        
        if isinstance(prominence, float):
            _prominence = prominence*max(x)
        else:
            _prominence = prominence

        ndx_peaks, _ = find_peaks(x, prominence=(_prominence, None))
        
        if len(ndx_peaks) == 0:
            if verbose: print('No peaks found.')
            continue

        t_peaks = t.iloc[ndx_peaks].values
        x_peaks = x[ndx_peaks]
        
        ndx_minima = argrelmin(x)
        t_minima = x_to_t(ndx_minima)
        x_minima = x[ndx_minima]
        
        results_half  = peak_widths(x, ndx_peaks, rel_height=.5)
        results_bottom  = peak_widths(x, ndx_peaks, rel_height=0.98)
        

        if how == 'closest':
            ndx_selected_peak = (np.argmin(np.abs(t_peaks - rt)))
        elif how == 'max':
            ndx_selected_peak = (np.argmax(x_peaks))


        t_0 = x_to_t(results_bottom[2][ndx_selected_peak])
        t_1 = x_to_t(results_bottom[3][ndx_selected_peak])
        
        t_selected_peak = t_peaks[ndx_selected_peak]
        x_selected_peak = x_peaks[ndx_selected_peak]
        
        for t_m, x_m in zip(t_minima, x_minima):
            if (t_0 < t_m) & (t_m < t_selected_peak):
                t_0 = t_m
            if (t_m > t_selected_peak) & (t_m < t_1):
                t_1 = t_m
                
        t_buffer = 0.01
        t_0 -= t_buffer
        t_1 += t_buffer
        
        interim_peaklist.loc[label, 'rt_min'] = np.round(t_0, 3)
        interim_peaklist.loc[label, 'rt_max'] = np.round(t_1, 3)

        if create_plots:       
            fig = plt.figure()
            plt.plot(smoothed.rt, smoothed.intensity, label='Smoothed')
            plt.plot(interpolated.rt, interpolated.intensity, label='Interpolated')
            plt.plot(t_peaks, x_peaks, "x")

            plt.plot(t_minima, x_minima, "|", ms=10, label='Detected minima')

            plt.hlines(results_half[1], x_to_t(results_half[2]), x_to_t(results_half[3]), color="C2")
            plt.hlines(results_bottom[1], x_to_t(results_bottom[2]), x_to_t(results_bottom[3]), color="C2")
            
            plt.vlines([rt_min, rt_max], 0, max(x), label='RT span')
            plt.vlines([t_min, t_max], 0, max(x), ls='--', label='RT span smoothed')
            
            plt.vlines([rt], 0, max(x), color='C4', lw=2, label='Original RT')

            plt.plot(t_selected_peak, x_selected_peak, "o", color='red', ms=10, label='Selected peak')
            
            plt.hlines(0, t_0, t_1, lw=2, label='Selected RT span')

            plt.title(label)
            plt.xlabel('Retention Time [min]')
            plt.ylabel('Peak Intensity')
            plt.legend()
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.grid()
            figures['label'] = fig
            
            if show_plots: plt.show()

    return interim_peaklist.reset_index(), figures