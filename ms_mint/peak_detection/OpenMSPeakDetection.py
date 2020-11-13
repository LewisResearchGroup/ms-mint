import pandas as pd
import pyopenms as oms

from numpy import mean, max, min, abs

class OpenMSFFMetabo():
    def __init__(self, progress_callback=None):
        self._feature_map = None
        self._progress = 0
        self._progress_callback = progress_callback

    def fit(self, filenames, max_peaks_per_file=1000):
        try:
            feature_map = oms.FeatureMap()
        except:
            pass
        n_files = len(filenames)
        for i, fn in enumerate(filenames):
            self.progress = 100*(i+1)/n_files
            feature_map += oms_ffmetabo_single_file(
                fn, max_peaks_per_file=max_peaks_per_file
            )
        self._feature_map = feature_map
    
    @property
    def progress(self):
        return self._progress
    
    @progress.setter    
    def progress(self, x):
        self._progress = x
        if self._progress_callback is not None:
            self._progress_callback(x)

    def transform(self, min_quality=1e-3, condensed=True, 
                  max_delta_mz_ppm=10, max_delta_rt=0.1):

        features = []
        n_total = self._feature_map.size()
        for i, feat in enumerate(self._feature_map):    

            self.progress = 100*(i+1)/n_total

            quality = feat.getOverallQuality()

            if ( min_quality is not None ) and ( quality < min_quality ):
                continue

            mz = feat.getMZ()
            rt = feat.getRT() / 60
            rt_width = feat.getWidth() / 60
            rt_min = max([0, (rt - rt_width)])
            rt_max = (rt + rt_width)
            
            data = {'peak_label': f'mz:{mz:07.4f}-rt:{rt:03.1f}',
                    'mz_mean': mz, 'mz_width': 10,
                    'rt_min': rt_min, 'rt_max': rt_max, 'rt': rt, 
                    'intensity_threshold': 0,
                    'oms_quality': quality,
                    'peaklist': 'OpenMSFFMetabo'}
            
            features.append(data)

        peaklist = pd.DataFrame(features)
        peaklist = peaklist.reset_index(drop=True)

        if condensed:
            peaklist = condense_peaklist(peaklist, 
                        max_delta_mz_ppm=max_delta_mz_ppm, 
                        max_delta_rt=max_delta_rt,
                        progress_callback=self._progress_callback)  
        return peaklist


    def fit_transform(self, filenames, max_peaks_per_file=1000, 
                      min_quality=1e-3, condensed=True, 
                      max_delta_mz_ppm=10, max_delta_rt=0.1, 
                      progress_callback=None):

        self.fit(filenames, max_peaks_per_file=max_peaks_per_file)
        return self.transform(min_quality=1e-3, condensed=True, 
                              max_delta_mz_ppm=10, max_delta_rt=0.1)


def oms_ffmetabo_single_file(filename, max_peaks_per_file=5000):

    feature_map = oms.FeatureMap()
    mass_traces = []
    mass_traces_split = []
    mass_traces_filtered = []
    exp = oms.MSExperiment()
    peak_map = oms.PeakMap()
    options = oms.PeakFileOptions()
    
    options.setMSLevels([1])

    if filename.lower().endswith('.mzxml'):
        fh = oms.MzXMLFile()

    elif filename.lower().endswith('.mzml'):
        fh = oms.MzMLFile()
    else:
        assert False, filename

    fh.setOptions(options)

    # Peak map
    fh.load(filename, exp)

    #for chrom in exp.getChromatograms():
    #    peak_map.addChrom(chrom)

    for spec in exp.getSpectra():
        peak_map.addSpectrum(spec)

    mass_trace_detect = oms.MassTraceDetection()
    mass_trace_detect.run(peak_map, mass_traces, max_peaks_per_file)

    elution_peak_detection = oms.ElutionPeakDetection()
    elution_peak_detection.detectPeaks(mass_traces, mass_traces_split)

    feature_finding_metabo = oms.FeatureFindingMetabo()
    feature_finding_metabo.run(
                mass_traces_split,
                feature_map,
                mass_traces_filtered)

    feature_map.sortByOverallQuality()
    return feature_map


def condense_peaklist(peaklist, max_delta_mz_ppm=10, max_delta_rt=0.1, progress_callback=None):
    cols = ['mz_mean', 'rt_min', 'rt_max', 'rt']
    peaklist = peaklist.sort_values(cols)[cols]

    n_before = len(peaklist)
    n_after = None

    while n_before != n_after:
        n_before = len(peaklist)
        new_peaklist = pd.DataFrame(columns=cols)

        for i, (ndx_a, peak_a) in enumerate( peaklist.iterrows() ):
            if progress_callback is not None:
                progress_callback(100*(i+1)/n_before)
            mz_a, rt_min_a, rt_max_a, rt_a = peak_a
            merged = False
            for ndx_b, peak_b in new_peaklist[ abs(new_peaklist.mz_mean - mz_a) < 0.01 ].iterrows():
                if ( peaks_are_close(peak_a, peak_b, 
                                    max_delta_mz_ppm=max_delta_mz_ppm, 
                                    max_delta_rt=max_delta_rt) ):
                    
                    merged_peak = merge_peaks(peak_a, peak_b)
                    new_peaklist.loc[ndx_b, cols] = merged_peak
                    merged = True
                    break        
            if not merged:
                new_peaklist = pd.concat([new_peaklist, as_df(peak_a)])\
                                .sort_values(cols).reset_index(drop=True)
        peaklist = new_peaklist
        n_after = len(new_peaklist)
        
    return fix_peaklist(new_peaklist.reset_index(drop=True))


def merge_peaks(peak_a, peak_b):
    mz_a, rt_min_a, rt_max_a, rt_a = peak_a
    mz_b, rt_min_b, rt_max_b, rt_b = peak_b
    return (mean([mz_a, mz_b]), 
            min([rt_min_a, rt_min_b, rt_max_a, rt_max_b]), 
            max([rt_min_a, rt_min_b, rt_max_a, rt_max_b]), 
            mean([rt_a, rt_b]))


def peaks_are_close(peak_a, peak_b, max_delta_mz_ppm=10, max_delta_rt=0.1):
    mz_a, rt_min_a, rt_max_a, rt_a = peak_a
    mz_b, rt_min_b, rt_max_b, rt_b = peak_b
    mz_limit = max([mz_a*max_delta_mz_ppm*1e-6, 
                    mz_b*max_delta_mz_ppm*1e-6])    
    if rt_a is None or rt_b is None:
        rt_mean_a = mean([rt_min_a, rt_max_a])
        rt_mean_b = mean([rt_min_b, rt_max_b])
        rt_delta = abs(rt_mean_a - rt_mean_b)
    else:
        rt_delta = abs(rt_a-rt_b)   
    if abs(mz_a - mz_b) < mz_limit:
        if rt_delta < max_delta_rt:
            return True
    return False


def as_df(peak):
    mz, rt_min, rt_max, rt = peak
    data = dict(mz_mean=mz, rt_min=rt_min, rt_max=rt_max, rt=rt)
    df = pd.DataFrame( data, index=[0] )
    return df


def fix_peaklist(peaklist):
    if 'peak_label' not in peaklist.columns:     
        peaklist['peak_label'] = ( peaklist.index.astype(str) + '_' +
                                   'mz:' + peaklist.mz_mean.apply(lambda x: f'{x:7.3f}') + '_' +
                                   'rt:' + peaklist['rt'].apply(lambda x: f'{x:3.1f}') )
    if 'mz_width' not in peaklist.columns:     
        peaklist['mz_width'] = 10
    if 'peaklist_name' not in peaklist.columns:
        peaklist['peaklist_name'] = 'OMS'
    if 'intensity_threshold' not in peaklist.columns:
        peaklist['intensity_threshold'] = 0
    if 'rt' not in peaklist.columns:
        peaklist['rt'] = peaklist[['rt_min', 'rt_max']].mean(axis=1)
    if 'rt_span' not in peaklist.columns:
        peaklist['rt_span'] = peaklist.rt_max - peaklist.rt_min
    return peaklist
