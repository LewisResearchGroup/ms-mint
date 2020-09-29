import pandas as pd
import pyopenms as oms

from os.path import basename
from numpy import mean, max, min, abs
from tqdm import tqdm



class OpenMSPeakDetection():
    def __init__(self, kind_ff='centroided', kind_peaklist='basic'):
        self.features = None
        self.kind_ff = kind_ff
        self.kind_peaklist = kind_peaklist

    def fit(self, filenames):
        print('Peak detection')
        if self.kind_ff == 'centroided':
            self.centroided(filenames)

    def transform(self, kind_peaklist=None):
        if self.features is None:
            return None
        if kind_peaklist is None:
            kind_peaklist =self.kind_peaklist
        if kind_peaklist == 'unique_masses':
            return self.peaklist_by_unique_masses()
        elif kind_peaklist == 'basic':
            return self.initial_peaklist()

    def fit_transform(self, filenames):
        self.fit(filenames)
        return self.transform()

    def centroided(self, filenames):
        options = oms.PeakFileOptions()
        options.setMSLevels([1])
        fh = oms.MzXMLFile()
        fh.setOptions(options)

        # Load data
        input_map = oms.MSExperiment()
        for fn in filenames:
            fh.load(fn, input_map)
        input_map.updateRanges()
        
        ff = oms.FeatureFinder()
        ff.setLogType(oms.LogType.CMD)

        # Run the feature finder
        name = "centroided"
        features = oms.FeatureMap()
        seeds = oms.FeatureMap()
        params = oms.FeatureFinder().getParameters(name)
        ff.run(name, input_map, features, params, seeds)
        self.features = features

    def peaklist_by_unique_masses(self):
        features = self.features
        mz_values = pd.DataFrame([ f.getMZ() for f in features], columns=['MZ'])
        unique_masses = mz_values.MZ.round(3).value_counts().index
        peaklist = pd.DataFrame(unique_masses.sort_values(), columns=['mz_mean'])
        peaklist['mz_width'] = 10
        peaklist['rt_min'] = 0
        peaklist['rt_max'] = 12
        peaklist['peak_label'] = peaklist['mz_mean'].astype(str)
        peaklist['peaklist'] = 'FeatureFinder'
        peaklist['intensity_threshold'] = 0
        return peaklist

    def initial_peaklist(self):
        features = self.features        
        peaklist = pd.DataFrame([(f.getRT()/60, f.getMZ()) for f in features] , 
                                columns=['rt_mean', 'mz_mean']).round(3)
        peaklist['mz_width'] = 10
        peaklist['rt_min'] = peaklist.rt_mean - 0.3
        peaklist['rt_max'] = peaklist.rt_mean + 0.3
        peaklist['peak_label'] = peaklist['mz_mean'].astype(str) + '-' + peaklist['rt_mean'].astype(str)
        peaklist['peaklist'] = 'FeatureFinder'
        peaklist['intensity_threshold'] = 0
        return peaklist
    
    def get_peaklist(self):
        return self.initial_peaklist()


class OpenMSFFMetabo():
    def __init__(self):
        self._feature_map = None
        self._progress = 0

    def fit(self, filenames):
        feature_map = oms.FeatureMap()
        for fn in tqdm(filenames):
            feature_map += oms_ffmetabo_single_file(fn)
        self._feature_map = feature_map
    
    def transform(self, min_quality=1e-3, condensed=True, 
                  max_delta_mz_ppm=10, max_delta_rt=0.1):
        features = []
        for feat in tqdm(self._feature_map, total=self._feature_map.size()):    
            
            quality = feat.getOverallQuality()
            if ( min_quality is not None ) and ( quality < min_quality ):
                continue

            rt_min = max([0, (feat.getRT() - feat.getWidth()) / 60])
            rt_max = (feat.getRT() + feat.getWidth()) / 60
            
            data = {'peak_label': f'{feat.getMZ():06.3f}-{feat.getRT():06.3f}', 
                    'mz_mean': feat.getMZ(), 'mz_width': 10, 
                    'rt_min': rt_min, 'rt_max': rt_max, 'rt': feat.getRT() / 60, 
                    'intensity_threshold': 0, 
                    'oms_quality': quality, 
                    'peaklist': 'OpenMSFFMetabo'}
            
            features.append(data)

        peaklist = pd.DataFrame(features)
        peaklist = peaklist.reset_index(drop=True)

        if condensed:
            peaklist = condense_peaklist(peaklist, 
                        max_delta_mz_ppm=max_delta_mz_ppm, 
                        max_delta_rt=max_delta_rt)  
        return peaklist


    def fit_transform(self, filenames, **kwargs):
        self.fit(filenames)
        return self.transform(**kwargs)


def oms_ffmetabo_single_file(filename, max_peaks=5000, verbose=False):

    feature_map = oms.FeatureMap()

    mass_traces = []
    mass_traces_split = []
    mass_traces_filtered = []
    exp = oms.MSExperiment()

    options = oms.PeakFileOptions()
    options.setMSLevels([1])

    if filename.lower().endswith('.mzxml'):
        fh = oms.MzXMLFile()
    elif filename.lower().endswith('.mzml'):
        fh = oms.MzMLFile()

    fh.setOptions(options)

    # Peak map
    peak_map = oms.PeakMap()

    fh.load(filename, exp)

    for chrom in exp.getChromatograms():
        peak_map.addChrom(chrom)

    for spec in exp.getSpectra():
        peak_map.addSpectrum(spec)

    mass_trace_detect = oms.MassTraceDetection()
    mass_trace_detect.run(peak_map, mass_traces, max_peaks)

    elution_peak_detection = oms.ElutionPeakDetection()
    elution_peak_detection.detectPeaks(mass_traces, mass_traces_split)

    feature_finding_metabo = oms.FeatureFindingMetabo()
    feature_finding_metabo.run(
                mass_traces_split,
                feature_map,
                mass_traces_filtered)

    if verbose:
        print('# Filename:', basename(filename))
        print('# Spectra:', len( exp.getSpectra() ))
        print('# Chromatograms:', len( exp.getChromatograms() ) )
        print('# Mass traces:', len(mass_traces) )
        print('# Mass traces split:', len(mass_traces_split) )
        print('# Mass traces filtered:', len(mass_traces_filtered) )
        print('# Features:', feature_map.size() )

    feature_map.sortByOverallQuality()
    return feature_map


def condense_peaklist(peaklist, max_delta_mz_ppm=10, max_delta_rt=0.1):
    print('Condensing peaklist')
    cols = ['mz_mean', 'rt_min', 'rt_max', 'rt']
    peaklist = peaklist.sort_values(cols)[cols]
    new_peaklist = pd.DataFrame(columns=cols)
    
    for ndx_a, peak_a in tqdm(peaklist.iterrows(), total=len(peaklist)):
        mz_a, rt_min_a, rt_max_a, rt_a = peak_a
        merged = False
        for ndx_b, peak_b in new_peaklist.iterrows():
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
            
    return fix_peaklist(new_peaklist.reset_index(drop=True))


def merge_peaks(peak_a, peak_b):
    mz_a, rt_min_a, rt_max_a, rt_a = peak_a
    mz_b, rt_min_b, rt_max_b, rt_b = peak_b
    return (mean([mz_a, mz_b]), 
            min([rt_min_a, rt_min_b]), 
            max([rt_max_a, rt_max_b]), 
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
    to_formated_str = lambda x: f'{x:.3f}'
    if 'peak_label' not in peaklist.columns:     
        peaklist['peak_label'] = ( peaklist.index.astype(str) + '-' +
                                   peaklist.mz_mean.apply(to_formated_str) + '-' +
                                   peaklist['rt_min'].apply(to_formated_str) )
    if 'mz_width' not in peaklist.columns:     
        peaklist['mz_width'] = 10
    if 'peaklist' not in peaklist.columns:
        peaklist['peaklist'] = 'experimental'
    if 'intensity_threshold' not in peaklist.columns:
        peaklist['intensity_threshold'] = 0
    if 'rt' not in peaklist.columns:
        peaklist['rt'] = peaklist[['rt_min', 'rt_max']].mean(axis=1)
    if 'rt_span' not in peaklist.columns:
        peaklist['rt_span'] = peaklist.rt_max - peaklist.rt_min
    return peaklist
