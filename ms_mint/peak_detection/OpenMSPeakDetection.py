import pyopenms as oms
import pandas as pd

class OpenMSPeakDetection():
    def __init__(self, kind_ff='centroided', kind_peaklist='basic'):
        self.features = None
        self.kind_ff = kind_ff
        self.kind_peaklist = kind_peaklist

    def fit(self, files):
        if self.kind_ff == 'centroided':
            self.centroided(files)

    def transform(self, kind_peaklist=None):
        if self.features is None:
            return None
        if kind_peaklist is None:
            kind_peaklist =self.kind_peaklist
        if kind_peaklist == 'unique_masses':
            return self.peaklist_by_unique_masses()
        elif kind_peaklist == 'basic':
            return self.initial_peaklist()

    def fit_transform(self, files):
        self.fit(files)
        return self.transform()

    def centroided(self, files):
        options = oms.PeakFileOptions()
        options.setMSLevels([1])
        fh = oms.MzXMLFile()
        fh.setOptions(options)

        # Load data
        input_map = oms.MSExperiment()
        for file in files:
            fh.load(file, input_map)
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
        peaklist['rt_min'] = peaklist.rt_mean - 0.1
        peaklist['rt_max'] = peaklist.rt_mean + 0.1
        peaklist['peak_label'] = peaklist['mz_mean'].astype(str) + '-' + peaklist['rt_mean'].astype(str)
        peaklist['peaklist'] = 'FeatureFinder'
        peaklist['intensity_threshold'] = 0
        return peaklist