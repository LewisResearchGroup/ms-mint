import os
import uuid
import datetime
import itertools
import pandas as pd
import numpy as np
import time

from .tools import integrate_peaks, peak_rt_projections,\
    mzxml_to_pandas_df, check_peaklist, STANDARD_PEAKLIST,\
    restructure_rt_projections, STANDARD_PEAKFILE,\
    read_peaklists

import warnings

from multiprocessing import Process, Pool, Manager, cpu_count
from glob import glob
import mint

from datetime import date


def process_in_parallel(args):
    '''Pickleable function for parallel processing.'''
    filename = args['filename']
    peaklist = args['peaklist']
    q = args['q']
    q.put('filename')
    df = mzxml_to_pandas_df(filename)[['retentionTime', 'm/z array', 'intensity array']]
    df['mzxmlFile'] = filename
    result = integrate_peaks(df, peaklist)
    result['mzxmlFile'] = filename
    result['mzxmlPath'] = os.path.dirname(filename)
    result['fileSize[MB]'] = os.path.getsize(filename) / 1024 / 1024
    result['intensity sum'] = df['intensity array'].sum()
    rt_projection = {filename: peak_rt_projections(df, peaklist)}
    return result, rt_projection


class Mint(object):
    def __init__(self):
        self._files = []
        self._peaklist_files = []
        self._peaklist = pd.DataFrame([])
        self._rt_projections = None
        columns = ['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 
                  'rtmin', 'rtmax', 'peakArea', 'mzxmlFile', 'mzxmlPath', 
                  'peakListFile']
        self._results = pd.DataFrame({i: [] for i in columns})
        self._n_files = 0
        self._n_files_processed = 0
        self._callback_progress = None
        self._all_df = None

    def run(self, nthreads=None):
        if nthreads is None:
            nthreads = min(cpu_count(), self._n_files)
        self.run_parallel(nthreads)

    def run_parallel(self, nthreads=1):
        pool = Pool(processes=nthreads)
        m = Manager()
        q = m.Queue()
        args = []
        for i, filename in enumerate(self.files):
            args.append({'filename': filename,
                         'peaklist': self.peaklist,
                         'q':q})
        
        results = pool.map_async(process_in_parallel, args)
        
        pool.close()
        pool.join()
        self._process_results_data_(results.get())

    def _process_results_data_(self, results):
        self.results = pd.concat([i[0] for i in results])
        rt_projections = {}
        [rt_projections.update(i[1]) for i in results]
        self.rt_projections = restructure_rt_projections(rt_projections)  

    @property
    def files(self):
        return self._files
    
    @files.setter
    def files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        self._files = list_of_files
        self._n_files = len(list_of_files)

    @property
    def peaklist_files(self):
        return self._peaklist_files

    @property
    def peaklist(self):
        return self._peaklist

    @peaklist.setter
    def peaklist(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        self._peaklist_files = list_of_files
        self._peaklist = read_peaklists(list_of_files)
        
    @property
    def results(self):
        return self._results
    
    @results.setter
    def results(self, df):
        self._results = df

    @property
    def rt_projections(self):
        return self._rt_projections

    @rt_projections.setter
    def rt_projections(self, data):
        self._rt_projections = data 

    @property
    def crosstab(self):
        cols = ['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax']
        return pd.crosstab(self.results.peakLabel, 
                           self.results.mzxmlFile, 
                           self.results['peakArea'], 
                           aggfunc=sum).astype(np.float64)
    @property
    def callback_progress(self):
        return self._callback_progress
    
    @callback_progress.setter
    def callback_progress(self, func=None):
        self._callback_progress = func
