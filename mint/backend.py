import os
import uuid
import datetime
import itertools
import pandas as pd
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

class Mint(object):
    def __init__(self):
        self._mzxml_files = []
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

    def process_files(self, nthreads=None):
        if nthreads is None:
            nthreads = min(cpu_count(), self._n_files)
        self._run_parallel_(nthreads)

    def _run_parallel_(self, nthreads=1):
        pool = Pool(processes=nthreads)
        m = Manager()
        q = m.Queue()
        args = []
        for i, filename in enumerate(self.mzxml_files):
            args.append({'filename': filename,
                         'peaklist': self.peaklist,
                         'q':q})
        results = pool.map_async(process_in_parallel, args)

        # monitor progress
        while True:
            if results.ready():
                break
            else:
                size = q.qsize()
                self._n_files_processed = size
                if self._callback_progress is not None:
                    self._callback_progress(self._n_files, size)
                time.sleep(2)

        pool.close()
        pool.join()
        self._callback_progress(self._n_files, size)
        self._process_results_data_(results.get())

    def _process_results_data_(self, results):
        self.results = pd.concat([i[0] for i in results])[[
            'peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax',
            'peakArea', 'mzxmlFile', 'mzxmlPath', 'peakListFile']]
        rt_projections = {}
        [rt_projections.update(i[1]) for i in results]
        self.rt_projections = restructure_rt_projections(rt_projections)  
        self.all_df = pd.concat([i[2] for i in results])

    @property
    def mzxml_files(self):
        return self._mzxml_files
    
    @mzxml_files.setter
    def mzxml_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        self._mzxml_files = list_of_files
        self._n_files = len(list_of_files)

    @property
    def peaklist_files(self):
        return self._peaklist_files

    @peaklist_files.setter
    def peaklist_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        self._peaklist_files = list_of_files
        self.peaklist = read_peaklists(list_of_files)

    @property
    def peaklist(self):
        return self._peaklist

    @peaklist.setter
    def peaklist(self, df_of_peaks):
        self._peaklist = df_of_peaks     
   
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

    def crosstab(self):
        cols = ['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax']
        return pd.crosstab(self.results.peakLabel, 
                                    self.results.mzxmlFile, 
                                    self.results['peakArea'], 
                                    aggfunc=sum)
    @property
    def callback_progress(self):
        return self._callback_progress
    
    @callback_progress.setter
    def callback_progress(self, func):
        self._callback_progress = func


def process_in_parallel(args):
    '''Pickleable function for peak integration.'''
    filename = args['filename']
    peaklist = args['peaklist']
    q = args['q']
    q.put('filename')
    df = mzxml_to_pandas_df(filename)
    df['mzxmlFile'] = filename
    result = integrate_peaks(df, peaklist)
    result['mzxmlFile'] = filename
    result['mzxmlPath'] = os.path.dirname(filename)
    rt_projection = {filename: peak_rt_projections(df, peaklist)}
    return result, rt_projection, df

