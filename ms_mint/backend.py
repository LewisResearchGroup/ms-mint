import os
import pandas as pd
import numpy as np
import time

from .tools import read_peaklists, process, restructure_rt_projections

from multiprocessing import Pool, Manager, cpu_count
from datetime import date

import ms_mint

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
        self._all_df = None
        self.version = ms_mint.__version__
        self.progress = 0
        self.runtime = None

    def run(self, nthreads=None):
        '''Process files in self.files with self.peaklist. If nthreads != 1 use
        multiprocessing.'''
        if (self.n_files == 0) or (self.n_peaklist_files == 0):
            return None
        if nthreads is None:
            nthreads = min(cpu_count(), self.n_files)
        print('Run MINT')
        start = time.time()
        if nthreads > 1:
            self.run_parallel(nthreads)
        else:
            results = []
            for i, filename in enumerate(self.files):
                args = {'filename': filename,
                        'peaklist': self.peaklist,
                        'q':None}
                results.append(process(args))
                self.progress = int(100 * (i - (nthreads/2)) // self.n_files)
            self._process_results_data_(results)
            self.progress = 100
        end = time.time()
        self.runtime = ( end - start )
        self.runtime_per_file = (self.runtime / self.n_files)
        self.runtime_per_peak = (self.runtime / self.n_files / len(self.peaklist))
        print(f'Total runtime: {self.runtime:.2f}s')
        print(f'Runtime per file: {self.runtime_per_file:.2f}s')
        print(f'Runtime per peak ({len(self.peaklist)}): {self.runtime_per_peak:.2f}s')


    def run_parallel(self, nthreads=1):
        '''Create multiprocessing queue and process all files in
           self.files with self.peaklist in parallel.'''
        pool = Pool(processes=nthreads)
        manager = Manager()
        queue = manager.Queue()
        args = []
        for i, filename in enumerate(self.files):
            args.append({'filename': filename,
                         'peaklist': self.peaklist,
                         'queue': queue})
        results = pool.map_async(process, args)
        # monitor progress
        while True:
            if results.ready():
                break
            else:
                size = queue.qsize()
                self.progress = 100 * (size - (nthreads//2)) // self.n_files
                time.sleep(1)
        self.progress = 100
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
    
    @property
    def n_files(self):
        return len(self.files)
    
    @files.setter
    def files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        for f in list_of_files:
            assert os.path.isfile(f), f'File not found ({f})'
        self._files = list_of_files

    @property
    def peaklist_files(self):
        return self._peaklist_files

    @property
    def n_peaklist_files(self):
        return len(self.peaklist_files)
    
    @property
    def peaklist(self):
        return self._peaklist

    @peaklist.setter
    def peaklist(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        for f in list_of_files:
            assert os.path.isfile(f), f'File not found ({f})'    
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
    def crosstab(self, col_name='peakArea'):
        return pd.crosstab(self.results.peakLabel, 
                           self.results.msFile, 
                           self.results[col_name], 
                           aggfunc=sum).astype(np.float64)
       
    def export(self, outfile=None):
        if outfile is None:
            file_buffer = io.BytesIO()
            writer = pd.ExcelWriter(file_buffer)
        else:
            writer = pd.ExcelWriter(outfile)
        self.results.to_excel(writer, 'Results Complete', index=False)
        self.crosstab.T.to_excel(writer, 'PeakArea Summary', index=True)
        meta = pd.DataFrame({'Version': [self.version], 
                                'Date': [str(date.today())]}).T[0]
        meta.to_excel(writer, 'MetaData', index=True, header=False)
        writer.close()
        if outfile is None:
            return file_buffer.seek(0)
