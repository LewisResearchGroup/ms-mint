import os
import pandas as pd
import numpy as np
import time
import io

from .tools import read_peaklists, process,\
    restructure_rt_projections, PEAKLIST_COLUMNS,\
    check_peaklist

from multiprocessing import Pool, Manager, cpu_count
from datetime import date

import ms_mint

class Mint(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._files = []
        self._peaklist_files = []
        self._peaklist = pd.DataFrame(columns=PEAKLIST_COLUMNS)
        self._rt_projections = None
        columns = PEAKLIST_COLUMNS + ['peak_area', 'ms_file', 'ms_path', 'peaklist']
        self._results = pd.DataFrame({i: [] for i in columns})
        #self._callback_progress = None
        self._all_df = None
        self.version = ms_mint.__version__
        self.progress = 0
        self.runtime = None
        self._status = 'waiting'

    def run(self, nthreads=None, mode='standard'):
        '''
        Run MINT with set up peaklist and ms-files.
        ----
        Args
            - nthreads: int or None, default = None
                * None: Run with min(n_cpus, c_files) CPUs
                * 1: Run without multiprocessing on one CPU
                * >1: Run with multiprocessing enabled using nthreads threads.
            - mode: str, default = 'standard'
                * 'standard': calculates peak shaped projected to RT dimension
                * 'express': omits calculation of other features, only peak_areas
        '''
        self._status = 'running'
        if (self.n_files == 0) or ( len(self.peaklist) == 0):
            return None
        if nthreads is None:
            nthreads = min(cpu_count(), self.n_files)
            
        print('Run MINT')
        start = time.time()
        if nthreads > 1:
            self.run_parallel(nthreads, mode)
        else:
            results = []
            for i, filename in enumerate(self.files):
                args = {'filename': filename,
                        'peaklist': self.peaklist,
                        'q': None, 
                        'mode': mode}
                results.append(process(args))
                self.progress = int(100 * (i - (nthreads/2)) // self.n_files)
            self._process_results_data_(results, mode=mode)
            self.progress = 100
            
        end = time.time()
        self.runtime = ( end - start )
        self.runtime_per_file = (self.runtime / self.n_files)
        self.runtime_per_peak = (self.runtime / self.n_files / len(self.peaklist))
        
        print(f'Total runtime: {self.runtime:.2f}s')
        print(f'Runtime per file: {self.runtime_per_file:.2f}s')
        print(f'Runtime per peak ({len(self.peaklist)}): {self.runtime_per_peak:.2f}s')
        self._status = 'waiting'

    def run_parallel(self, nthreads=1, mode='standard'):
        pool = Pool(processes=nthreads)
        m = Manager()
        q = m.Queue()
        args = []
        for i, filename in enumerate(self.files):
            args.append({'filename': filename,
                         'peaklist': self.peaklist,
                         'q':q,
                         'mode': mode})
        results = pool.map_async(process, args)
        
        # monitor progress
        while True:
            if results.ready():
                break
            else:
                size = q.qsize()
                self.progress = 100 * (size - (nthreads//2)) // self.n_files
                time.sleep(1)

        self.progress = 100
        pool.close()
        pool.join()
        self._process_results_data_(results.get(), mode=mode)

    def _process_results_data_(self, results, mode):
        self.results = pd.concat([i[0] for i in results])
        if mode == 'standard':
            rt_projections = {}
            [rt_projections.update(i[1]) for i in results]
            self.rt_projections = restructure_rt_projections(rt_projections)
             
    @property
    def status(self):
        return self._status

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

    @peaklist_files.setter
    def peaklist_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        if not isinstance(list_of_files, list):
            raise ValueError('Input should be a list of files.')
        for f in list_of_files:
            assert os.path.isfile(f), f'File not found ({f})'    
        self._peaklist_files = list_of_files
        self.peaklist = read_peaklists(list_of_files)
        
    @property
    def n_peaklist_files(self):
        return len(self.peaklist_files)
    
    @property
    def peaklist(self):
        return self._peaklist

    @peaklist.setter
    def peaklist(self, peaklist):
        check_peaklist(peaklist)
        self._peaklist = peaklist

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
    def crosstab(self, col_name='peak_area'):
        return pd.crosstab(self.results.peak_label, 
                           self.results.ms_file, 
                           self.results[col_name], 
                           aggfunc=sum).astype(np.float64)
    #@property
    #def callback_progress(self):
    #    return self._callback_progress
    
    #@callback_progress.setter
    #def callback_progress(self, func=None):
    #    self._callback_progress = func
        
    def export(self, outfile=None):
        if outfile is None:
            file_buffer = io.BytesIO()
            writer = pd.ExcelWriter(file_buffer)
        else:
            writer = pd.ExcelWriter(outfile)#, engine='xlsxwriter')
        self.results.to_excel(writer, 'Results Complete', index=False)
        self.crosstab.T.to_excel(writer, 'PeakArea Summary', index=True)
        meta = pd.DataFrame({'Version': [self.version], 
                                'Date': [str(date.today())]}).T[0]
        meta.to_excel(writer, 'MetaData', index=True, header=False)
        writer.close()
        if outfile is None:
            return file_buffer.seek(0)

       

