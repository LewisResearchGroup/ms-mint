# ms_mint/Mint.py

import os
import pandas as pd
import numpy as np
import time

from pathlib import Path as P

from multiprocessing import Pool, Manager, cpu_count

from tqdm import tqdm

from .processing import process_ms1_files_in_parallel
from .io import export_to_excel
from .standards import MINT_RESULTS_COLUMNS, PEAKLIST_COLUMNS, DEPRICATED_LABELS
from .peaklists import read_peaklists, check_peaklist, standardize_peaklist
from .peak_detection import OpenMSFFMetabo

import ms_mint

class Mint(object):
    def __init__(self, verbose:bool=False):
        self._verbose = verbose
        self._version = ms_mint.__version__
        self._progress_callback = None
        self.reset()
        if self.verbose:
            print('Mint Version:', self.version , '\n')
        self.peak_detector = OpenMSFFMetabo()

    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value:bool):
        self._verbose = value
    
    @property
    def version(self):
        return self._version
    
    def reset(self):
        self._files = []
        self._peaklist_files = []
        self._peaklist = pd.DataFrame(columns=PEAKLIST_COLUMNS)
        self._results = pd.DataFrame({i: [] for i in MINT_RESULTS_COLUMNS})
        self._all_df = None
        self._progress = 0
        self.runtime = None
        self._status = 'waiting'
        self._messages = []

    def clear_peaklist(self):
        self._peaklist = pd.DataFrame(columns=PEAKLIST_COLUMNS)

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
            
        if self.verbose: print(f'Run MINT with {nthreads} processes:')
        
        start = time.time()
        if nthreads > 1:
            self.run_parallel(nthreads=nthreads, mode=mode)
        else:
            results = []
            for i, filename in tqdm(enumerate(self.ms_files), total=self.n_files):
                args = {'filename': filename,
                        'peaklist': self.peaklist,
                        'q': None, 
                        'mode': mode}
                results.append(process_ms1_files_in_parallel(args))
                self.progress = int(100 * (i / self.n_files))
            self.results = pd.concat(results)
            self.progress = 100
            
        end = time.time()
        self.runtime = ( end - start )
        self.runtime_per_file = (self.runtime / self.n_files)
        self.runtime_per_peak = (self.runtime / self.n_files / len(self.peaklist))
        
        if self.verbose: 
            print(f'Total runtime: {self.runtime:.2f}s')
            print(f'Runtime per file: {self.runtime_per_file:.2f}s')
            print(f'Runtime per peak ({len(self.peaklist)}): {self.runtime_per_peak:.2f}s\n')
            print('Results:', self.results )
        self._status = 'done'

    def detect_peaks(self, **kwargs):
        detected = self.peak_detector.fit_transform(self.ms_files, **kwargs)
        if detected is not None:
            self.peaklist = pd.concat([self.peaklist, detected])

    def run_parallel(self, nthreads=1, mode='standard'):
        pool = Pool(processes=nthreads)
        m = Manager()
        q = m.Queue()
        args = []
        for i, filename in enumerate(self.ms_files):
            args.append({'filename': filename,
                         'peaklist': self.peaklist,
                         'queue': q,
                         'mode': mode})
                   
        results = pool.map_async(process_ms1_files_in_parallel, args)
        
        # monitor progress
        while True:
            if results.ready():
                break
            else:
                size = q.qsize()
                self.progress = max(0, min(nthreads/2, int(100 * (size-nthreads / self.n_files) ) ) )
                time.sleep(1)
    
        self.progress = 100

        pool.close()
        pool.join()
        results = results.get()
        self.results = pd.concat(results)

    @property
    def messages(self):
        return self._messages
             
    @property
    def status(self):
        return self._status

    @property
    def files(self):
        return self._files

    @property
    def ms_files(self):
        return self._files
    
    @files.setter
    def files(self, list_of_files):
        print('Mint.files is deprecated, please use Mint.ms_files instead!')
        self.ms_files = list_of_files

    @property
    def n_files(self):
        return len(self.ms_files)
    
    @ms_files.setter
    def ms_files(self, list_of_files):
        if isinstance(list_of_files, str):
            list_of_files = [list_of_files]
        list_of_files = [str(P(i)) for i in list_of_files]
        for f in list_of_files:
            if not os.path.isfile(f): 
                print(f'W File not found ({f})')
        self._files = list_of_files
        if self.verbose:
            print( 'Set files to:\n' + '\n'.join(self.ms_files) + '\n' )

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
        if self.verbose: print( 'Set peaklist files to:\n'.join(self.peaklist_files) + '\n')
        self.peaklist = read_peaklists(list_of_files)

        
    @property
    def n_peaklist_files(self):
        return len(self.peaklist_files)
    
    @property
    def peaklist(self):
        return self._peaklist

    @peaklist.setter
    def peaklist(self, peaklist):
        peaklist = standardize_peaklist(peaklist)
        errors = check_peaklist(peaklist)
        if len(errors) != 0:
            peaklist = peaklist.drop_duplicates()
            error_string = '\n'.join(errors)
            if self.verbose:
                print(f'Errors in peaklist:\n{error_string}')
            self._messages = errors
        self._peaklist = peaklist
        if self.verbose:
            print('Set peaklists to:\n', self.peaklist.to_string(), '\n')

    @property
    def results(self):
        return self._results
    
    @results.setter
    def results(self, df):
        self._results = df

    @property
    def rt_projections(self):
        return DeprecationWarning('rt_projections is deprecated. Peak shapes are now '
                                 'directly stored in the results table (mint.results).')
        
    def crosstab(self, col_name='peak_area'):
        return pd.crosstab(self.results.peak_label, 
                           self.results.ms_file, 
                           self.results[col_name], 
                           aggfunc=sum).astype(np.float64)
    @property
    def progress_callback(self):
        return self._progress_callback
    
    @progress_callback.setter
    def progress_callback(self, func=None):
        self._progress_callback = func
    
    @property
    def progress(self):
        return self._progress
    
    @progress.setter
    def progress(self, value):
        assert value >= 0, value
        assert value <= 100, value
        self._progress = value
        if self.progress_callback is not None:
            self.progress_callback(value)

    def export(self, filename=None):
        fn = filename
        if fn is None:
            buffer = export_to_excel(self, filename=fn)
            return buffer
        elif fn.endswith('.xlsx'):
            export_to_excel(self, filename=fn)
        elif fn.endswith('.csv'):
            self.results.to_csv(fn, index=False)

    def load(self, fn):
        if self.verbose: print('Loading MINT state')
        if isinstance(fn, str):
            if fn.endswith('xlsx'):
                results = pd.read_excel(fn, sheet_name='Results').rename(columns=DEPRICATED_LABELS)
                ms_files = results.ms_file.drop_duplicates()
                self.results = results
                self.peaklist = pd.read_excel(fn, sheet_name='Peaklist')
                self.ms_files = ms_files
                return None
            elif fn.endswith('.csv'):
                results = pd.read_csv(fn).rename(columns=DEPRICATED_LABELS)
                ms_files = results.ms_file.drop_duplicates()
                peaklist = results[PEAKLIST_COLUMNS].drop_duplicates()
                self.results = results
                self.ms_files = ms_files
                self.peaklist = peaklist
                return None
        
