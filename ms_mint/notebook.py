import os
import numpy as np
import ipywidgets as widgets
import seaborn as sns

from copy import copy
from ipywidgets import Button, HBox, VBox, Textarea,\
    Layout

from ipywidgets import IntProgress as Progress
from matplotlib import pyplot as plt 

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint as MintBase
from .plotly_tools import plot_peak_shapes
from .vis.clustering import hierarchical_clustering

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter

class Mint(MintBase):
    def __init__(self):
        super().__init__() 

        self.ms_files_button = SelectFilesButton(text='Select MS-files', callback=self.list_files)
        self.peaklist_files_button = SelectFilesButton(text='Peaklist', callback=self.list_files)
        self.detect_peaks_button = Button(description="Detect Peaks", callback=self.detect_peaks)

        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '90%', 
                    'height': '500px', 
                    'font_family': 'monospace'})
        
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run)
        self.run_button.style.button_color = 'lightgray'
        
        self.download_button = Button(description="Export")
        self.download_button.on_click(self.export)
        self.download_button.style.button_color = 'lightgray'

        self._results = None
        self.progress_bar = Progress(min=0, max=100, layout=Layout(width='90%'), 
                                 description='Progress:', bar_style='info')

        self.output = widgets.Output()
        self.progress_callback = self.set_progress

    def show(self):
        return VBox([
                    HBox([self.peaklist_files_button,
                          self.ms_files_button,
                          self.detect_peaks_button,
                           ]),
                    self.message_box,
                    HBox([self.run_button, 
                          self.download_button]),
                    self.progress_bar                  
                ])
            
    def files(self, files):
        super(Mint, self).files = files
        self.ms_files_button.files = files
        self.list_files()

    def list_files(self, b=None):
        text = 'mzXML files to process:\n'
        [self.files.append(i) for i in self.ms_files_button.files if (i.endswith('.mzXML') or (i.endswith('.mzML')))]
        try:
            [self.peaklist_files.append(i) for i in self.peaklist_files_button.files]
        except:
            pass
        for i, line in enumerate(self.files):
            text += line+'\n'
            if i > 10:
                line+'...\n'
                break
        text += '\nUsing peak list:\n'
        if len(self.peaklist_files) != 0:
            text += '\n'.join([str(i) for i in self.peaklist_files])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
        if (self.n_files != 0) and (self.n_peaklist_files != 0):
            self.run_button.style.button_color = 'lightgreen'
        else:
            self.run_button.style.button_color = 'lightgray'
       
    def run(self, b=None, **kwargs):
        self.progress = 0
        super(Mint, self).run(**kwargs)
        self.message_box.value += f'\n\nDone processing.'
        if self.results is not None:
            self.download_button.style.button_color = 'lightgreen'
    
    def detect_peaks(self):
        self.message_box.value += f'\n\nRun peak detection.'
        self.progress_bar.value = 50
        super(Mint, self).detect_peaks()
        self.progress_bar.value = 100

    def set_progress(self, value):
        self.progress_bar.value = value
    
    def export(self, b=None, filename=None):
        home = os.getenv("HOME")
        if filename is None:
            filename = 'MINT_output.xlsx'
        filename = os.path.join(home, filename)
        super(Mint, self).export(filename)
        self.message_box.value += f'\n\nExported results to: {filename}'
        
    def plot_clustering(self, data=None, title=None, figsize=(8,8), 
                        vmin=-3, vmax=3, xnbins=None, ynbins=None):

        simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.crosstab().apply(np.log1p)
        data.columns = [os.path.basename(i) for i in data.columns]        
        data = ((data.T - data.T.mean()) / data.T.std())
    
        self.clustered, fig = hierarchical_clustering( 
            data, vmin=vmin, vmax=vmax, figsize=figsize, 
            xnbins=xnbins, ynbins=ynbins )

        return fig

