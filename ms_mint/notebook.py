import os
import numpy as np
import ipywidgets as widgets

from ipywidgets import Button, HBox, VBox, Textarea, Layout

from ipywidgets import IntProgress as Progress

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint as MintBase
from .vis.plotly.plotly_tools import plot_heatmap
from .vis.mpl import plot_peak_shapes, hierarchical_clustering

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
from pathlib import Path as P 

HOME = str(P.home())

class Mint(MintBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

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
        self.download_button.on_click(self.export_action)
        self.download_button.style.button_color = 'lightgray'

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
        text = f'{self.n_files} MS-files to process:\n'
        [self.ms_files.append(i) for i in self.ms_files_button.files 
                                 if (i.lower().endswith('.mzxml') or 
                                    (i.lower.endswith('.mzml')))]
        try:
            [self.peaklist_files.append(i) for i in self.peaklist_files_button.files]
        except:
            pass
        for i, line in enumerate(self.ms_files):
            text += line+'\n'
            if i > 10:
                text += line+'\n...\n'
                break
        text += '\nUsing peak list:\n'
        if len(self.peaklist_files) != 0:
            text += '\n'.join([str(i) for i in self.peaklist_files])
        elif len(self.peaklist) != 0:
            text += self.peaklist.to_string()
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
        self.message_box.value += '\n\nDone processing.'
        if self.results is not None:
            self.download_button.style.button_color = 'lightgreen'
    
    def detect_peaks(self, **kwargs):
        self.message_box.value += '\n\nRun peak detection.'
        self.progress_bar.value = 50
        super(Mint, self).detect_peaks(**kwargs)
        self.progress_bar.value = 100

    def set_progress(self, value):
        self.progress_bar.value = value
    
    def export_action(self, b=None, filename=None):
        if filename is None:
            filename = 'MINT__results.xlsx'
            filename = os.path.join(HOME, filename)
        self.export(filename)
        self.message_box.value += f'\n\nExported results to: {filename}'
       
    def plot_clustering(self, data=None, title=None, figsize=(8,8), target_var='peak_max',
                        vmin=-3, vmax=3, xmaxticks=None, ymaxticks=None, transpose=False,
                        normalize_by_row=True, transform_func='log1p', 
                        transform_filenames_func='basename', **kwargs):
        '''
        Performs a cluster analysis and plots a heatmap. If no data is provided, 
        data is taken form self.crosstab(target_var).
        First the data is transformed with the transform function (default is log(x+1)).
        If normalize_by_row==True the (transformed) data is also row-normalized (z=(x-μ)/σ)).
        A hierachical cluster analysis is performed with the (transformed) data.
        The original data is then re-ordered accoring to the resulting clusters.
        The result is stored in self.clustered.
        '''

        simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.crosstab(target_var).copy()

        tmp_data = data.copy()

        if transform_func == 'log1p':
            transform_func = np.log1p

        if transform_func is not None:
            tmp_data = tmp_data.apply(transform_func)

        if transform_filenames_func == 'basename':
            transform_filenames_func = os.path.basename

        if transform_filenames_func is not None:
            tmp_data.columns = [transform_filenames_func(i) for i in tmp_data.columns] 

        if normalize_by_row:
            tmp_data = ((tmp_data.T - tmp_data.T.mean()) / tmp_data.T.std())

        if transpose: tmp_data = tmp_data.T    

        clustered, fig, ndx_x, ndx_y = hierarchical_clustering( 
            tmp_data, vmin=vmin, vmax=vmax, figsize=figsize, 
            xmaxticks=xmaxticks, ymaxticks=ymaxticks, **kwargs )

        if not transpose:
            self.clustered = data.iloc[ndx_y, ndx_x]
        else:
            self.clustered = data.iloc[ndx_x, ndx_y]

        return fig
    
    def plot_peak_shapes(self, **kwargs):
        return plot_peak_shapes(self.results, **kwargs)

    def plot_heatmap(self, target_var='peak_max', **kwargs):
        return plot_heatmap(self.crosstab(target_var), **kwargs)