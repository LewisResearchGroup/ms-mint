import os
import ipywidgets as widgets

from ipywidgets import Button, HBox, VBox, Textarea,\
    Layout

from ipywidgets import IntProgress as Progress

from tkinter import Tk, filedialog

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint
from .plotly_tools import plot_rt_projections

class JupyterGUI():
    def __init__(self, mint=None):
        
        if mint is None:
            self.mint = Mint()
        else:
            self.mint = mint
        
        self.ms_files_button = SelectFilesButton(text='Select MS-files', callback=self.list_files)
        self.peaklist_files_button = SelectFilesButton(text='Peaklist', callback=self.list_files)
        
        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '90%', 
                    'height': '500px', 
                    'font_family': 'monospace'})
        
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run_mint)
        self.run_button.style.button_color = 'lightgray'

        self.download_button = Button(description="Download")
        self.download_button.on_click(self.export)
        self.download_button.style.button_color = 'lightgray'

        self._results = None
        self.progress = Progress(min=0, max=100, layout=Layout(width='90%'), 
                                 description='Progress:', bar_style='info')
        self.output = widgets.Output()
        self.mint.progress_callback = self.set_progress

    def show(self):
        return VBox([
                    HBox([self.ms_files_button, 
                           self.peaklist_files_button, 
                           ]),
                    self.message_box,
                    HBox([self.run_button, 
                          self.download_button]),
                    self.progress                    
                ])
            
    def list_files(self, b=None):
        text = 'mzXML files to process:\n'
        self.mint.files = [i for i in self.ms_files_button.files if (i.endswith('.mzXML') or (i.endswith('.mzML')))]
        try:
            self.mint.peaklist_files = [i for i in self.peaklist_files_button.files]
        except:
            pass
        for i, line in enumerate(self.mint.files):
            text += line+'\n'
            if i > 10:
                line+'...\n'
                break
        text += '\nUsing peak list:\n'
        if len(self.mint.peaklist_files) != 0:
            text += '\n'.join([str(i) for i in self.mint.peaklist_files])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
        if (self.mint.n_files != 0) and (self.mint.n_peaklist_files != 0):
            self.run_button.style.button_color = 'lightgreen'
        else:
            self.run_button.style.button_color = 'lightgray'
       
    def run_mint(self, b, **kwargs):
        self.mint.progress = 0
        self.mint.run(**kwargs)
        self.message_box.value += f'\n\nDone processing.'
        if self.mint.results is not None:
            self.download_button.style.button_color = 'lightgreen'

    def set_progress(self, value):
        self.progress.value = value
    
    def export(self, b):
        home = os.getenv("HOME")
        filename = os.path.join(home, 'MINT_output.xlsx')
        self.mint.export(filename)
        self.message_box.value += f'\n\nExported results to: {filename}'
    
    @property
    def results(self):
        return self.mint.results
    
    @property
    def crosstab(self):
        return self.mint.crosstab    
    
    @property
    def rt_projections(self):
        return self.mint.rt_projections
    
    def plot_rt_projections(self, **kwargs):
        return plot_rt_projections(self.mint, **kwargs)
