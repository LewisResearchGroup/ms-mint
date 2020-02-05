import os
import traitlets
import ipywidgets as widgets

from ipywidgets import Button, HBox, VBox, Textarea, HTML,\
    SelectMultiple, Select, Layout, Label, IntSlider

from ipywidgets import IntProgress as Progress

from IPython.display import display
from tkinter import Tk, filedialog

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint
from .tools import browser_export

class GUI():
    def __init__(self, mint=None):
        
        if mint is None:
            self.mint = Mint()
        else:
            self.mint = mint
        
        self.ms_files_button = SelectFilesButton(text='Select mzXML', callback=self.list_files)
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
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.export)
        self._results = None
        self.progress = Progress(min=0, max=100, layout=Layout(width='90%'))
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
                    self.progress,
                    HBox([Label('Peak', layout=Layout(width='30%')),
                          Label('File', layout=Layout(width='30%')), 
                          Label('Highlight', layout=Layout(width='30%'))]),
                    
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
       
    def run_mint(self, b, **kwargs):
        self.mint.progress = 0
        with self.output:
            self.mint.run(**kwargs)
        self.message_box.value += f'\n\nDone processing.'

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