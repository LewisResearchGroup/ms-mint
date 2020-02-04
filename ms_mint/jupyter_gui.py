import traitlets

import ipywidgets as widgets

from ipywidgets import Button, HBox, VBox, Textarea, HTML,\
    SelectMultiple, Select, Layout, Label, IntSlider

from ipywidgets import IntProgress as Progress

from IPython.display import display
from tkinter import Tk, filedialog

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint



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
        
        self.list_button = Button(description="List Files")
        self.list_button.on_click(self.list_files)
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run_mint)
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.export_mint_results  )
        self._results = None
        self.progress = Progress(min=0, max=100, layout=Layout(width='90%'))
        self.peaks = []
        self._rt_projections = None
        self.plot_button = Button(description="Plot Peaks")
        self.plot_button.on_click(self.plot_button_on_click)
        self.plot_peak_selector = SelectMultiple(
            options=[], layout=Layout(width='30%', height='90px', Label='test'))
        self.plot_file_selector = SelectMultiple(
            options=[], layout=Layout(width='30%', height='90px'))
        self.plot_highlight_selector = SelectMultiple(
            options=[], layout=Layout(width='30%', height='90px'))
        self.download_html = HTML("""Nothing to download""")
        self.output = widgets.Output()
        self.peakLabels = []
        self.plot_ncol_slider = IntSlider(min=1, max=5, step=1, value=1)

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
                    HBox([self.plot_peak_selector,
                          self.plot_file_selector,
                          self.plot_highlight_selector]),
                    self.plot_button,
                    HBox([Label('N columns'), self.plot_ncol_slider])
                ])
            
    def list_files(self, b=None):
        text = 'mzXML files to process:\n'
        self.mint.files = [i for i in self.ms_files_button.files if (i.endswith('.mzXML') or (i.endswith('.mzML')))]
        try:
            self.mint.peaklist_files = [i for i in self.peaklist_files_button.files]
        except:
            pass
        for line in self.mint.files:
            text += line+'\n'
        text += '\n\nUsing peak list:\n'
        if len(self.mint.peaklist_files) != 0:
            text += '\n'.join([str(i) for i in self.mint.peaklist_files])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
    
    def plot_button_on_click(self, button):
        pass
    
    def run_mint(self, b, **kwargs):
        with self.output:
            self.mint.run(**kwargs)
    
    def export_mint_results(self, b, **kwargs):
        self.mint.export(**kwargs)
        