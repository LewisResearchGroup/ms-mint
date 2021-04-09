import os, io
import ipywidgets as widgets
import tempfile

from glob import glob
from pathlib import Path as P 

from ipywidgets import Button, HBox, VBox, Textarea, Layout, FileUpload, Tab
from ipywidgets import IntProgress as Progress
from ipyfilechooser import FileChooser
from IPython.display import display
from IPython.core.display import HTML

from .Mint import Mint as MintBase


HOME = str(P.home())

class Mint(MintBase):
    def __init__(self, *args, **kwargs):

        self.progress_callback = self.set_progress

        super().__init__(progress_callback=self.progress_callback, *args, **kwargs)
        
        fc = FileChooser()
        fc.show_only_dirs = True
        fc.default_path = tempfile.gettempdir()
        
        self.ms_storage_path = fc

        self.ms_upload = FileUpload()

        self.peaklist_files_button = FileUpload(description='Peaklists', accept='csv,xlsx', multiple=False)

        self.peaklist_files_button.observe(self.load_peaklist, names='value')

        self.load_ms_button = Button(description='Load MS-files')

        self.load_ms_button.on_click(self.search_files)

        self.detect_peaks_button = Button(description="Detect Peaks")
        self.detect_peaks_button.on_click(self.detect_peaks)

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

        self.optimize_rt_button = Button(description="Find closest peaks")
        self.optimize_rt_button.on_click(self.action_optimize_rt)
        
        self.download_button = Button(description="Export")
        self.download_button.on_click(self.export_action)
        self.download_button.style.button_color = 'lightgray'

        self.progress_bar = Progress(min=0, max=100, layout=Layout(width='90%'), 
                                 description='Progress:', bar_style='info')

        self.output = widgets.Output()

        tabs = Tab()
        tabs.children = [
                         HBox([self.ms_storage_path, self.ms_upload, self.load_ms_button]),
                         HBox([self.peaklist_files_button, self.detect_peaks_button, self.optimize_rt_button]),
                         ]

        tabs.set_title(0, 'MS-Files')
        tabs.set_title(1, 'Peaklists')

        self.layout = VBox([                      
                      tabs,
                            self.message_box,
                      HBox([self.run_button, 
                            self.download_button]),
                            self.progress_bar   
                ])
    
    def load_peaklist(self, value):
        for fn, data in value['new'].items():
            self.load(io.BytesIO(data['content']))
        self.list_files()

    def action_optimize_rt(self, b):
        if (self.n_files > 0) and len(self.peaklist)>0:
            self.optimize_rt()
        
    def message(self, text):
        self.message_box.value = f'{text}\n' + self.message_box.value

    def clear_messages(self):
        self.message_box.value = ''

    def search_files(self, b=None):
        self.ms_files = (glob(os.path.join(self.ms_storage_path.selected_path, '*mzXML')) +
                         glob(os.path.join(self.ms_storage_path.selected_path, '*mzML')) +
                         glob(os.path.join(self.ms_storage_path.selected_path, '*mzHDF')) + 
                         glob(os.path.join(self.ms_storage_path.selected_path, '*mzxml')) +
                         glob(os.path.join(self.ms_storage_path.selected_path, '*mzml')) +
                         glob(os.path.join(self.ms_storage_path.selected_path, '*mzhdf')) )
        self.list_files()

    def show(self):
        display(HTML("<style>textarea, input { font-family: monospace; }</style>"))
        return self.layout
            
    def files(self, files):
        super(Mint, self).ms_files = files
        self.ms_files_button.files = files
        self.list_files()

    def add_ms_files(self, fns):
        if fns is not None:
            self.ms_files = self.ms_files + fns
        self.list_files()

    def add_peaklist_files(self, fns):
        if fns is not None:
            self.peaklist_files = self.peaklist_files + fns
        self.list_files()

    def list_files(self, b=None):
        text = f'{self.n_files} MS-files to process:\n'
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
        self.message(text)
        if (self.n_files != 0) and (self.n_peaklist_files != 0):
            self.run_button.style.button_color = 'lightgreen'
        else:
            self.run_button.style.button_color = 'lightgray'
       
    def run(self, b=None, **kwargs):
        self.progress = 0
        super(Mint, self).run(**kwargs)
        self.message('Done processing MS-files.')
        if self.results is not None:
            self.download_button.style.button_color = 'lightgreen'
    
    def detect_peaks(self, b=None, **kwargs):
        self.message('\n\nRun peak detection.')
        super(Mint, self).detect_peaks(**kwargs)

    def set_progress(self, value):
        self.progress_bar.value = value
    
    def export_action(self, b=None, filename=None):
        if filename is None:
            filename = 'MINT__results.xlsx'
            filename = os.path.join(HOME, filename)
        self.export(filename)
        self.message(f'\n\nExported results to: {filename}')

