import os
import ipywidgets as widgets

from ipywidgets import Button, HBox, VBox, Textarea, Layout

from ipywidgets import IntProgress as Progress

from .SelectFilesButton import SelectFilesButton
from .Mint import Mint as MintBase
from pathlib import Path as P 

HOME = str(P.home())

class Mint(MintBase):
    def __init__(self, *args, **kwargs):

        self.progress_callback = self.set_progress

        super().__init__(progress_callback=self.progress_callback, *args, **kwargs) 

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

        self.optimize_rt_button = Button(description="Optimize RT")
        self.optimize_rt_button.on_click(self.optimize_retention_times)
        
        self.download_button = Button(description="Export")
        self.download_button.on_click(self.export_action)
        self.download_button.style.button_color = 'lightgray'

        self.progress_bar = Progress(min=0, max=100, layout=Layout(width='90%'), 
                                 description='Progress:', bar_style='info')

        self.output = widgets.Output()

        self.layout = VBox([
                    HBox([self.peaklist_files_button,
                          self.ms_files_button,
                           ]),
                    HBox([self.detect_peaks_button,
                          self.optimize_rt_button]),
                    self.message_box,
                    HBox([self.run_button, 
                          self.download_button]),
                    self.progress_bar                  
                ])

    def show(self):
        return self.layout
            
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
        super(Mint, self).detect_peaks(**kwargs)

    def set_progress(self, value):
        self.progress_bar.value = value
    
    def export_action(self, b=None, filename=None):
        if filename is None:
            filename = 'MINT__results.xlsx'
            filename = os.path.join(HOME, filename)
        self.export(filename)
        self.message_box.value += f'\n\nExported results to: {filename}'