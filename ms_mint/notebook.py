import os, io
import ipywidgets as W

from glob import glob
from pathlib import Path as P

#from ipywidgets import Button, HBox, VBox, Textarea, Layout, FileUpload, Tab
#from ipywidgets import IntProgress as Progress
from ipyfilechooser import FileChooser
from IPython.display import display
from IPython.core.display import HTML

from .Mint import Mint as MintBase
from .MintPlotter import MintPlotter

HOME = str(P.home())


class Mint(MintBase):
    def __init__(self, *args, **kwargs):

        self.progress_callback = self.set_progress

        super().__init__(progress_callback=self.progress_callback, *args, **kwargs)

        fc = FileChooser()
        fc.show_only_dirs = True
        fc.default_path = os.getcwd()

        self.ms_storage_path = fc

        self.target_files_button = W.FileUpload(
            description="Peaklists", accept="csv,xlsx", multiple=False
        )
        self.target_files_button.observe(self.load_target, names="value")

        self.load_ms_button = W.Button(description="Load MS-files")
        self.load_ms_button.on_click(self.search_files)

        self.detect_peaks_button = W.Button(description="Detect Peaks")
        self.detect_peaks_button.on_click(self.detect_peaks)

        self.message_box = W.Textarea(
            value="",
            placeholder="Please, select ms-files define a target list.",
            description="",
            disabled=True,
            layout={"width": "90%", "height": "500px", "font_family": "monospace"},
        )

        self.run_button = W.Button(description="Run")
        self.run_button.on_click(self.run)
        self.run_button.style.button_color = "lightgray"

        self.optimize_rt_button = W.Button(description="Find closest peaks")
        self.optimize_rt_button.on_click(self.action_optimize_rt)

        self.download_button = W.Button(description="Export")
        self.download_button.on_click(self.export_action)
        self.download_button.style.button_color = "lightgray"

        self.progress_bar = W.IntProgress(
            min=0,
            max=100,
            layout=W.Layout(width="90%"),
            description="Progress:",
            bar_style="info",
        )

        self.output = W.Output()

        tabs = W.Tab()
        tabs.children = [
            W.HBox([self.ms_storage_path, self.load_ms_button]),
            W.HBox(
                [
                    self.target_files_button,
                    self.detect_peaks_button,
                    self.optimize_rt_button,
                ]
            ),
        ]

        tabs.set_title(0, "MS-Files")
        tabs.set_title(1, "Peaklists")

        self.layout = W.VBox(
            [
                tabs,
                self.message_box,
                W.HBox([self.run_button, self.download_button]),
                self.progress_bar,
            ]
        )

        self.plot = MintPlotter(self)

    def load_target(self, value):
        for fn, data in value["new"].items():
            self.load(io.BytesIO(data["content"]))
        self.message(f'{len(self.targets)} targets loaded.')

    def action_optimize_rt(self, b):
        if (self.n_files > 0) and len(self.target) > 0:
            self.optimize_rt()

    def message(self, text):
        self.message_box.value = f"{text}\n" + self.message_box.value

    def clear_messages(self):
        self.message_box.value = ""

    def search_files(self, b=None):
        self.ms_files = (
            glob(os.path.join(self.ms_storage_path.selected_path, "*mzXML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzHDF"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzxml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzhdf"))
        )
        self.message(f'{self.n_files} MS-files loaded.')

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

    def add_target_files(self, fns):
        if fns is not None:
            self.target_files = self.target_files + fns

    def list_files(self, b=None):
        text = f"{self.n_files} MS-files to process:\n"
        for i, line in enumerate(self.ms_files):
            text += line + "\n"
            if i > 10:
                text += line + "\n...\n"
                break
        text += "\nUsing peak list:\n"
        if len(self.target_files) != 0:
            text += "\n".join([str(i) for i in self.target_files])
        elif len(self.target) != 0:
            text += self.target.to_string()
        else:
            text += "\nNo target defined."

        if (self.n_files != 0) and (self.n_target_files != 0):
            self.run_button.style.button_color = "lightgreen"
        else:
            self.run_button.style.button_color = "lightgray"
        print(text)
        self.message(text)

    def run(self, b=None, **kwargs):
        self.message(f'Start processing...')
        self.progress = 0
        super(Mint, self).run(**kwargs)
        self.message("...finished processing.")
        if self.results is not None:
            self.download_button.style.button_color = "lightgreen"

    def detect_peaks(self, b=None, **kwargs):
        self.message("\n\nRun peak detection.")
        super(Mint, self).detect_peaks(**kwargs)

    def set_progress(self, value):
        self.progress_bar.value = value

    def export_action(self, b=None, filename=None):
        if filename is None:
            filename = "MINT__results.xlsx"
            filename = os.path.join(os.getcwd(), filename)
        self.export(filename)
        self.message(f"/nExported results to: {filename}")

    def state(self):
        return f'{len(self.ms_files)} {len(self.targets)}'