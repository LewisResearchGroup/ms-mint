"""
Experimental module to run Mint interactively inside the Jupyter notebook.


code-block::

    from ms_mint.notebook import Mint

    mint = Mint()

    mint.display()


"""

import os, io
import ipywidgets as W

from glob import glob
from pathlib import Path as P

from ipyfilechooser import FileChooser
from IPython.display import display
from IPython.core.display import HTML

from .Mint import Mint as _Mint_

HOME = str(P.home())


class Mint(_Mint_):
    """
    MINT with added functions for interactive use in Jupyter Notebook (experimental).
    """

    def __init__(self, *args, **kwargs):

        self.progress_callback = self._set_progress_

        super().__init__(progress_callback=self.progress_callback, *args, **kwargs)

        fc = FileChooser()
        fc.show_only_dirs = True
        fc.default_path = os.getcwd()

        self.ms_storage_path = fc

        self.target_files_button = W.FileUpload(
            description="Peaklists", accept="csv,xlsx", multiple=False
        )
        self.target_files_button.observe(self._load_target_from_bytes_, names="value")

        self.load_ms_button = W.Button(description="Load MS-files")
        self.load_ms_button.on_click(self._search_files_)

        self.message_box = W.Textarea(
            value="",
            placeholder="Please, select ms-files define a target list.",
            description="",
            disabled=True,
            layout={"width": "90%", "height": "500px", "font_family": "monospace"},
        )

        self.run_button = W.Button(description="Run")
        self.run_button.on_click(self._run_)
        self.run_button.style.button_color = "lightgray"

        self.download_button = W.Button(description="Export")
        self.download_button.on_click(self._export_action_)
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

    def _load_target_from_bytes_(self, value):
        for fn, data in value["new"].items():
            self.load(io.BytesIO(data["content"]))
        self._message_(f"{len(self.targets)} targets loaded.")

    @property
    def messages(self):
        return self._messages

    def _message_(self, text):
        self.message_box.value = f"{text}\n" + self.message_box.value

    def _clear_messages_(self):
        self.message_box.value = ""

    def _search_files_(self, b=None):
        self.ms_files = (
            glob(os.path.join(self.ms_storage_path.selected_path, "*mzXML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzHDF"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzxml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzhdf"))
        )
        self.message(f"{self.n_files} MS-files loaded.")

    def display(self):
        """
        Display control elements in Jupyter notebook.

        :return: IPython Widgets elements.
        """
        display(HTML("<style>textarea, input { font-family: monospace; }</style>"))
        return self.layout

    def _run_(self, b=None, **kwargs):
        self.message(f"Start processing...")
        self.progress = 0
        self.run(**kwargs)
        self.message("...finished processing.")
        if self.results is not None:
            self.download_button.style.button_color = "lightgreen"

    def _set_progress_(self, value):
        self.progress_bar.value = value

    def _export_action_(self, b=None, filename=None):
        if filename is None:
            filename = "MINT__results.xlsx"
            filename = os.path.join(os.getcwd(), filename)
        self.export(filename)
        self.message(f"/nExported results to: {filename}")
