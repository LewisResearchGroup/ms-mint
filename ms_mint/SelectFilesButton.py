import ipywidgets as widgets
from ipywidgets import HTML
import traitlets
from IPython.display import display
from tkinter import Tk, filedialog

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, text='Button', default_color=None, callback=None, assign_to=None):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = text
        self.icon = "square-o"
        self.style.button_color = default_color
        # Set on click behavior.
        self.on_click(self.do_stuff)
        self.callback = callback
        self.assign_to = assign_to
        display(HTML("<style>textarea, input { font-family: monospace; }</style>"))
        display(HTML("<style>.container { width:%d%% !important; }</style>" %90))
   
    def do_stuff(self, b):
        try:
            self.select_files(b)
        except:
            return None
        if self.assign_to is not None:
            self.assign_to = b.files
        if self.callback is not None:
            self.callback()

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        b.files = filedialog.askopenfilename(multiple=True)
