import ipywidgets as widgets
from ipywidgets import HTML
import traitlets
from IPython.display import display
from tkinter import Tk, filedialog

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, text='Button', default_color='orange', callback=None):
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
        display(HTML("<style>textarea, input { font-family: monospace; }</style>"))
        display(HTML("<style>.container { width:%d%% !important; }</style>" %90))
   
    def do_stuff(self, b):
        self.select_files(b)
        self.callback()
        if len(self.files) > 0:
            self.style.button_color = "lightgreen"
        else:
            self.style.button_color = "red"

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        try:
            # Create Tk root
            root = Tk()
            # Hide the main window
            root.withdraw()
            # Raise the root to the top of all windows.
            root.call('wm', 'attributes', '.', '-topmost', True)
            # List of selected fileswill be set to b.value
            b.files = filedialog.askopenfilename(multiple=True)
        except:
            pass
