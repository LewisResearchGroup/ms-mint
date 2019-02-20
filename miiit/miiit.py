import os
import uuid
import datetime
import traitlets
import ipywidgets as widgets
import pandas as pd

from ipywidgets import Button
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
from tkinter import Tk, filedialog
from tqdm import tqdm_notebook
from pyteomics import mzxml

from IPython.display import clear_output
from ipywidgets import Button, HBox, VBox, Textarea, HTML
import time

from pathlib import Path as P


MIIIT_ROOT = os.path.dirname(__file__)

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, text='Button', callback=None):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = text
        self.icon = "square-o"
        self.style.button_color = "orange"
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

def integrate_peaks(filename, peaklist):
    df = mzxml_to_pandas_df(filename)
    peaklist = get_peaklistfrom(peaklist)
    peaklist.index = range(len(peaklist))
    results = []
    for peak in to_peaks(peaklist):
        result = integrate_peak(df, *peak)
        results.append(result)
    results = pd.concat(results)
    results.index = range(len(results))
    results['mzxmlFile'] = os.path.basename(filename)
    results['mzxmlPath'] = os.path.dirname(filename)
    return pd.merge(peaklist, results, right_index=True, left_index=True)


def integrate_peak(df, mz, dmz, rtmin, rtmax, peaklabel=None):
    slizE =slice_ms1_mzxml(df, rtmin, rtmax, mz, dmz)
    peakArea = slizE['intensity array'].sum()
    peakAreaTop10 = slizE['intensity array'].sort_values().tail(10).sum()
    peakAreaTop3 = slizE['intensity array'].sort_values().tail(3).sum()
    peakAreaTop = slizE['intensity array'].sort_values().tail(1).sum()
    peakNumberOfPoints = len(slice_ms1_mzxml(df, rtmin, rtmax, mz, dmz))
    result = pd.DataFrame({'rtmin': [rtmin], 
                           'rtmax': [rtmax],
                           'peakMz': [mz],
                           'peakMzWidth[ppm]': [dmz],
                           'peakArea': [peakArea],
                           'peakAreaTop': [peakAreaTop],
                           'peakAreaTop3': [peakAreaTop3],                
                           'peakAreaTop10': [peakAreaTop10],
                           'peakNumberOfPoints': [peakNumberOfPoints]})
    return result[['peakArea', 'peakAreaTop', 'peakAreaTop3', 'peakAreaTop10', 'peakNumberOfPoints']]


def get_peaklistfrom(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    peaklist = []
    for file in filenames:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, usecols=['peakLabel', 'peakMz', 'peakMzWidth[ppm]','rtmin', 'rtmax'])
            df['peakListFile'] = file
            peaklist.append(df)
    return pd.concat(peaklist)


def to_peaks(peaklist):
    return [list(i) for i in list(peaklist[['peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax']].values)]

def mzxml_to_pandas_df(filename):
    slices = []
    file = mzxml.MzXML(filename)
    while True:
        try:
            slices.append(pd.DataFrame(file.next()))
        except:
            break
    df = pd.concat(slices)
    df_to_numeric(df)
    return df


def df_to_numeric(df):
    for col in df:
        df.loc[:, col] = pd.to_numeric(df[col], errors='ignore')


def slice_ms1_mzxml(df, rtmin, rtmax, mz, dmz):
    '''
    Returns a slize of a metabolomics mzXML file.
    df - pandas.DataFrame that has columns 
            * 'retentionTime'
            * 'm/z array'
            * 'rtmin'
            * 'rtmax'
    rtmin - minimal retention time
    rtmax - maximal retention time
    mz - center of mass (m/z)
    dmz - width of the mass window in ppm
    '''
    df_slice = df.loc[(rtmin <= df.retentionTime) &
                      (df.retentionTime <= rtmax) &
                      (mz-0.0001*dmz <= df['m/z array']) & 
                      (df['m/z array'] <= mz+0.0001*dmz)]
    return df_slice

from ipywidgets import IntProgress as Progress

class App():
    def __init__(self):
        self.mzxml = SelectFilesButton(text='Select mzXML', callback=self.list_files)
        self.peaklist = SelectFilesButton(text='Peaklist', callback=self.list_files)
        #self.peaklist.files = [str(P(MIIIT_ROOT)/P('static/Standard_Peaklist.csv'))]
        self.peaklist.files = [P(os.path.abspath(f'{MIIIT_ROOT}/../static/Standard_Peaklist.csv'))]
        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '95%', 'height': '500px', 'font_family': 'monospace'})
        self.list_button = Button(description="List Files")
        self.list_button.on_click(self.list_files)
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run)
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.download)
        self.results = None
        self.download_html = HTML("""Nothing to download""")
        self.out = widgets.Output(layout={'border': '1px solid black'})
        self.progress = Progress(min=0, max=100)
        #os.chdir(os.getenv("HOME"))
        
    def run(self, b):
        try:
            # print('Running')
            # time.sleep(1)
            results = []
            peaklist = self.peaklist.files
            for i in peaklist:
                self.message_box.value = check_peaklist(i)
            n_files = len(self.mzxml.files)
            processed_files = []
            with self.out:
                for i, filename in enumerate(self.mzxml.files):
                    processed_files.append(filename)
                    run_text = 'Processing: \n' + '\n'.join(processed_files[-20:])
                    self.message_box.value = run_text
                    self.progress.value = 100 * (i+1) / n_files
                    self.progress.description = f'{i+1}/{n_files}'
                    result = integrate_peaks(filename, peaklist)
                    results.append(result)
            self.results = pd.concat(results)
            self.message_box.value = self.results_crosstab().to_string()
            self.download(None)
            return self.results
        except Exception as e:
            self.message_box.value = str(e)

    def list_files(self, b=None):
        text = 'mzXML files to process:\n'
        self.mzxml.files = [i for i in self.mzxml.files if i.endswith('.mzXML')]
        for line in self.mzxml.files:
            text += line+'\n'
        text += '\n\nUsing peak list:\n'
        if len(self.peaklist.files) != 0:
            text += '\n'.join([str(i) for i in self.peaklist.files])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
        
    def download(self, b):
        if self.results is None:
            print('First you have to create some results.')
        else:
            uid = str(uuid.uuid4()).split('-')[-1]
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            os.makedirs('output', exist_ok=True)
            filename = P('output') / P('{}-metabolomics_peak_intensity-{}.xlsx'.format(now, uid))
            writer = pd.ExcelWriter(filename)
            self.results.to_excel(writer, 'MainTable', index=False)
            self.results_crosstab('peakArea').to_excel(writer, 'peakArea', index=True)
            writer.save()
            self.download_html.value = """<a download='{}' href='{}'>Download</a>""".format(filename, filename)
    
    def results_crosstab(self, varName='peakArea'):
        return pd.merge(self.results[['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax']].drop_duplicates(),
                        pd.crosstab(self.results.peakLabel, self.results.mzxmlFile, self.results[varName], aggfunc=sum),
                        on='peakLabel')    
    
    def gui(self):
        return VBox([HBox([self.mzxml, self.peaklist, self.run_button]),
              self.message_box,
              self.progress,
              self.download_html])

def check_peaklist(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Cannot find peaklist ({filename}).')
    try:
        df = pd.read_csv(P(filename))
    except:
        return f'Cannot open peaklist {filename}'
    try:
        df[['peakLabel', 'peakMz', 'peakMzWidth[ppm]','rtmin', 'rtmax']]
    except:
        return f"Not all columns found.\n\
 Please make sure the peaklist file has at least:\
 'peakLabel', 'peakMz', 'peakMzWidth[ppm]','rtmin', 'rtmax'"
    return f'Peaklist file ok ({filename})'

app = App()