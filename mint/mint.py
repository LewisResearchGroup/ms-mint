import os
import uuid
import time
import datetime

import pandas as pd

import ipywidgets as widgets    
from ipywidgets import Button, HBox, VBox, Textarea, HTML,\
    SelectMultiple, Select, Layout, Label, IntSlider
from ipywidgets import IntProgress as Progress
from IPython.display import display, clear_output

from tqdm import tqdm_notebook
from pyteomics import mzxml
from pathlib import Path as P

from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool, PanTool, ResetTool,\
    WheelZoomTool, ZoomInTool, ZoomOutTool, SaveTool
from bokeh.layouts import gridplot

#from functools import lru_cache



from .SelectFilesButton import SelectFilesButton

MIIIT_ROOT = os.path.dirname(__file__)
STANDARD_PEAKLIST = os.path.abspath(str(P(MIIIT_ROOT)/P('../static/Standard_Peaklist.csv')))

DEVEL = True

rt_projections_test_data = {
    'TestLabel1': {
        'File1': pd.Series([1, 0, 0], 
                           index=pd.Index([1, 2, 3], 
                                          name='retentionTime')),
        'File2': pd.Series([0, 1, 0], 
                           index=pd.Index([1, 2.4, 2.8], 
                                          name='retentionTime'))
        },
    'TestLabel2': {
        'File1': pd.Series([7., .8, .8], 
                           index=pd.Index([4, 5, 6], 
                                          name='retentionTime')),
        'File2': pd.Series([53., .56, .12], 
                           index=pd.Index([4, 5.4, 6], 
                                          name='retentionTime'))
        }
    }

class App():
    def __init__(self):
        output_notebook(hide_banner=True)
        self.mzxml = SelectFilesButton(text='Select mzXML', callback=self.list_files)
        self.peaklist = SelectFilesButton(text='Peaklist', default_color='lightgreen', callback=self.list_files)
        self.peaklist.files = [P(os.path.abspath(f'{MIIIT_ROOT}/../static/Standard_Peaklist.csv'))]
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
        self.run_button.on_click(self.run)
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.download)
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



        if DEVEL:
            self.rt_projections = rt_projections_test_data

    def run(self, b=None):
        try:
            results = []
            rt_projections = {}
            peaklist = self.peaklist.files
            for i in peaklist:
                self.message_box.value = check_peaklist(i)
            n_files = len(self.mzxml.files)
            processed_files = []
            with self.output:
                for i, filename in enumerate(self.mzxml.files):
                    processed_files.append(filename)
                    run_text = 'Processing: \n' + '\n'.join(processed_files[-20:])
                    self.message_box.value = run_text
                    self.progress.value = 100 * (i+1) / n_files
                    self.progress.description = f'{i+1}/{n_files}'
                    df = mzxml_to_pandas_df(filename)
                    result = integrate_peaks(df, peaklist)
                    result['mzxmlFile'] = os.path.basename(filename)
                    result['mzxmlPath'] = os.path.dirname(filename)
                    results.append(result)
                    rt_projection = peak_rt_projections(df, peaklist)
                    rt_projections[filename] = rt_projection
            self.results = pd.concat(results)
            self.rt_projections = restructure_rt_projections(rt_projections)
            if len(processed_files) == 1:
                self.message_box.value = self.results.to_string()
            else:
                self.message_box.value = self.results_crosstab().to_string()
            self.download(None)
            self.update_highlight_selector()
            self.update_peak_selector()
        except Exception as e:
            self.message_box.value = str(e)

    def update_peak_selector(self):
        new_values =  tuple(self.rt_projections.keys())
        self.plot_peak_selector.options = new_values

    def update_highlight_selector(self):
        new_values = tuple(list(self.rt_projections.values())[0].keys())
        self.plot_highlight_selector.options = new_values
        self.plot_file_selector.options = new_values


    @property
    def results(self):
        return self._results_df
    
    @results.setter 
    def results(self, df):
        self._results_df = df
    
    @property
    def rt_projections(self):
        return self._rt_projections
    
    @rt_projections.setter
    def rt_projections(self, data):
        self._rt_projections = data
        self.update_peak_selector()
        self.update_highlight_selector()


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
            filename = P('output') / P('{}-metabolomics_peak_intensity-{}.xlsx'\
                .format(now, uid))
            writer = pd.ExcelWriter(filename)
            self.results.to_excel(writer, 'MainTable', index=False)
            self.results_crosstab('peakArea')\
                .to_excel(writer, 'peakArea', index=True)
            writer.save()
            self.download_html.value = \
                """<a download='{}' href='{}'>Download</a>"""\
                .format(filename, filename)
    
    def results_crosstab(self, varName='peakArea'):
        '''
        Returns a comprehensive dataframe with the
        extraction results of one or more files.
        '''
        cols = ['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax']
        return pd.merge(self.results[cols].drop_duplicates(),
                        pd.crosstab(self.results.peakLabel, 
                                    self.results.mzxmlFile, 
                                    self.results[varName], 
                                    aggfunc=sum),
                        on='peakLabel')    
    
    def gui(self):
        return VBox([HBox([self.mzxml, 
                           self.peaklist, 
                           self.run_button, 
                           self.download_html]),
                    self.message_box,
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

    def display_output(self):
        display(self.output)

    def plot_button_on_click(self, button):
        self.plot()
    
    def plot(self):
        rt_proj_data = self.rt_projections
        peakLabels = self.plot_peak_selector.value
        files = self.plot_file_selector.value
        highlight = self.plot_highlight_selector.value
        n_cols = self.plot_ncol_slider.value
        if (len(peakLabels) == 0) or (len(files) == 0):
            return None
        plots = []
        hover_tool = HoverTool(tooltips=[('File', '$name')])
        tools = [hover_tool, 
                 WheelZoomTool(), 
                 PanTool(), 
                 ResetTool(), 
                 ZoomInTool(), 
                 ZoomOutTool(),
                 SaveTool()]
        with self.output:
            clear_output()
            for label in list(peakLabels):
                tmp_data = rt_proj_data[label]
                p = figure(title=f'Peak: {label}', x_axis_label='Retention Time', 
                           y_axis_label='Intensity', tools=tools)
                for file, rt_proj in tmp_data.items():
                        x = rt_proj.index
                        y = rt_proj.values
                        if not file in files:
                            continue
                        if file in highlight:
                            color = 'red'
                            legend = os.path.basename(file)
                        else:
                            color = 'blue'
                            legend = None
                        p.line(x, y, color=color, name=file, line_width=2, legend=legend)
                        if legend is not None:
                            p.legend.label_text_font_size = "8pt"
                p.legend.click_policy = "mute"


                plots.append(p)
            grid = gridplot(plots, ncols=n_cols, sizing_mode='stretch_both', plot_height=250)
            show(grid)

def integrate_peaks_from_filename(mzxml, peaklist=STANDARD_PEAKLIST):
    df = mzxml_to_pandas_df(mzxml)
    peaks = integrate_peaks(df)
    peaks['mzxmlFile'] = mzxml
    return peaks 

def integrate_peaks(df, peaklist=STANDARD_PEAKLIST):
    '''
    Takes the output of mzxml_to_pandas_df() and
    batch-calculates peak properties.
    '''
    peaklist = get_peaklistfrom(peaklist)
    peaklist.index = range(len(peaklist))
    results = []
    for peak in to_peaks(peaklist):
        result = integrate_peak(df, **peak)
        results.append(result)
    results = pd.concat(results)
    results.index = range(len(results))
    return pd.merge(peaklist, results, right_index=True, left_index=True)

def integrate_peak(df, mz, dmz, rtmin, rtmax, peaklabel):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calculates peak properties of one peak specified by
    the input arguements.
    '''
    slizE =slice_ms1_mzxml(df, rtmin=rtmin, rtmax=rtmax, mz=mz, dmz=dmz)
    peakArea = slizE['intensity array'].sum()
    result = pd.DataFrame({'peakLabel': peaklabel,
                           'rtmin': [rtmin], 
                           'rtmax': [rtmax],
                           'peakMz': [mz],
                           'peakMzWidth[ppm]': [dmz],
                           'peakArea': [peakArea]})
    return result[['peakArea']]

def peak_rt_projections(df, peaklist=STANDARD_PEAKLIST):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    batch-calcualtes the projections of peaks onto
    the RT dimension to visualize peak shapes.
    '''
    peaklist = get_peaklistfrom(peaklist)
    peaklist.index = range(len(peaklist))
    results = []
    for peak in to_peaks(peaklist):
        result = peak_rt_projection(df, **peak)
        results.append(result)
    return results

def peak_rt_projection(df, mz, dmz, rtmin, rtmax, peaklabel):
    '''
    Takes the output of mzxml_to_pandas_df() and 
    calcualtes the projections of one peak, 
    specicied by the input parameters, onto
    the RT dimension to visualize peak shapes.
    '''
    slizE = slice_ms1_mzxml(df, rtmin=rtmin, rtmax=rtmax, mz=mz, dmz=dmz)
    rt_projection = slizE[['retentionTime', 'm/z array', 'intensity array']]\
                    .set_index(['retentionTime', 'm/z array'])\
                    .unstack()\
                    .sum(axis=1)
    return [mz, dmz, rtmin, rtmax, peaklabel, rt_projection]

def get_peaklistfrom(filenames):
    '''
    Extracts peak data from csv file.
    '''
    if isinstance(filenames, str):
        filenames = [filenames]
    peaklist = []
    cols_to_import = ['peakLabel',
                      'peakMz',
                      'peakMzWidth[ppm]',
                      'rtmin',
                      'rtmax']
    for file in filenames:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, usecols=cols_to_import,
                             dtype={'peakLabel': str})
            df['peakListFile'] = file
            peaklist.append(df)
    return pd.concat(peaklist)


def to_peaks(peaklist):
    '''
    Takes a dataframe with at least the columns:
    ['peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax', 'peakLabel'].
    Returns a list of dictionaries that define peaks.
    '''
    cols_to_import = ['peakMz', 
                      'peakMzWidth[ppm]',
                      'rtmin', 
                      'rtmax', 
                      'peakLabel']
    tmp = [list(i) for i in list(peaklist[cols_to_import].values)]
    output = [{'mz': el[0],
               'dmz': el[1], 
               'rtmin': el[2],
               'rtmax': el[3], 
               'peaklabel': el[4]} for el in tmp]
    return output

def mzxml_to_pandas_df(filename):
    '''
    Reads mzXML file and returns a pandas.DataFrame.
    '''
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
    '''
    Converts dataframe to numeric types if possible.
    '''
    for col in df.columns:
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

def restructure_rt_projections(data):
    output = {}
    for el in list(data.values())[0]:
        output[el[4]] = {}
    for filename in data.keys():
        for item in data[filename]:
            peaklabel = item[4]
            rt_proj = item[5]
            output[peaklabel][filename] = rt_proj
    return output
