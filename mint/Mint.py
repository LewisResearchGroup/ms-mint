import os
import uuid
import datetime
import itertools
import pandas as pd
import time

import ipywidgets as widgets    
from ipywidgets import Button, HBox, VBox, Textarea, HTML,\
    SelectMultiple, Layout, Label, IntSlider
from ipywidgets import IntProgress as Progress
from IPython.display import display, clear_output

from pathlib import Path as P

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool, PanTool, ResetTool,\
    WheelZoomTool, ZoomInTool, ZoomOutTool, SaveTool
from bokeh.layouts import gridplot
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import DataTable
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource

from bokeh.layouts import row
from bokeh.io import show, output_notebook
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application

from .SelectFilesButton import SelectFilesButton
from .SelectFolderButton import SelectFolderButton

from .tools import integrate_peaks, peak_rt_projections,\
    mzxml_to_pandas_df, check_peaklist, STANDARD_PEAKLIST,\
    restructure_rt_projections, STANDARD_PEAKFILE,\
    read_peaklists

import warnings

from multiprocessing import Process, Pool, Manager, cpu_count
from glob import glob

class Mint():
    def __init__(self, port=None):
        output_notebook(hide_banner=True)
        self.mzxml = SelectFilesButton(text='Select mzXML', callback=self.list_files)
        self.mzxml_folder = SelectFolderButton(text='Select Directory', callback=self.list_files_from_folder)
        self.peaklist = SelectFilesButton(text='Peaklist', 
            default_color='lightgreen', callback=self.list_files)
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run)    
        self.stop_button = Button(description="STOP")
        self.stop_button.on_click(self.stop)   
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.download)
        self.port = port
        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '99%', 
                    'height': '400px', 
                    'font_family': 'monospace'})

        self.list_button = Button(description="List Files")
        self.list_button.on_click(self.list_files)
        self.report_issue = HTML("""<a href="https://github.com/LSARP/mint/issues" target="_blank">Report an issue</a>""")
        self.progress = Progress(min=0, max=100, layout=Layout(width='99%'))
        self.download_html = HTML("""Nothing to download""")
        self.output = widgets.Output(layout=Layout(width='99%'))
        self.output_plotting = widgets.Output(layout=Layout(width='99%'))
        # Plotting elements
        self.button_show_table = Button(description="Show Results Table")
        self.button_show_table.on_click(self.show_table)
        self.button_show_plots = Button(description="Plot Peaks")
        self.button_show_plots.on_click(self.button_show_plots_on_click)
        self.plot_peak_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='200px', Label='test'))
        self.plot_file_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='200px'))
        self.plot_highlight_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='200px'))
        self.plot_ncol_slider = IntSlider(min=1, max=5, step=1, value=3)
        self.plot_legend_font_size = IntSlider(min=1, max=20, step=1, value=6)
        warnings.filterwarnings('ignore')
        # default values
        self.peaklist.files = [STANDARD_PEAKFILE]
        self._results_df = None
        self.peaks = []
        self._rt_projections = None
        self.peakLabels = []
        self._stop = False

    def stop(self, b=None):
        self._stop = True

    def run(self, b=None, nthreads=None):
            self._stop = False
            with self.output:
                if nthreads is None:
                    nthreads = cpu_count()
                results = []
                rt_projections = {}
                for i in self.peaklist.files:
                    self.message_box.value = check_peaklist(i)
                peaklist = read_peaklists(self.peaklist.files)
                n_files = len(self.mzxml.files)
                processed_files = []
                args = []

                self.message_box.value += f'\n\nUsing {nthreads} cores.'
                pool = Pool(processes=nthreads)
                m = Manager()
                q = m.Queue()
                
                for i, filename in enumerate(self.mzxml.files):
                    args.append({'filename': filename,
                                 'peaklist': peaklist,
                                 'q':q})
                results = pool.map_async(process, args)

                # monitor progress
                while True:
                    if results.ready():
                        break
                    elif self._stop:
                        pool.terminate()
                        return None
                    else:
                        size = q.qsize()
                        self.progress.value = 100 * size / n_files
                        self.progress.description = f'{size}/{n_files}'
                        time.sleep(1)

                pool.close()
                pool.join()

                results = results.get()
                self.results = pd.concat([i[0] for i in results])[[
                    'peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax',
                    'peakArea', 'mzxmlFile', 'mzxmlPath', 'peakListFile']]
                [rt_projections.update(i[1]) for i in results]
                self.rt_projections = restructure_rt_projections(rt_projections)
                self.message_box.value = 'Done'
                self.download(None)
                self.update_highlight_selector()
                self.update_peak_selector()
                self.show_table()

    def update_peak_selector(self):
        new_values =  tuple(self.rt_projections.keys())
        self.plot_peak_selector.options = new_values

    def update_highlight_selector(self):
        new_values = tuple(list(self.rt_projections.values())[0].keys())
        self.plot_highlight_selector.options = new_values
        self.plot_file_selector.options = new_values

    def update_file_selector(self):
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
        self.mzxml.files = [i for i in self.mzxml.files if i.endswith('.mzXML')]
        text = '{} mzXML files to process:\n'.format(len(self.mzxml.files))

        for line in self.mzxml.files:
            text += line+'\n'
        text += '\n\nUsing peak list:\n'
        if len(self.peaklist.files) != 0:
            text += '\n'.join([str(i) for i in self.peaklist.files])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
        if len(self.mzxml.files) > 0:
            self.mzxml.style.button_color = 'lightgreen'
        else: 
            self.mzxml.style.button_color = 'red'
            self.mzxml_folder.files = []
            self.mzxml_folder.style.button_color = 'red'

    def list_files_from_folder(self, b=None):
        try:
            pattern = self.mzxml_folder.files[0]+'/**/*.mzXML'
            self.mzxml.files = glob(pattern, recursive=True)
            self.list_files()
        except:
            pass

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
                           self.mzxml_folder,
                           self.peaklist, 
                           self.run_button, 
                           self.stop_button,
                           self.download_html]),
                    self.message_box,
                    self.progress,
                    HBox([self.button_show_table, self.report_issue]),
                    ])

    def gui_plotting(self):
        gui = VBox([HBox([Label('Peak', layout=Layout(width='33%')),
                          Label('File', layout=Layout(width='33%')), 
                          Label('Highlight', layout=Layout(width='33%'))]),
                    HBox([self.plot_peak_selector,
                          self.plot_file_selector,
                          self.plot_highlight_selector]),
                    HBox([Label('N columns'), self.plot_ncol_slider, 
                          Label('Legend fontsize'), self.plot_legend_font_size]),
                    HBox([self.button_show_plots])
                    ])
        return gui

    def display_output(self):
        display(self.output)

    def display_plots(self):
        display(self.output_plotting)

    def button_show_plots_on_click(self, b=None):
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
                 SaveTool(),
                 'tap']
        with self.output_plotting:
            clear_output()
            for label in list(peakLabels):
                tmp_data = rt_proj_data[label]
                p = figure(title=f'Peak: {label}', x_axis_label='Retention Time', 
                           y_axis_label='Intensity', tools=tools)
                colors = itertools.cycle(palette)
                for high_ in [False, True]:
                    for file, rt_proj in tmp_data.items():
                            x = rt_proj.index
                            y = rt_proj.values
                            if not file in files:
                                continue
                            if file in highlight:
                                color = next(colors)
                                legend = os.path.basename(file)
                                alpha = 1   
                                if not high_:
                                    continue
                            else:
                                color = 'blue'
                                legend = None
                                alpha = 0.5
                                if high_:
                                    continue
                            p.line(x, y,
                                name=file, 
                                line_width=2, 
                                legend=legend,
                                selection_color="firebrick",
                                color=color, alpha=alpha)
                            if legend is not None:
                                p.legend.label_text_font_size = "{}pt".format(self.plot_legend_font_size.value)
                p.legend.click_policy = "mute"
                plots.append(p)
            grid = gridplot(plots, ncols=n_cols, sizing_mode='stretch_both', plot_height=250)
            show(grid)

    def show_table(self, b=None):
        if self.results is None:
            return None

        df = self.results.astype(str)
        columns = [TableColumn(field=col, title=col) for col in df.columns]
        source = ColumnDataSource(df)

        def callback(attrname, old, new):
            selectionIndex = source.selected.indices
            peakLabels = tuple(self.results.iloc[selectionIndex]['peakLabel'].drop_duplicates())
            mzxmlFiles = tuple(self.results.iloc[selectionIndex]['mzxmlFile'].drop_duplicates())
            self.plot_peak_selector.value = peakLabels
            self.plot_file_selector.value = mzxmlFiles

        source.selected.on_change('indices', callback)

        data_table = DataTable(columns=columns, 
                               source=source, 
                               sizing_mode='stretch_both')

        # Create the Document Application
        def modify_doc(doc):
            layout = row(data_table)
            doc.add_root(layout)      

        handler = FunctionHandler(modify_doc)
        app = Application(handler)                       

        with self.output:
            clear_output()
            if self.port is not None:
                show(app, notebook_url=f'localhost:{self.port}')
            else:
                show(app)

def process(args):
    '''Pickleable function for parallel processing.'''
    filename = args['filename']
    peaklist = args['peaklist']
    q = args['q']
    q.put('filename')
    df = mzxml_to_pandas_df(filename)
    result = integrate_peaks(df, peaklist)
    result['mzxmlFile'] = filename
    result['mzxmlPath'] = os.path.dirname(filename)
    rt_projection = {filename: peak_rt_projections(df, peaklist)}
    return result, rt_projection
