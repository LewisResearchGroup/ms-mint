import os
import uuid
import datetime
import itertools
import pandas as pd

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

from .SelectFilesButton import SelectFilesButton

from .tools import integrate_peaks, peak_rt_projections,\
    mzxml_to_pandas_df, check_peaklist,\
    restructure_rt_projections
import warnings

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

class Mint():
    def __init__(self):
        output_notebook(hide_banner=True)
        self.mzxml = SelectFilesButton(text='Select mzXML', callback=self.list_files)
        self.peaklist = SelectFilesButton(text='Peaklist', 
            default_color='lightgreen', callback=self.list_files)
        self.peaklist.files = [P(os.path.abspath(
            f'{MIIIT_ROOT}/../static/Standard_Peaklist.csv'))]
        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '99%', 
                    'height': '200px', 
                    'font_family': 'monospace'})
        self.list_button = Button(description="List Files")
        self.list_button.on_click(self.list_files)
        self.run_button = Button(description="Run")
        self.run_button.on_click(self.run)
        self.download_button = Button(description="Download")
        self.download_button.on_click(self.download)
        self._results_df = None
        self.progress = Progress(min=0, max=100, layout=Layout(width='99%'))
        self.peaks = []
        self._rt_projections = None

        self.download_html = HTML("""Nothing to download""")
        self.output = widgets.Output(layout=Layout(width='99%'))
        self.peakLabels = []
        # Plotting stuff
        self.plot_button_results = Button(description="Show Results Table")
        self.plot_button_results.on_click(self.show_results)
        self.plot_button = Button(description="Plot Peaks")
        self.plot_button.on_click(self.plot_button_on_click)
        self.plot_peak_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='90px', Label='test'))
        self.plot_file_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='90px'))
        self.plot_highlight_selector = SelectMultiple(
            options=[], layout=Layout(width='33%', height='90px'))
        self.plot_ncol_slider = IntSlider(min=1, max=5, step=1, value=3)
        self.plot_legend_font_size = IntSlider(min=1, max=20, step=1, value=6)
        warnings.filterwarnings('ignore')

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
            self.results = pd.concat(results)[[
                'peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax',
                'peakArea', 'mzxmlFile', 'mzxmlPath', 'peakListFile']]
            self.rt_projections = restructure_rt_projections(rt_projections)
            if len(processed_files) == 1:
                self.message_box.value = 'Done'
            self.download(None)
            self.update_highlight_selector()
            self.update_peak_selector()
            self.show_results()
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
                    HBox([self.plot_button_results, self.plot_button]),
                    HBox([Label('N columns'), self.plot_ncol_slider, 
                          Label('Legend fontsize'), self.plot_legend_font_size])
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
                 SaveTool(),
                 'tap']
        with self.output:
            clear_output()
            for label in list(peakLabels):
                tmp_data = rt_proj_data[label]
                p = figure(title=f'Peak: {label}', x_axis_label='Retention Time', 
                           y_axis_label='Intensity', tools=tools)
                colors = itertools.cycle(palette)    
                for file, rt_proj in tmp_data.items():
                        x = rt_proj.index
                        y = rt_proj.values
                        if not file in files:
                            continue
                        if file in highlight:
                            color = next(colors)
                            legend = os.path.basename(file)
                        else:
                            color = 'blue'
                            legend = None
                        p.line(x, y,
                               name=file, 
                               line_width=2, 
                               legend=legend,
                               selection_color="firebrick",
                               color=color)
                        if legend is not None:
                            p.legend.label_text_font_size = "{}pt".format(self.plot_legend_font_size.value)
                p.legend.click_policy = "mute"
                plots.append(p)
            grid = gridplot(plots, ncols=n_cols, sizing_mode='stretch_both', plot_height=250)
            show(grid)
            self.show_results() 

    def show_results(self, b=None):
        if self.results is None:
            return None
        df = self.results.astype(str)
        columns = [TableColumn(field=col, title=col) for col in df.columns]
        data_table = DataTable(columns=columns, 
                               source=ColumnDataSource(df), 
                               sizing_mode='stretch_both')
        with self.output:
            if b is not None:
                clear_output()
            show(data_table)