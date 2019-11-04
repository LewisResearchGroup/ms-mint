import os
import uuid
import datetime
import itertools
import pandas as pd
import time

from os.path import abspath

import ipywidgets as widgets    
from ipywidgets import Button, HBox, VBox, Textarea, HTML,\
    SelectMultiple, Layout, Label, IntSlider, ButtonStyle
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
import mint

from datetime import date

import plotly.graph_objects as go
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
from os.path import basename

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

import plotly.express as px

from .tools import process_in_parallel

class Mint():
    def __init__(self, port=None):
        '''The port needs to be the same as the port of the Jupyter notebook 
        that runs Mint.'''
        output_notebook(hide_banner=True)
        self._b_color_default = 'lightgreen'
        self._b_color_warning = 'red'
        self._b_color_not_ready = 'orange'

        self.B_add_mzxmls = SelectFilesButton(
            text='Select mzXML', callback=self.add_mzxmls)
        self.B_add_mzxmls_dir = SelectFolderButton(
            text='mzXML Directory', callback=self.add_mzxmls)
        self.B_peaklist = SelectFilesButton(
            text='Peaklist', default_color=self._b_color_default, callback=self.update_status)
        self.B_export = SelectFolderButton(text='Export', callback=self.export)
        self.B_run = Button(description="Run")
        self.B_run.on_click(self.run) 
        self.B_run.style.button_color = self._b_color_not_ready
        self.port = port
        self.message_box = Textarea(
            value='',
            placeholder='Please select some files and click on Run.',
            description='',
            disabled=True,
            layout={'width': '99%', 
                    'height': '400px', 
                    'font_family': 'monospace'})
                
        self.regex_filter = widgets.Text(
            value='*.*',
            placeholder='Type regular expression e.g. *.*',
            description='File Filter:',
            disabled=False
        )

        self.B_list = Button(description="List Files")
        self.B_list.on_click(self.update_status)
        self.report_issue = HTML("""<a href="https://github.com/LSARP/mint/issues" target="_blank">Report an issue</a>""")
        self.progress = Progress(min=0, max=100, layout=Layout(width='99%'))
        self.output = widgets.Output(layout=Layout(width='99%'))
        self.output_plotting = widgets.Output(layout=Layout(width='99%'))

        # Plotting elements
        self.B_show_table = Button(description="Show Results Table")
        self.B_show_table.on_click(self.show_table)
        self.B_show_plots = Button(description="Plot Peaks")
        self.B_show_plots.on_click(self.show_plots)
        self.B_show_plots.style.button_color = self._b_color_default
        self.B_show_heatmap = Button(description="Heatmap")
        self.B_show_heatmap.on_click(self.show_heatmap)
        self.B_show_heatmap.style.button_color = self._b_color_default
        self.B_show_3dproj = Button(description="3D Peaks")
        self.B_show_3dproj.on_click(self.show_3dproj)
        self.B_show_3dproj.style.button_color = self._b_color_default
        self.B_clear = Button(description="Clear Files")
        self.B_clear.on_click(self.clear_files)
        self.B_clear.style.button_color = 'red'
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
        self.B_peaklist.files = [STANDARD_PEAKFILE]
        self.B_add_mzxmls_dir.files = []
        self._results_df = None
        self.peaks = []
        self._mzxmls = []
        self._rt_projections = None
        self.peakLabels = []
        self._stop = False
        self._raw_data = None
        
        # Extra features
        self.store_raw_data = False


    def stop(self, b=None):
        '''Not yet implemented.'''
        self._stop = True


    ### Input files
    
    @property
    def mzxmls(self):
        return self._mzxmls

    @mzxmls.setter
    def mzxmls(self, a_list):
        self._mzxmls = a_list

    def add_mzxmls(self):
        new_files = self.B_add_mzxmls.files + self.mzxmls_from_dir()
        new_files = [i for i in new_files if i.lower().endswith('.mzxml')]
        new_files = [i for i in new_files if not i in self.mzxmls]
        self._mzxmls.extend(new_files)
        self.update_status()
        self.B_add_mzxmls.files = []
        self.B_add_mzxmls_dir.files = []
   
    @property
    def peaklists(self):
        return self.B_peaklist.files
    
    @peaklists.setter
    def peaklists(self, a_list):
        self.B_peaklist.files = a_list
   
    def clear_files(self, b=None):
        self.mzxmls = []
        self.update_status()
        
        
    ### GUI
   
    def gui(self):
        return VBox([HBox([self.B_add_mzxmls,
                           self.B_add_mzxmls_dir,
                           self.B_peaklist,
                           self.B_clear,
                           self.report_issue]),
                    self.regex_filter,
                    self.message_box,
                    HBox([self.B_run, self.B_export, self.progress]),
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
                    HBox([self.B_show_plots, self.B_show_heatmap, self.B_show_3dproj])
                    ])
        return gui

    ### Run procedures
        
    def run(self, b=None, nthreads=None):
        '''Main procedure for data extraction.'''
        self._stop = False
        with self.output:
            if nthreads is None:
                nthreads = cpu_count()
            peaklists = self.peaklists
            for i in peaklists:
                self.message_box.value = check_peaklist(i)
            peaklist = read_peaklists(peaklists)
            mzxmls = self.mzxmls
            n_files = len(mzxmls)
            processed_files = []
            args = []

            self.message_box.value += f'\n\nUsing {nthreads} cores.'

            pool = Pool(processes=nthreads)
            m = Manager()
            q = m.Queue()
            
            for i, filename in enumerate(mzxmls):
                args.append({'filename': filename,
                                'peaklist': peaklist,
                                'q':q})

            results = pool.map_async(process_in_parallel, args)

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
            self.results = pd.concat([i[0] for i in results])
            if self.store_raw_data:
                self.raw_data = pd.concat([i[2] for i in results])
            # [['peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax',
            #   'peakArea', 'mzxmlFile', 'mzxmlPath', 'peakListFile']]
            rt_projections = {}
            [rt_projections.update(i[1]) for i in results]
            self.rt_projections = restructure_rt_projections(rt_projections)
            self.message_box.value = 'Done'
            self.update_highlight_selector()
            self.update_peak_selector()
            self.peakLabels = list(self.rt_projections.keys())
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

    def update_status(self, b=None):
        mzxmls = self.mzxmls
        text = '{} mzXML files to process:\n'.format(len(mzxmls))
        for line in self.mzxmls:
            text += line+'\n'
        text += '\n\nUsing peak list:\n'
        if len(self.peaklists) != 0:
            text += '\n'.join([str(i) for i in self.peaklists])
        else:
            text += '\nNo peaklist defined.'
        self.message_box.value = text
        if len(self.mzxmls) > 0:
            self.B_add_mzxmls.style.button_color = self._b_color_default
            self.B_add_mzxmls_dir.style.button_color = self._b_color_default
        else: 
            self.B_add_mzxmls.style.button_color = self._b_color_not_ready
            self.B_add_mzxmls_dir.style.button_color = self._b_color_not_ready
        if len(self.peaklists) > 0 and len(self.mzxmls) > 0:
            self.B_run.style.button_color = self._b_color_default
        else:
            self.B_run.style.button_color = self._b_color_not_ready

    def mzxmls_from_dir(self):
        content = self.B_add_mzxmls_dir.files
        if len(content) == 0 or len(content[0]) == 0:
            return []
        pattern = self.B_add_mzxmls_dir.files[0]+f'/**/{self.regex_filter.value}'
        return glob(pattern, recursive=True)

    def export(self):
        if self.results is None:
            return None
        uid = str(uuid.uuid4()).split('-')[-1]
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        folder = self.B_export.files[0]
        if isinstance(folder, tuple):
            return None
        
        filename = P(folder) / P('Mint-{}-{}.xlsx'\
                .format(now, uid))

        writer = pd.ExcelWriter(filename)
        self.results.to_excel(writer, 'MainTable', index=False)
        self.results_crosstab.T('peakArea')\
            .to_excel(writer, 'peakArea', index=True)
        meta = pd.DataFrame({'Version': [mint.__version__], 
                                'Date': [str(date.today())]}).T[0]
        meta.to_excel(writer, 'MetaData', index=True, header=False)
        
        try:
            writer.save()
            self.message_box.value = f'Exported results to {filename}!'
            self.B_export.style.button_color = self._b_color_default
        except PermissionError:
            self.message_box.value = 'Cannot write file!'
            self.B_export.style.button_color = self._b_color_warning

    def results_crosstab(self, varName='peakArea'):
        '''
        Returns a comprehensive dataframe with the
        extraction results of one or more files.
        '''
        cols = ['peakLabel']
        return pd.merge(self.results[cols].drop_duplicates(),
                        pd.crosstab(self.results.peakLabel, 
                                    self.results.mzxmlFile, 
                                    self.results[varName], 
                                    aggfunc=sum),
                        on='peakLabel').set_index('peakLabel').T
    
    def display_output(self):
        display(self.output)

    def display_plots(self):
        display(self.output_plotting)

    def show_plots(self, b=None):
        if self.results is None:
            return None
        self.plot()

    def show_heatmap(self, b=None):
        if self.results is None:
            return None
        with self.output_plotting:
            clear_output()
            height = widgets.IntSlider(
                min=500, max=5000, step=50, value=10, 
                description='Figure height', continuous_update=False)
            interact(self.get_heatmap, 
                color = ['linear', 'log'],
                height=height,
                normalize_columns=False, 
                show_abs_path=False,
                clustering=False, 
                show_dendrogram=False,
                transpose=False)

    def get_heatmap(self, 
            color='linear', 
            height=500,
            show_abs_path=False, 
            normalize_columns=False, 
            clustering=True, 
            show_dendrogram=False,
            transpose=False):

        assert color in ['linear',  'log']
        data = self.results_crosstab()

        if normalize_columns:
            data = (data / data.max()).fillna(0)

        if not show_abs_path:
            data.index = [basename(i) for i in data.index]

        if clustering:
            D = squareform(pdist(data, metric='euclidean'))
            Y = linkage(D, method='complete')
            Z = dendrogram(Y, orientation='left', no_plot=(not show_dendrogram))['leaves']
            data = data.iloc[Z,:]

        if color == 'log':
            data = data.apply(np.log1p) 

        if transpose:
            data = data.T
            
        fig = go.Figure(
            data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index),
            )
        fig.update_layout(height=height, 
            yaxis={'title': 'mzXML', 'tickmode': 'array', 'automargin': True})
        fig.show()

    def show_3dproj(self, b=None):
        if self.results is None:
            return None
        with self.output_plotting:
            clear_output()
            interact(self.get_rt_3d_plots, peakLabel=self.peakLabels)

    def get_rt_3d_plots(self, peakLabel):
        data = self.rt_projections[peakLabel]
        samples = []
        for i, key in enumerate(list(data.keys())):
            sample = data[key].to_frame().reset_index()
            sample.columns = ['retentionTime', 'intensity']
            sample['peakArea'] = sample.intensity.sum()
            sample['Filename'] = os.path.basename(key)
            samples.append(sample)
        samples = pd.concat(samples)
        fig = px.line_3d(samples, x='retentionTime', y='peakArea' ,z='intensity', color='Filename')
        return fig

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
            jobs = []
            plots = []
            for label in list(peakLabels):  
                args = {'data': rt_proj_data[label],
                        'label': label,
                        'tools': tools,
                        'colors': itertools.cycle(palette),
                        'files': files,
                        'highlight': highlight,
                        'legend_font_size': self.plot_legend_font_size.value}
                plots.append(plot_peaks(args))
            grid = gridplot(plots, ncols=n_cols, 
                            sizing_mode='stretch_both', 
                            plot_height=250)
            show(grid)

    def show_table(self, b=None):
        if self.results is None:
            return None
        cols = ['mzxmlFile', 'peakLabel', 'peakMz', 'peakMzWidth[ppm]', 'rtmin', 'rtmax', 'peakArea'] 
        df = self.results[cols].astype(str)
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

def plot_peaks(args):
    data = args['data']
    label = args['label']
    colors = args['colors']
    tools = args['tools']
    files = args['files']
    highlight = args['highlight']
    legend_font_size = args['legend_font_size']

    p = figure(title=f'Peak: {label}', 
            x_axis_label='Retention Time', 
            y_axis_label='Intensity',
            tools=tools)
    
    for high_ in [False, True]:
        for file, rt_proj in data.items():
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
                    txt = "{}pt".format(legend_font_size)
                    p.legend.label_text_font_size = txt
                    
    p.legend.click_policy = "mute"
    return p