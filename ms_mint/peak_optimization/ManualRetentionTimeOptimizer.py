import numpy as np
import pandas as pd
import ipywidgets as W
import plotly.express as px

from tqdm import tqdm
from IPython.display import display


from .io import ms_file_to_df


class ManualRetentionTimeOptimizer:
    def __init__(self, mint):

        self.df = pd.concat(
            [ms_file_to_df(fn).assign(ms_file=fn) for fn in tqdm(mint.ms_files)]
        )

        self.out = W.Output()

        self.mint = mint

        self.w_rt_min = W.FloatText(value=0, description="RT min:", disabled=False)

        self.w_rt_max = W.FloatText(
            value=13,
            description="RT max:",
            disabled=False,
        )

        self.set_rt_button = W.Button(description="Set new RT")
        self.delete_button = W.Button(description="Remove from peaklist")

        self.menu = W.Dropdown(options=mint.peaklist.peak_label, value=None)

        def update(*args):
            peak_label = self.menu.value
            self.plot(peak_label)

        def update_rt(button):
            rt_min, rt_max = (
                self.w_rt_min.value,
                self.w_rt_max.value,
            )
            peak_label = self.menu.value
            self.mint.peaklist.loc[
                self.mint.peaklist.peak_label == peak_label, "rt_min"
            ] = rt_min
            self.mint.peaklist.loc[
                self.mint.peaklist.peak_label == peak_label, "rt_max"
            ] = rt_max
            self.plot(peak_label)

        def remove_peak(button):
            peak_label = self.menu.value
            mint.peaklist = mint.peaklist[mint.peaklist.peak_label != peak_label]
            new_options = mint.peaklist.peak_label
            self.menu.options = new_options

        self.menu.observe(update, names="value")
        self.set_rt_button.on_click(update_rt)
        self.delete_button.on_click(remove_peak)

        self.layout = W.VBox(
            [
                self.menu,
                self.w_rt_min,
                self.w_rt_max,
                self.set_rt_button,
                self.out,
                self.delete_button,
            ],
        )

    def plot(self, peak_label):
        peak_data = self.mint.peaklist[
            self.mint.peaklist.peak_label == peak_label
        ].T.iloc[:, 0]
        mz_mean, mz_width, rt_min, rt_max = peak_data[
            ["mz_mean", "mz_width", "rt_min", "rt_max"]
        ]
        dmz = mz_mean * 1e-6 * mz_width
        selection = self.df[np.abs(self.df["m/z array"] - mz_mean) <= dmz]
        fig = px.line(
            data_frame=selection,
            x="retentionTime",
            y="intensity array",
            color="ms_file",
            title=peak_label,
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(hovermode="closest", xaxis=dict(range=[rt_min, rt_max]))

        self.out.clear_output()
        with self.out:
            display(fig)

        self.w_rt_min.value, self.w_rt_max.value = rt_min, rt_max

    def show(self):
        return self.layout
