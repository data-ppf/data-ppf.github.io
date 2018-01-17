import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib.ticker import LinearLocator

from IPython.display import display
from ipywidgets import HBox, IntSlider, Label, Layout, ToggleButton, VBox

from labs_commons import *

MU = 0
SIGMA = 1
N_OBSERVATIONS = 100


class AnalyticGaussianExplorer(BaseExplorer):
    """
    Explorer for the analytic Gaussian curve
    """

    X_LIMITS = (-4, 4)

    def __init__(self):
        self.xs = np.linspace(*self.X_LIMITS, N_OBSERVATIONS + 1)
        self.ys = norm.pdf(self.xs, loc=MU, scale=SIGMA)
        super().__init__()

    def _this_init(self):
        self._plot()

    def _plot(self):
        self.axes['default'] = self.fig.add_subplot(111)
        self.axes['default'].plot(self.xs, self.ys)
        self.axes['default'].set_xlabel("X")
        self.axes['default'].set_ylabel("Probability Density")
        self.axes['default'].set_title("The Normal Distribution")


class NumericalGaussianExplorer(BaseExplorer):

    def __init__(self):
        self.sample = norm.rvs(
            size=N_OBSERVATIONS, loc=MU, scale=SIGMA
        )
        self.args_sorted = np.argsort(self.sample)

    def _this_init(self):
        self.widgets["sorted_index"] = IntSlider(
            value=0, min=0, max=N_OBSERVATIONS, readout=False
        )
        self.widgets["sorted_index"].observe(self.resort)
        self.control_panel = HBox([
            Label("Scrambled"),
            self.widgets["sorted_index"],
            Label("Sorted")
        ], layout=self.CENTER_LAYOUT)
        self._plot()

    def display_sample(self):
        print(self.sample)

    def display_statistics(self):
        display(pd.DataFrame(
            self.sample, columns=["gaussian_sample"]
        ).describe())

    def display_graph(self):
        super().__init__()

    def _plot(self):
        self.axes["default"] = self.fig.add_subplot(111)
        self.bars = self.axes["default"].bar(
            np.arange(N_OBSERVATIONS), self.sample
        )
        self.axes['default'].set_xticks([])
        self.axes['default'].set_yticks([])
        self.axes['default'].set_xlabel("Observation")
        self.axes['default'].set_ylabel("Value")
        self.axes["default"].set_title('Samples from a Gaussian')

    def resort(self, change):
        threshold = self.widgets["sorted_index"].value
        sorted_observations = self.sample[
            self.args_sorted[self.args_sorted < threshold]
        ]
        for bar, value in zip(self.bars[:threshold], sorted_observations):
            bar.set_height(value)
        for bar, value in zip(self.bars[threshold:], self.sample[threshold:]):
            bar.set_height(value)
        for bar in self.bars:
            bar.set_y(0)


class GaussianExplorer(BaseExplorer):

    SAMPLE_SIZE = {"min": int(1e2), "max": int(1e3), "step": int(1e2)}
    N_BINS = {"min": int(1e1), "max": int(1e2), "step": int(1e1)}

    def __init__(self):
        self.base_sample = norm.rvs(
            size=self.SAMPLE_SIZE["max"],
            loc=MU,
            scale=SIGMA
        )
        x_max_abs = np.abs(self.base_sample).max()
        self.parent_xs = np.linspace(
            -x_max_abs,
            x_max_abs,
            N_OBSERVATIONS
        )
        self.parent_ys = norm.pdf(
            self.parent_xs,
            loc=MU,
            scale=SIGMA
        )
        super().__init__()

    def _this_init(self):
        self.widgets["mean"] = ToggleButton(description='Show Mean')
        self.widgets["mean"].observe(self.toggle_mean)
        self.widgets["median"] = ToggleButton(description='Show Median')
        self.widgets["median"].observe(self.toggle_median)
        self.widgets["parent"] = ToggleButton(
            description="Show Parent Gaussian Distribution",
            layout=Layout(width='300px')
        )
        self.widgets["parent"].observe(self.toggle_parent)
        self.widgets["sample_size"] = IntSlider(
            min=self.SAMPLE_SIZE["min"],
            max=self.SAMPLE_SIZE["max"],
            step=self.SAMPLE_SIZE["step"],
            value=self.SAMPLE_SIZE["min"],
            description="Sample Size"
        )
        self.widgets["sample_size"].observe(self.resample)
        self.widgets["bins"] = IntSlider(
            min=self.N_BINS["min"],
            max=self.N_BINS["max"],
            step=self.N_BINS["step"],
            value=self.N_BINS["min"],
            description="Number of Bins",
            style={'description_width': 'initial'}
        )
        self.widgets["bins"].observe(self.redraw_histogram)
        self.control_panel = VBox([
            HBox([
                self.widgets["mean"],
                self.widgets["median"],
                self.widgets["parent"]
            ], layout=self.CENTER_LAYOUT),
            HBox([
                self.widgets["sample_size"],
                self.widgets["bins"]
            ], layout=self.CENTER_LAYOUT)
        ])
        self._plot()
        self.resample()
        self.toggle_parent()

    def resample(self, change=None):
        self.sample = np.random.choice(
            self.base_sample,
            size=self.widgets["sample_size"].value,
            replace=False
        )
        self.redraw_histogram()
        self.toggle_mean()
        self.toggle_median()

    def _plot(self):
        self.axes["histogram"] = self.fig.add_subplot(111)
        self.axes["histogram"].set_ylabel("Count")
        self.axes["histogram"].set_xlabel("Value")
        self.axes["histogram"].set_title(
            "The Normal Distribution: Sample vs. Parent"
        )
        self.axes["parent"] = self.axes["histogram"].twinx()
        self.axes["parent"].set_ylabel("Probability Density")
        self.axes["parent"].fill_between(
            self.parent_xs,
            0,
            self.parent_ys,
            color=self.DEFAULT_COLORS[3],
            alpha=0.1,
            label="Parent Gaussian Distribution"
        )
        self.axes["parent"].set_ylim(ymin=0)
        self.axes["parent"].legend(loc=1)
        self.axes["parent"].grid(None)
        self.axes["parent"].yaxis.set_major_locator(
            LinearLocator(self.STANDARD_TICKS)
        )
        self.axes["parent"].yaxis.set_major_formatter(
            self.STANDARD_FLOAT_TICK
        )

    def redraw_histogram(self, change=None):
        if hasattr(self, "histogram"):
            [bar.remove() for bar in self.histogram]
            del self.histogram
        _, _, self.histogram = self.axes["histogram"].hist(
            self.sample,
            bins=self.widgets["bins"].value,
            color=self.DEFAULT_COLORS[0],
            label="histogram"
        )
        self.axes["histogram"].relim()
        self.axes["histogram"].autoscale()
        self.axes["histogram"].legend(loc=2)
        self.axes["histogram"].yaxis.set_major_locator(
            LinearLocator(self.STANDARD_TICKS)
        )
        self.axes["histogram"].yaxis.set_major_formatter(
            self.STANDARD_FLOAT_TICK
        )

    def toggle_mean(self, change=None):
        if hasattr(self, "mean_line"):
            self.mean_line.remove()
            del self.mean_line
        if self.widgets["mean"].value:
            self.mean_line = self.axes["histogram"].axvline(
                self.sample.mean(),
                color=self.DEFAULT_COLORS[1],
                linewidth=0.5,
                label="Mean"
            )
        self.axes["histogram"].legend(loc=2)

    def toggle_median(self, change=None):
        if hasattr(self, "median_line"):
            self.median_line.remove()
            del self.median_line
        if self.widgets["median"].value:
            self.median_line = self.axes["histogram"].axvline(
                np.median(self.sample),
                color=self.DEFAULT_COLORS[2],
                linewidth=0.5,
                label="Median"
            )
        self.axes["histogram"].legend(loc=2)

    def toggle_parent(self, change=None):
        self.axes["parent"].set_visible(self.widgets["parent"].value)


class GaussianBoxplotExplorer(BaseExplorer):

    SAMPLE_SIZE = {"min": int(1e2), "max": int(1e4), "step": int(1e2)}

    def __init__(self):
        self.base_sample = norm.rvs(
            size=self.SAMPLE_SIZE["max"],
            loc=MU,
            scale=SIGMA
        )
        super().__init__()

    def _this_init(self):
        self.widgets["sample_size"] = IntSlider(
            value=self.SAMPLE_SIZE["min"],
            min=self.SAMPLE_SIZE["min"],
            max=self.SAMPLE_SIZE["max"],
            step=self.SAMPLE_SIZE["step"],
            description="Sample Size"
        )
        self.widgets["sample_size"].observe(self.resample)
        self.control_panel = HBox(
            [self.widgets["sample_size"]],
            layout=self.CENTER_LAYOUT
        )
        self.axes["default"] = self.fig.add_subplot(111)
        self.axes["default"].set_title(
            "Boxplot of sample from the Normal Distribution"
        )
        self.resample()
        self.boxplot["medians"][0].set_label("median")
        self.boxplot["boxes"][0].set_label("interquartile range")
        self.boxplot["fliers"][0].set_label("outlier(s)")
        self.boxplot["whiskers"][0].set_label("whiskers")
        self.boxplot["caps"][0].set_label("caps")
        self.axes["default"].legend()

    def redraw_boxplot(self):
        if hasattr(self, "boxplot"):
            for component_list in self.boxplot.values():
                for component in component_list:
                    component.remove()
            del self.boxplot
        self.boxplot = self.axes["default"].boxplot(self.sample)

    def resample(self, change=None):
        self.sample = np.random.choice(
            self.base_sample,
            size=self.widgets["sample_size"].value,
            replace=False
        )
        self.redraw_boxplot()
