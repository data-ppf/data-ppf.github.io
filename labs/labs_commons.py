import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from IPython.core.display import HTML
from IPython.display import display
from ipywidgets import Layout

plt.style.use("ggplot")

custom_styling = HTML("""
<style>
div.output_subarea.output_html.rendered_html > div {
    text-align: center;
}
</style>
""")


class BaseExplorer:
    """
    Attributes:
        axes ({str: matplotlib.axes.Axes}):
            Dict of all the axes of the figure
        control_panel (ipywidgets.*):
            Object containing most/all of the ipywidgets for the
            Explorer, for the purposes of layout. Usually a
            ipywidget.HBox or ipywidget.VBox
        fig (matplotlib.figure.Figure):
            Central figure of the Explorer
        widgets ({str: ipywidgets.*}):
            Dict of the widgets involved in controlling the Explorer
    """

    DEFAULT_COLORS = [
        color
        for color_dict in plt.rcParams["axes.prop_cycle"]
        for _, color in color_dict.items()
    ]
    CENTER_LAYOUT = Layout(justify_content="center")
    STANDARD_TICKS = 11
    STANDARD_FLOAT_TICK = FormatStrFormatter('%.2f')

    def __init__(self):
        plt.ioff()
        self.fig = plt.figure()
        self.axes = dict()
        self.widgets = dict()
        self.control_panel = None
        self._this_init()
        if self.control_panel is not None:
            display(self.control_panel)
        plt.ion()
        plt.show()

    def _this_init(self):
        """Child class init.

        The method for child classes to add in widgets, subplots, etc.
        in between creating the instance objects and showing them.
        """
        raise NotImplementedError
