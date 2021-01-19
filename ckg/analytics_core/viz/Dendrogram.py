import numpy as np
import scipy as scp
from collections import OrderedDict
import plotly.graph_objs as go

def plot_dendrogram(Z_dendrogram, cutoff_line=True, value=15, orientation='bottom', hang=30, hide_labels=False, labels=None,
                    colorscale=None, hovertext=None, color_threshold=None):
    """
    Modified version of Plotly _dendrogram.py that returns a dendrogram Plotly figure object with cutoff line.

    :param Z_dendrogram: Matrix of observations as array of arrays
    :type Z_dendrogram: ndarray
    :param cutoff_line: plot distance cutoff line
    :type cutoff_line: boolean
    :param value: dendrogram distance for cutoff line
    :type value: float or int
    :param orientation: 'top', 'right', 'bottom', or 'left'
    :type orientation: str
    :param hang: dendrogram distance of leaf lines
    :type hang: float
    :param hide_labels: show leaf labels
    :type hide_labels: boolean
    :param labels: List of axis category labels(observation labels)
    :type labels: list
    :param colorscale: Optional colorscale for dendrogram tree
    :type colorscale: list
    :param hovertext: List of hovertext for constituent traces of dendrogram
                               clusters
    :type hovertext: list[list]
    :param color_threshold: Value at which the separation of clusters will be made
    :type color_threshold: double
    :return: Plotly figure object

    Example::

        figure = plot_dendrogram(dendro_tree, hang=0.9, cutoff_line=False)
    """

    dendrogram = Dendrogram(Z_dendrogram, orientation, hang, hide_labels, labels, colorscale, hovertext=hovertext, color_threshold=color_threshold)

    if cutoff_line == True:
        dendrogram.layout.update({'shapes':[{'type':'line',
                                 'xref':'paper',
                                 'yref':'y',
                                 'x0':0, 'y0':value,
                                 'x1':1, 'y1':value,
                                 'line':{'color':'red'}}]})

    figure = dict(data=dendrogram.data, layout=dendrogram.layout)
    figure['layout']['template'] = 'plotly_white'

    return figure

class Dendrogram(object):
    """Refer to plot_dendrogram() for docstring."""
    def __init__(self, Z_dendrogram, orientation='bottom', hang=1, hide_labels=False, labels=None, colorscale=None, hovertext=None,
                 color_threshold=None, width=np.inf, height=np.inf, xaxis='xaxis', yaxis='yaxis'):
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}

        if self.orientation in ['left', 'bottom']:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ['right', 'bottom']:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        (dd_traces, xvals, yvals,
            ordered_labels, leaves) = self.get_dendrogram_traces(Z_dendrogram, hang, colorscale,
                                                                 hovertext,
                                                                 color_threshold)

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(l_border,
                                       r_border + 1,
                                       int((r_border - l_border) / len(yvals)))
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height, hide_labels=hide_labels)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        """
        Returns colorscale used for dendrogram tree clusters.

        :param colorscale: colors to use for the plot in rgb format
        :type colorscale: list
        :return (dict): default colors mapped to the user colorscale
        """

        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        d = {'r': 'red',
             'g': 'green',
             'b': 'blue',
             'c': 'cyan',
             'm': 'magenta',
             'y': 'yellow',
             'k': 'black',
             'w': 'white'}
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            colorscale = [
                'rgb(0,116,217)',  # blue
                'rgb(35,205,205)',  # cyan
                'rgb(61,153,112)',  # green
                'rgb(40,35,35)',  # black
                'rgb(133,20,75)',  # magenta
                'rgb(255,65,54)',  # red
                'rgb(255,255,255)',  # white
                'rgb(255,220,0)']  # yellow

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(colorscale):
                default_colors[k] = colorscale[i]

        return default_colors

    def set_axis_layout(self, axis_key, hide_labels):
        """
        Sets and returns default axis object for dendrogram figure.

        :param axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :type axis_key: str
        :return (dict): An axis_key dictionary with set parameters.
        """
        axis_defaults = {
                'type': 'linear',
                'ticks': 'outside',
                'mirror': 'allticks',
                'rangemode': 'tozero',
                'showticklabels': True,
                'zeroline': False,
                'showgrid': False,
                'showline': True,
            }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ['left', 'right']:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]['tickvals'] = \
                [zv*self.sign[axis_key] for zv in self.zero_vals]
            self.layout[axis_key_labels]['ticktext'] = self.labels
            self.layout[axis_key_labels]['tickmode'] = 'array'

        self.layout[axis_key].update(axis_defaults)

        if hide_labels == True:
            self.layout[axis_key].update({'showticklabels': False})
        else: pass

        return self.layout[axis_key]

    def set_figure_layout(self, width, height, hide_labels):
        """
        Sets and returns default layout object for dendrogram figure.

        :param width: plot width
        :type width: int
        :param height: plot height
        :type height: int
        :param hide_labels: show leaf labels
        :type hide_labels: boolean
        :return: Plotly layout
        """
        self.layout.update({
            'showlegend': False,
            'autosize': False,
            'hovermode': 'closest',
            'width': width,
            'height': height
        })

        self.set_axis_layout(self.xaxis, hide_labels=hide_labels)
        self.set_axis_layout(self.yaxis, hide_labels=False)

        return self.layout


    def get_dendrogram_traces(self, Z_dendrogram, hang, colorscale, hovertext, color_threshold):
        """
        Calculates all the elements needed for plotting a dendrogram.
        
        :param Z_dendrogram: Matrix of observations as array of arrays
        :type Z_dendrogram: ndarray
        :param hang: dendrogram distance of leaf lines
        :type hang: float
        :param colorscale: Color scale for dendrogram tree clusters
        :type colorscale: list
        :param hovertext: List of hovertext for constituent traces of dendrogram
        :type hovertext: list
        :return (tuple): Contains all the traces in the following order:
         
                a. trace_list: List of Plotly trace objects for dendrogram tree
                b. icoord: All X points of the dendrogram tree as array of arrays with length 4
                c. dcoord: All Y points of the dendrogram tree as array of arrays with length 4
                d. ordered_labels: leaf labels in the order they are going to appear on the plot
                e. Z_dendrogram['leaves']: left-to-right traversal of the leaves
        """
        icoord = scp.array(Z_dendrogram['icoord'])
        dcoord = scp.array(Z_dendrogram['dcoord'])
        ordered_labels = scp.array(Z_dendrogram['ivl'])
        color_list = scp.array(Z_dendrogram['color_list'])
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
            if self.orientation in ['top', 'bottom']:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ['top', 'bottom']:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]

            coord = [list(a) for a in zip(xs, ys)]
            x_coord = []
            y_coord = []
            y_at_x = {}
            for n, seg in enumerate(coord):
                x, y = seg
                if y > 0 and y < y_at_x.get(x, np.inf):
                    y_at_x[x] = y
            for n, seg in enumerate(coord):
                x, y = seg
                if y == 0:
                    y = max(0, y_at_x.get(x, 0) - hang)
                x_coord.append(x)
                y_coord.append(y)

            trace = dict(
                type='scattergl',
                x=np.multiply(self.sign[self.xaxis], x_coord),
                y=np.multiply(self.sign[self.yaxis], y_coord),
                mode='lines',
                marker=dict(color='rgb(40,35,35)'),
                line=dict(color='rgb(40,35,35)', width=1),      #dict(color=colors[color_key]),
                text=hovertext_label,
                hoverinfo='text')

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ''

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ''

            trace['xaxis'] = 'x' + x_index
            trace['yaxis'] = 'y' + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, Z_dendrogram['leaves']
