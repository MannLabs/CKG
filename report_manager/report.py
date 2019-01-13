import os.path
import plotly.io as pio
import h5py as h5
import json
import plotly.utils

class Report:
    def __init__(self, identifier, plots={}):
        self._identifier = identifier
        self._plots = plots

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier
    
    @property
    def plots(self):
        return self._plots

    @plots.setter
    def plots(self, plots):
        self._plots = plots

    def get_plot(self, plot):
        if plot in self._plots:
            return self._plots[plot]
        return None

    def update_plots(self, plot):
        self._plots.update(plot)

    def print_report(self, directory, plot_format='pdf'):
        for plot_id in self._plots:
            name = "_".join(plot_id)
            for plot in self._plots[plot_id]:
                figure = plot.figure
                pio.write_image(figure, os.path.join(directory,name+"."+plot_format))

    def save_report(self, directory):
        dt = h5.special_dtype(vlen=str)
        with h5.File(os.path.join(directory, "report.h5"), "w") as f:
            for plot_id in self._plots:
                name = "-".join(plot_id)
                for plot in self._plots[plot_id]:
                    grp = f.create_group(name)
                    if hasattr(plot, 'figure'):
                        figure_json = json.dumps(plot.figure, cls=plotly.utils.PlotlyJSONEncoder)
                    elif hasattr(plot, 'data'):
                        figure_json = plot.data
                    else:
                        continue
                    fig_set = grp.create_dataset("figure", (1,), dtype=dt)
                    fig_set[:] = str(figure_json)
                    fig_set.attrs['identifier'] = plot.id
    
    #ToDo load Network data
    def read_report(self, directory):
        with h5.File(os.path.join(directory, "report.h5"), 'r') as f:
            for name in f:
                plot_id = name.split('-')
                if len(plot_id) >1:
                    analysis = plot_id[0]
                    plot_type = plot_id[1]
                figure_json = f[name+"/figure"][0]
                identifier = f[name+"/figure"].attrs["identifier"]
                figure = json.loads(figure_json)
                plot = {"id":identifier, "figure":figure}
                self.update_plots(plot)
