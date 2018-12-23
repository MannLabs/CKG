import os.path
import plotly.io as pio

class Report:
    def __init__(self,identifier, plots = {}):
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
        if plot in self.plots:
            return self.plots[plot]
        return None

    def update_plots(self, plot):
        self.plots.update(plot)

    def save_report(self, directory, plot_format='pdf'):
        for plot_id in self.plots:
            name = "_".join(plot_id)
            for plot in self.plots[plot_id]:
                figure = plot.figure
                pio.write_image(figure, os.path.join(directory,name+"."+plot_format))
