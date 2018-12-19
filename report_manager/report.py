

class Report:
    def __init__(self,identifier, plots = {}):
        self.identifier = identifier
        self.plots = plots

    @property
    def identifier(self):
        return self.identifier

    @identifier.setter
    def identifier(self, identifier):
        self.identifier = identifier
    
    @property
    def plots(self):
        return self.plots

    @plots.setter
    def plots(self, plots):
        self.plots = plots

    def get_plot(self, plot):
        if plot in self.plots:
            return self.plots[plot]
        return None

    def update_plots(self, plot):
        self.plots.update(plot)
