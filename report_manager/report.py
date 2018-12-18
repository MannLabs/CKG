

class Report:
    def __init__(self,identifier, plots = {}):
        self.identifier = identifier
        self.plots = plots

    @property
    def idenfitifer(self):
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

    def getPlot(self, plot):
        if plot in self.plots:
            return self.plots[plot]
        return None

    def updatePlots(self, plot):
        self.plots.update(plot)
