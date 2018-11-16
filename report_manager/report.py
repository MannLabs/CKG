

class Report:
    def __init__(self,identifier, plots = {}):
        self.identifier = identifier
        self.plots = plots

    def getIdenfitifer(self):
        return self.identifier

    def getPlots(self):
        return self.plots

    def getPlot(self, plot):
        if plot in self.plots:
            return self.plots[plot]
        return None

    def setIdentifier(self, identifier):
        self.identifier = identifier

    def setPlots(self, plots):
        self.plots = plots

    def updatePlots(self, plot):
        self.plots.update(plot)
