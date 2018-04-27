import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from plots import basicFigures
import pandas as pd
import dash_auth
import basicApp

class ProjectApp(basicApp.BasicApp):
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None):
        self.projectId = projectId
        basicApp.BasicApp.__init__(self, title, subtitle, description, layout, logo, footer)

    def buildPage(self):
        self.addBasicLayout()
        
        data = pd.DataFrame([("a", "1", 2), ("b", "1", 3), ("a","2",12), ("b","2",2)], columns = ["name", "x", "y"])
        figure = basicFigures.getBarPlotFigure(data, identifier = "myPlot", title= "Oh what a figure")
        self.AddToLayout(figure)
        
        data = pd.DataFrame([("Protein 1", 1, 2), ("Protein 2", 1, 3), ("Protein 3", 2, 0.5), ("Protein 4",2 ,4)], columns = ["name", "AS1", "AS2"])
        figure2 = basicFigures.getScatterPlotFigure(data, identifier= "myPlot2", x = "AS1", y = "AS2", x_title = "Analytical Sample 1", y_title = "Analytical Sample 2", title = "Correlation Analytical Samples")
        self.AddToLayout(figure2)
        
        data = pd.DataFrame([("Protein 1", 1, 2, 2, 5), ("Protein 2", 1, 3, 3, 3), ("Protein 3", 2, 0.5, 0, 0.8), ("Protein 4",2 ,4, 10, 20)], columns = ["name", "AS1", "AS2", "AS3", "AS4"])
        data = data.set_index("name")
        figure3 = basicFigures.getHeatmapFigure(data, identifier = "Heat", title = "What a heatmap!")
        self.AddToLayout(figure3)
                    
        figure4 = basicFigures.getComplexHeatmapFigure(data, identifier = "Heat 2", title = "WHHHHHHA")
        self.AddToLayout(figure4)
        
        data = pd.DataFrame([("Protein 1", "Protein 2", 2, 5), ("Protein 2", "Protein 3", 3, 3), ("Protein 3", "Protein 1", 0.5, 0), ("Protein 1","Protein 4" ,4, 10),("Protein 2","Protein 4" ,4, 10),("Protein 1","Protein 5" ,4, 10),("Protein 5","Protein 4" ,4, 10),("Protein 3","Protein 6" ,4, 10),("Protein 6","Protein 4" ,4, 10),("Protein 7","Protein 4" ,4, 10)], columns = ["source", "target", "weight", "score"])
        properties = {"Protein 1":{"color":"#edf8fb","size":15}, "Protein 2":{"color":"#bfd3e6", "size":20}, "Protein 3":{"color":"#9ebcda", "size":12}, "Protein 4":{"color":"#8c96c6","size":15}, "Protein 5":{"color":"#8c6bb1", "size":14}, "Protein 6":{"color":"#88419d", "size":13}, "Protein 7":{"color":"#6e016b", "size":18}}
        figure5 = basicFigures.get3DNetworkFigure(data, sourceCol = "source", targetCol = "target", node_properties = properties, identifier = "Net", title= "This is a 3D network")
        self.AddToLayout(figure5)


