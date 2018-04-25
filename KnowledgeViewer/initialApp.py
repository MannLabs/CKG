import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash()

def getPageTitle(title):
    return html.H1(children= title)

def getPageSubtitle(subtitle):
    return html.Div(children=subtitle)

def getBarPlotFigure(data, identifier, title):
    '''This function plots a simple barplot
    --> input:
        - data: is a Pandas DataFrame with three columns: "name" of the bars, 'x' values and 'y' values to plot
        - identifier: is the id used to identify the div where the figure will be generated
        - title: The title of the figure
    --> output:
        Barplot figure within the <div id="_dash-app-content">
    '''
    figure = {}
    figure["data"] = []
    figure["layout"] = {"title":title}
    for name in data.name.unique():
        figure["data"].append({'x': np.unique(data['x'].values), 'y':data.loc[data['name'] == name,'y'].values , 'type': 'bar', 'name': name})

        
    return dcc.Graph(id= identifier, figure = figure)

def getScatterPlotFigure(data, identifier, x, y, x_title, y_title, title):
    '''This function plots a simple Scatterplot
    --> input:
        - data: is a Pandas DataFrame with four columns: "name", x values and y values (provided as variables) to plot
        - identifier: is the id used to identify the div where the figure will be generated
        - title: The title of the figure
    --> output:
        Scatterplot figure within the <div id="_dash-app-content">
    '''
    figure = {}
    figure["data"] = []
    figure["layout"] = go.Layout(title = title,
                                xaxis={'title': x_title},
                                yaxis={'title': y_title},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest'
                                )
    for name in data.name.unique():
        figure["data"].append(go.Scatter(x = data.loc[data["name"] == name, x], 
                                        y = data.loc[data['name'] == name, y], 
                                        text = name, 
                                        mode = 'markers', 
                                        opacity=0.7,
                                        marker={
                                            'size': 15,
                                            'line': {'width': 0.5, 'color': 'white'}
                                            },
                                        name=name))
       
    return dcc.Graph(id= identifier, figure = figure)


def getHeatmapFigure(data, identifier, title):
    '''This function plots a simple Heatmap
    --> input:
        - data: is a Pandas DataFrame with the shape of the heatmap where index corresponds to rows
            and column names corresponds to columns, values in the heatmap corresponds to the row values
        - identifier: is the id used to identify the div where the figure will be generated
        - title: The title of the figure
    --> output:
        Heatmap figure within the <div id="_dash-app-content">
    '''
    figure = {}
    figure["data"] = []
    figure["layout"] = {"title":title}
    figure['data'].append(go.Heatmap(z=data.values.tolist(),
                                    x = list(data.columns),
                                    y = list(data.index)))
    return dcc.Graph(id = identifier, figure = figure)

def appendToDashLayout(data, layout):
    return layout.append(data)


def layoutData(layout):
    app.layout = html.Div(children= layout)


if __name__ == '__main__':
    layout = []
    title = getPageTitle("This is a test")
    layout.append(title)
    subtitle = getPageSubtitle("I am just playing with Dash")
    layout.append(subtitle)
    data = pd.DataFrame([("a", "1", 2), ("b", "1", 3), ("a","2",12), ("b","2",2)], columns = ["name", "x", "y"])
    figure = getBarPlotFigure(data, identifier = "myPlot", title= "Oh what a figure")
    layout.append(figure)
    data = pd.DataFrame([("Protein 1", 1, 2), ("Protein 2", 1, 3), ("Protein 3", 2, 0.5), ("Protein 4",2 ,4)], columns = ["name", "AS1", "AS2"])
    figure2 = getScatterPlotFigure(data, identifier= "myPlot2", x = "AS1", y = "AS2", x_title = "Analytical Sample 1", y_title = "Analytical Sample 2", title = "Correlation Analytical Samples")
    layout.append(figure2)
    data = pd.DataFrame([("Protein 1", 1, 2, 2, 5), ("Protein 2", 1, 3, 3, 3), ("Protein 3", 2, 0.5, 0), ("Protein 4",2 ,4, 10, 20)], columns = ["name", "AS1", "AS2", "AS3", "AS4"])
    data = data.set_index("name")
    figure3 = getHeatmapFigure(data, identifier = "Heat", title = "What a heatmap!")
    layout.append(figure3)
    layoutData(layout)

    app.run_server(debug=True)
