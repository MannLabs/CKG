import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as FF
from scipy.spatial.distance import pdist, squareform
from plotly.graph_objs import *
import networkx as nx
from utils import hex2rgb, getNumberText

def getBarPlotFigure(data, identifier, title, x_title, y_title, group= True, subplot = False):
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
    if group:
        for g in data["group"].unique():
            trace = go.Bar(
                        x = data.loc[data["group"] == g,'x'], # assign x as the dataframe column 'x'
                        y = data.loc[data["group"] == g, 'y'],
                        name = g
                        )
            figure["data"].append(trace)
    else:
        figure["data"].append(
                      go.Bar(
                            x=data['x'], # assign x as the dataframe column 'x'
                            y=data['y']
                        )
                    )
    figure["layout"] = go.Layout(
                            title = title,
                            xaxis = {"title":x_title},
                            yaxis = {"title":y_title},
                            height = 500,
                            width = 900
                        )
    if subplot:
        return (identifier, figure)
        
    return dcc.Graph(id= identifier, figure = figure)

def getScatterPlotFigure(data, identifier, x_title, y_title, title, subplot = False):
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
                                margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest',
                                height=500,
                                width= 500
                                )
    for name in data.name.unique():
        figure["data"].append(go.Scatter(x = data.loc[data["name"] == name, "x"], 
                                        y = data.loc[data['name'] == name, "y"], 
                                        text = name, 
                                        mode = 'markers', 
                                        opacity=0.7,
                                        marker={
                                            'size': 15,
                                            'line': {'width': 0.5, 'color': 'white'}
                                            },
                                        name=name))
    if subplot:
        return (identifier, figure)

    return dcc.Graph(id= identifier, figure = figure)

def plot_2D_scatter(x, y, text='', title='', xlab='', ylab='', hoverinfo='text', color='black', colorscale='Blues', size=8, showscale=False, symmetric_x=False, symmetric_y=False, pad=0.5, hline=False, vline=False, return_trace=False):
    range_x = [-max(abs(x))-pad, max(abs(x))+pad]if symmetric_x else []
    range_y = [-max(abs(y))-pad, max(abs(y))+pad]if symmetric_y else []
    trace = Scattergl(x=x, y=y, mode='markers', text=text, hoverinfo=hoverinfo, marker={'color': color, 'colorscale': colorscale, 'showscale': showscale, 'size': size})
    if return_trace:
        return trace
    else:
        layout = Layout(title=title, xaxis={'title': xlab, 'range': range_x}, yaxis={'title': ylab, 'range': range_y}, hovermode='closest')
        fig = Figure(data=[trace], layout=layout)
    
    return fig

def plotVolcano(results):
    plot_2D_scatter(
        x=results['x'],
        y=results['y'],
        text=results['text'],
        color=results['color'],
        symmetric_x=True,
        xlab='log2FC',
        ylab='-log10value',
    )
def getHeatmapFigure(data, identifier, title, subplot = False):
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
    figure["layout"] = {"title":title,
                        "height":500,
                        "width":700}
    figure['data'].append(go.Heatmap(z=data.values.tolist(),
                                    x = list(data.columns),
                                    y = list(data.index)))

    if subplot:
        return (identifier, figure)

    return dcc.Graph(id = identifier, figure = figure)

def getComplexHeatmapFigure(data, identifier, title, subplot = False):
    figure = FF.create_dendrogram(data.values, orientation='bottom', labels=list(data.columns))
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'
    dendro_side = FF.create_dendrogram(data.values, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    figure['data'] + (dendro_side['data'],)

    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    data_dist = pdist(data.values)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    heatmap = Data([
        Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale = 'YlGnBu'
        )
    ])

    heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    figure['data'] + (Data(heatmap),)

    # Edit Layout
    figure['layout'].update({'width':800, 'height':800,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    # Edit xaxis
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks':""})
    # Edit xaxis2
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                       'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks':""}})

    # Edit yaxis
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
    # Edit yaxis2
    figure['layout'].update({'yaxis2':{'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    if subplot:
        return (identifier, figure)


    return dcc.Graph(id = identifier, figure = figure)

def get3DNetworkFigure(data, sourceCol, targetCol, node_properties, identifier, title, subplot = False):
    '''This function generates a 3D network in plotly
        --> Input:
            - data: Pandas DataFrame with the format: source    target  edge_property1 ...
            - sourceCol: name of the column with the source node
            - targetCol: name of the column with the target node
            - node_properties: dictionary with the properties for each node: {node:{color:..,size:..}}
            - identifier: identifier used to label the div that contains the network figure
            - Title of the plot
    '''
    edge_properties = [c for c in data.columns if c not in [sourceCol, targetCol]]
    graph = nx.from_pandas_edgelist(data, sourceCol, targetCol, edge_properties)
    pos=nx.fruchterman_reingold_layout(graph, dim=3)
    edges = graph.edges()
    N = len(graph.nodes())
    
    Xn=[pos[k][0] for k in pos]
    Yn=[pos[k][1] for k in pos]
    Zn=[pos[k][2] for k in pos]# z-coordinates

    Xed=[]
    Yed=[]
    Zed=[]
    for edge in edges:
        Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
        Yed+=[pos[edge[0]][1],pos[edge[1]][1], None] 
        Zed+=[pos[edge[0]][2],pos[edge[1]][2], None]

    labels=[]
    colors = []
    sizes = []
    for node in node_properties:
        labels.append(node)
        colors.append(node_properties[node]["color"])
        sizes.append(node_properties[node]["size"])
    
    trace1=Scatter3d(x=Xed,
               y=Yed,
               z=Zed,
               mode='lines',
               line=scatter3d.Line(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )
    trace2=Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='proteins',
               marker=scatter3d.Marker(symbol='circle',
                             size=sizes,
                             color=colors,
                             colorscale='Viridis',
                             line=scatter3d.marker.Line(color='rgb(50,50,50)', width=5.5)
                             ),
               text=labels,
               hoverinfo='text'
               )
    axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,

          showticklabels=False,
          title=''
          )

    layout = Layout(
         title=title + "(3D visualization)",
         width=800,
         height=800,
         showlegend=True,
         scene=Scene(
         xaxis=XAxis(axis),
         yaxis=YAxis(axis),
         zaxis=ZAxis(axis),
        ),
     margin=Margin(
        t=100
    ),
    hovermode='closest',
    
    )

    data=Data([trace1, trace2])
    figure=Figure(data=data, layout=layout)

    if subplot:
        return (identifier, figure)

    return dcc.Graph(id = identifier, figure = figure)


def get2DPCAFigure(data, components, identifier, title, subplot = False):
    traces = []
    groups = data["groups"].unique()

    for name in groups:
        trace = Scatter(
            x=Y_sklearn[y==name,components[0]],
            y=Y_sklearn[y==name,components[1]],
            mode='markers',
            name=name,
            marker=scatter.Marker(
                size=12,
                line=scatter.Line(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

    d = Data(traces)
    layout = Layout(xaxis=layout.XAxis(title='PC'+str(components[0]), showline=False),
                    yaxis=layout.YAxis(title='PC'+str(components[1]), showline=False))
    figure = Figure(data=d, layout=layout)

    if subplot:
        return (identifier, figure)


    return  dcc.Graph(id = identifier, figure = figure)


def getSankeyPlot(data, sourceCol, targetCol, weightCol, edgeColorCol, node_colors, identifier, title, plot_attr = {'orientation': 'h', 'valueformat': '.0f', 'arrangement':'freeform','width':800, 'height':800, 'font':12}, subplot = False):
    '''This function generates a Sankey plot in Plotly
        --> Input:
            - data: Pandas DataFrame with the format: source    target  weight
            - sourceCol: name of the column with the source node
            - targetCol: name of the column with the target node
            - weightCol: name of the column with the edge weight
            - edgeColorCol: name of the column with the edge color
            - colors: dictionary with the color for each node: {node: rgba(r,g,b,alpha)}
            - identifier: identifier used to label the div that contains the network figure
            - Title of the plot
    '''
    data_trace = dict(
        type='sankey',
        domain = dict(
            x =  [0,1],
            y =  [0,1]
        ),
        orientation = 'h' if 'orientation' not in plot_attr else plot_attr['orientation'],
        valueformat = ".0f" if 'valueformat' not in plot_attr else plot_attr['valueformat'],
        arrangement = 'freeform' if 'arrangement' not in plot_attr else plot_attr['arrangement'],
        node = dict(
            pad = 15 if 'pad' not in plot_attr else plot_attr['pad'],
            thickness = 25 if 'thickness' not in plot_attr else plot_attr['thickness'],
            line = dict(
                color = "black",
                width = 0.3
            ),
            label =  list(node_colors.keys()),
            color =  ["rgba"+str(hex2rgb(c)) if c.startswith('#') else c  for c in list(node_colors.values())]
        ),    
        link = dict(
            source =  [list(node_colors.keys()).index(i) for i in data[sourceCol].tolist()],
            target =  [list(node_colors.keys()).index(i) for i in data[targetCol].tolist()],
            value =  data[weightCol].tolist(),
            color = ["rgba"+str(hex2rgb(c)) if c.startswith('#') else c for c in data[edgeColorCol].tolist()]
        ))
    layout =  dict(
        width= 800 if 'width' not in plot_attr else plot_attr['width'],
        height= 800 if 'height' not in plot_attr else plot_attr['height'],
        title = title,
        font = dict(
            size = 12 if 'font' not in plot_attr else plot_attr['font'],
        )
    )
    
    figure = dict(data=[data_trace], layout=layout)

    if subplot:
        return (identifier, figure)
    
    return dcc.Graph(id = identifier, figure = figure)

def getBasicTable(data, identifier, title, colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':800, 'height':800, 'font':12}, subplot = False):
    if subset is not None:
        data = data[subset]

    data_trace = go.Table(header=dict(values=data.columns,
                    fill = dict(color = colors[0]),
                    align = ['left'] * 5),
                    cells=dict(values=[data[c] for c in data.columns],
                    fill = dict(color= colors[1]),
                    align = ['left'] * 5))
    layout =  dict(
        width= 300 if 'width' not in plot_attr else plot_attr['width'],
        height= 300 if 'height' not in plot_attr else plot_attr['height'],
        title = title,
        font = dict(
            size = 12 if 'font' not in plot_attr else plot_attr['font'],
        )
    )
    
    figure = dict(data=[data_trace], layout=layout)

    if subplot:
        return (identifier, figure)
    
    return dcc.Graph(id = identifier, figure = figure)

def getViolinPlot(data, variableCol, groupCol, colors, identifier, title, plot_attr={'width':600, 'height':600, 'colorScale': False}, subplot = False):

    figure = FF.create_violin(data, 
                    data_header=variableCol, 
                    group_header= groupCol,
                    colors= colors, 
                    title = title,
                    height=500 if 'height' not in plot_attr else plot_attr['height'], 
                    width=500 if 'width' not in plot_attr else plot_attr['width'],
                    use_colorscale= False if 'colorScale' not in plot_attr else plot_attr['colorScale'])
    if subplot:
        return (identifier, figure)
    
    return dcc.Graph(id = identifier, figure = figure)

def getDashboardLayout():
    layout = dict(
                autosize=True,
                height=500,
                font=dict(color='#CCCCCC'),
                titlefont=dict(color='#CCCCCC', size='14'),
                margin=dict(
                           l=35,
                           r=35,
                           b=35,
                           t=45
                           ),
                hovermode="closest",
                plot_bgcolor="#191A1A",
                paper_bgcolor="#020202",
                legend=dict(font=dict(size=10), orientation='h'),
                title='Satellite Overview',
                mapbox=dict(
                            style="dark",
                            center=dict(
                            lon=-78.05,
                            lat=42.54
                            ),
                zoom=7,
                )
            )

    return layout


def getDashboardPlots(figures, cols, distribution):   
    divs = []
    
    className = getNumberText(cols)+' columns offset-by-one'
    i = 0
    for name,dist in distribution:
        row = [html.H5(
                        '',
                        id = name,
                        className= getNumberText(len(dist))+' columns'
                        )
        ]
        for c in dist:
            identifier, figure = figures[i]
            i = i + 1
            row.append(html.Div([
                            dcc.Graph(id= identifier, figure = dict(data= figure, layout=getDashboardLayout()))
                            ],
                            className=str(c)+' columns',
                            style={'margin-top': '20'}
                            )
                        )
        divs.append(html.Div(row, className = 'row'))
    
    return html.Div(divs, className = className)
