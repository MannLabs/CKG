import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as FF
from scipy.spatial.distance import pdist, squareform
from plotly.graph_objs import *
from kmapper import plotlyviz
import networkx as nx
from report_manager.utils import hex2rgb, getNumberText


def getPlotTraces(data, type = 'lines', div_factor=float(10^10000), horizontal=False):
    '''This function returns traces for different kinds of plots
    --> input:
        - data: is a Pandas DataFrame with one variable as data.index (i.e. 'x') and all others as columns (i.e. 'y')
        - type: 'lines', 'scaled markers', 'bars'
        - div_factor: relative size of the markers
        - horizontal: bar orientation
    --> output:
        List of traces
    '''
    if type == 'lines':
        traces = [go.Scatter(x=data.index, y=data[col], name = col, mode='markers+lines') for col in data.columns]

    elif type == 'scaled markers':
        traces = [go.Scatter(x = data.index, y = data[col], name = col, mode = 'markers', marker = dict(size = data[col].values/div_factor, sizemode = 'area')) for col in data.columns]

    elif type == 'bars':
        traces = [go.Bar(x = data.index, y = data[col], orientation = 'v', name = col) for col in data.columns]
        if horizontal == True:
            traces = [go.Bar(x = data[col], y = data.index, orientation = 'h', name = col) for col in data.columns]

    else: return 'Option not found'

    return traces



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
                        name = g)
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
                            height = 400,
                            width = 900
                        )
    if subplot:
        return (identifier, figure)

    return dcc.Graph(id= identifier, figure = figure)

##ToDo
def get_facet_grid_plot(data, identifier, title, args):
    figure = FF.create_facet_grid(data,
                                x='x',
                                y='y',
                                facet_col='type',
                                color_name='group',
                                color_is_cat=True,
                                trace_type=args['plot_type'],
                                xaxis=args["x_title"],
                                yaxis=args["y_title"],
                                )
    figure['layout']['title'] = title
    figure['layout']['paper_bgcolor'] = None
    figure['layout']['legend'] = None
    figure['layout']['opacity'] = 1
    
    return dcc.Graph(id= identifier, figure = figure)

##ToDo
def scatterplot_matrix(data, identifier, title, grouping_var='group'):
    classes=np.unique(data[grouping_var].values).tolist()
    class_code={classes[k]: k for k in range(len(classes))}
    color_vals=[class_code[cl] for cl in data[grouping_var]]
    text=[data.loc[ k, grouping_var] for k in range(len(data))]

    figure = {}
    figure["data"] = []
    dimensions = []
    for col in data.columns:
        dimensions.append(dict(label=col, values=data[col]))
    
    figure["data"].append(go.Splom(dimensions=dimensions,
                    text=text,
                    marker=dict(color=color_vals,
                                size=7,
                                showscale=False,
                                line=dict(width=0.5,
                                        color='rgb(230,230,230)'))
                    ))

    axis = dict(showline=True,
                zeroline=False,
                gridcolor='#fff',
                ticklen=len(data.columns))
    
    figure["layout"] = go.Layout(title = title,
                            xaxis = dict(axis),
                            yaxis = dict(axis),
                            dragmode='select',
                            width=600,
                            height=600,
                            autosize=False,
                            hovermode='closest',
                            plot_bgcolor='rgba(240,240,240, 0.95)',
                            )

    return dcc.Graph(id=identifier, figure=figure)

def getScatterPlotFigure(data, identifier, title, x_title, y_title, subplot = False):
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
                                height=900,
                                width= 900
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

def plot_2D_scatter(x, y, text='', title='', xlab='', ylab='', hoverinfo='text', color='black', colorscale='Blues', size=8, showscale=False, symmetric_x=False, symmetric_y=False, pad=0.1, hline=False, vline=False, range_x=None, range_y=None):
    figure = {"data":[],"layout":None}
    if range_x is None:
        range_x = [-max(abs(x))-pad, max(abs(x))+pad]#if symmetric_x else []
    if range_y is None:
        range_y = [-max(abs(y))-pad, max(abs(y))+pad]#if symmetric_y else []
    trace = Scattergl(x=x, y=y, mode='markers', text=text, hoverinfo=hoverinfo, marker={'color': color, 'colorscale': colorscale, 'showscale': showscale, 'size': size})
    figure["data"].append(trace)
    figure["layout"] = go.Layout(title=title, xaxis={'title': xlab, 'range': range_x}, yaxis={'title': ylab, 'range': range_y}, hovermode='closest')

    return figure


def plotVolcano(results, title):
    figure = plot_2D_scatter(
        x=results['x'],
        y=results['y'],
        text=results['text'],
        color=results['color'],
        symmetric_x=True,
        xlab='log2FC',
        ylab='-log10value',
        title = title,
        range_y=[0,max(abs(results['y']))+2.5]
    )

    return figure

def runVolcano(identifier, signature, lfc = 1, alpha = 0.05, title = ''):
    # Loop through signature
    color = []
    text = []
    for index, rowData in signature.iterrows():
        # Text
        text.append('<b>'+str(rowData['identifier'])+": "+str(index)+'<br>log2FC = '+str(round(rowData['log2FC'], ndigits=2))+'<br>p = '+'{:.2e}'.format(rowData['pvalue'])+'<br>FDR = '+'{:.2e}'.format(rowData['padj']))

        # Color
        if rowData['padj'] < 0.05:
            if rowData['log2FC'] < -lfc:
                color.append('#2b83ba')
            elif rowData['log2FC'] > lfc:
                color.append('#d7191c')
            elif rowData['log2FC'] < -0.5:
                color.append('#74a9cf')
            elif rowData['log2FC'] > 0.5:
                color.append('#fb6a4a')
            else:
                color.append('grey')
        else:
            if rowData['log2FC'] < -lfc:
                color.append("#abdda4")
            elif rowData['log2FC'] > lfc:
                color.append('#fdae61')
            else:
                color.append('grey')

    # Return
    volcano_plot_results = {'x': signature['log2FC'].values, 'y': signature['-Log pvalue'].values, 'text':text, 'color': color}
    figure = plotVolcano(volcano_plot_results, title)

    return  dcc.Graph(id= identifier, figure = figure)

def getHeatmapFigure(data, identifier, title, format='edgelist', subplot = False):
    '''This function plots a simple Heatmap
    --> input:
        - data: is a Pandas DataFrame with the shape of the heatmap where index corresponds to rows
            and column names corresponds to columns, values in the heatmap corresponds to the row values
        - identifier: is the id used to identify the div where the figure will be generated
        - title: The title of the figure
    --> output:
        Heatmap figure within the <div id="_dash-app-content">
    '''
    df = data.copy()
    if format == "edgelist":
        df = df.set_index('node1')
        df = df.pivot_table(values='weight', index=df.index, columns='node2', aggfunc='first')
        df = df.fillna(0)
    figure = {}
    figure["data"] = []
    figure["layout"] = {"title":title,
                        "height":500,
                        "width":700}
    figure['data'].append(go.Heatmap(z=df.values.tolist(),
                                    x = list(df.columns),
                                    y = list(df.index)))

    if subplot:
        return (identifier, figure)

    return dcc.Graph(id = identifier, figure = figure)

def getComplexHeatmapFigure(data, identifier, title, dist=False, format='edgelist', subplot=False):
    df = data.copy()
    figure = {'data':[], 'layout':{}}

    if format == "edgelist":
        df = df.set_index('node1')
        df = df.pivot_table(values='weight', index=df.index, columns='node2', aggfunc='first')
        df = df.fillna(0)
    
    dendro_up = FF.create_dendrogram(df.values, orientation='bottom', labels=df.columns)
    for i in range(len(dendro_up['data'])):
        dendro_up['data'][i]['yaxis'] = 'y2'

    dendro_side = FF.create_dendrogram(df.values, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    figure['data'].extend(dendro_up['data'])
    figure['data'].extend(dendro_side['data'])

    if dist:
        data_dist = pdist(df.values)
        heat_data = squareform(data_dist)
    else:
        heat_data = df.values
    
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]
    
    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale = 'YlOrRd',
            reversescale=True
        )
    ]

    heatmap[0]['x'] = dendro_up['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    figure['data'].extend(heatmap)

    figure['layout'] = dendro_up['layout']
    
    figure['layout'].update({'width':800, 'height':800,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks':""})
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
    figure['layout'].update({'yaxis2':{'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}}) 

    
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

    labels= list(graph.nodes())
    colors = []
    sizes = []
    for node in node_properties:
        colors.append(node_properties[node]["color"])
        sizes.append(node_properties[node]["size"])

    trace1=Scatter3d(x=Xed,
               y=Yed,
               z=Zed,
               mode='lines',
               opacity = 0.5,
               line=scatter3d.Line(color='rgb(155,155,155)', width=1),
               hoverinfo='text'
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
                             line=scatter3d.marker.Line(color='rgb(0,50,50)', width=15.5)
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
         width=1500,
         height=1500,
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
                size=18,
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

def getBasicTable(data, identifier, title, colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':1500, 'font':12}, subplot = False):
    if subset is not None:
        data = data[subset]
    data_trace = go.Table(header=dict(values=data.columns,
                    fill = dict(color = colors[0]),
                    align = ['left','center']),
                    cells=dict(values=[data[c].round(5) if data[c].dtype == np.float64 else data[c] for c in data.columns],
                    fill = dict(color= colors[1]),
                    align = ['left','center']))
    layout =  dict(
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

def getMapperFigure(data, identifier, title, labels):
    pl_brewer = [[0.0, '#67001f'],
             [0.1, '#b2182b'],
             [0.2, '#d6604d'],
             [0.3, '#f4a582'],
             [0.4, '#fddbc7'],
             [0.5, '#000000'],
             [0.6, '#d1e5f0'],
             [0.7, '#92c5de'],
             [0.8, '#4393c3'],
             [0.9, '#2166ac'],
             [1.0, '#053061']]
    figure = plotlyviz.plotlyviz(data, title=title, colorscale=pl_brewer, color_function_name="Group",factor_size=7, edge_linewidth=2.5,
                        node_linecolor="rgb(200,200,200)", width=1200, height=1200, bgcolor="rgba(240, 240, 240, 0.95)",
                        left=50, bottom=50, summary_height=300, summary_width=400, summary_left=20, summary_right=20,
                        hist_left=25, hist_right=25, member_textbox_width=800, custom_tooltips=labels)
    return  dcc.Graph(id = identifier, figure=figure)

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

