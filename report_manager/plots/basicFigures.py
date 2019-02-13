import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as FF
from plotly import tools
from scipy.spatial.distance import pdist, squareform
from plotly.graph_objs import *
from kmapper import plotlyviz
import networkx as nx
from networkx.readwrite import json_graph
from dash_network import Network
from report_manager.utils import hex2rgb, getNumberText
from report_manager.plots import Dendrogram
from report_manager.plots import wgcnaFigures


def getPlotTraces(data, key='full', type = 'lines', div_factor=float(10^10000), horizontal=False):
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
        traces = [go.Scatter(x=data.index, y=data[col], name = col+' '+key, mode='markers+lines') for col in data.columns]

    elif type == 'scaled markers':
        traces = [go.Scatter(x = data.index, y = data[col], name = col+' '+key, mode = 'markers', marker = dict(size = data[col].values/div_factor, sizemode = 'area')) for col in data.columns]

    elif type == 'bars':
        traces = [go.Bar(x = data.index, y = data[col], orientation = 'v', name = col+' '+key) for col in data.columns]
        if horizontal == True:
            traces = [go.Bar(x = data[col], y = data.index, orientation = 'h', name = col+' '+key) for col in data.columns]

    else: return 'Option not found'

    return traces


def get_distplot(data, identifier, args):
    df = data.copy()
    graphs = []

    df = df.set_index(args['group'])
    df = df.transpose()
    df = df.dropna()

    for i in df.index.unique():
        hist_data = []
        for c in df.columns.unique():
            hist_data.append(df.loc[i,c].values.tolist())
        group_labels = df.columns.unique().tolist()
        # Create distplot with custom bin_size
        fig = FF.create_distplot(hist_data, group_labels, bin_size=.5, curve_type='normal')
        fig['layout'].update(height=600, width=1000, title='Distribution plot '+i)
        graphs.append(dcc.Graph(id=identifier+"_"+i, figure=fig))

    return graphs

def get_barplot(data, identifier, args):
    '''This function plots a simple barplot
    --> input:
        - data: is a Pandas DataFrame with three columns: "name" of the bars, 'x' values and 'y' values to plot
        - identifier: is the id used to identify the div where the figure will be generated
        - title: The title of the figure
    --> output:
        Barplot figure within the <div id="_dash-app-content">
    '''
    height = 400
    width = 900
    figure = {}
    figure["data"] = []
    if "group" in args:
        for g in data[args["group"]].unique():
            trace = go.Bar(
                        x = data.loc[data["group"] == g,args['x']], # assign x as the dataframe column 'x'
                        y = data.loc[data["group"] == g, args['y']],
                        name = g)
            figure["data"].append(trace)
    else:
        figure["data"].append(
                      go.Bar(
                            x=data[args['x']], # assign x as the dataframe column 'x'
                            y=data[args['y']]
                        )
                    )
    figure["layout"] = go.Layout(
                            title = args['title'],
                            xaxis={"title":args["x_title"]},
                            yaxis={"title":args["y_title"]},
                            height = height,
                            width = width
                        )

    return dcc.Graph(id= identifier, figure = figure)

##ToDo
def get_facet_grid_plot(data, identifier, args):
    figure = FF.create_facet_grid(data,
                                x=args['x'],
                                y=args['y'],
                                marker={'opacity':1.},
                                facet_col=args['class'],
                                color_name=args['group'],
                                color_is_cat=True,
                                trace_type=args['plot_type'],
                                )
    figure['layout']['title'] = args['title'].title()
    figure['layout']['paper_bgcolor'] = None
    figure['layout']['legend'] = None

    return dcc.Graph(id= identifier, figure = figure)

def get_scatterplot_matrix(data, identifier, args):
    df = data.copy()
    if "format" in args:
        if args["format"] == "long":
            columns = [args["variables"], args["values"]]
            df = data.copy()
            groups = df[args["group"]]
            df = df[columns]
            df = df.pivot(columns=args["variables"], values=args["values"])
            df['group'] = groups
    classes=np.unique(df[args["group"]].values).tolist()
    class_code={classes[k]: k for k in range(len(classes))}
    color_vals=[class_code[cl] for cl in df[args["group"]]]
    text=[data.loc[ k, args["group"]] for k in range(len(df))]

    figure = {}
    figure["data"] = []
    dimensions = []
    for col in df.columns:
        if col != args["group"]:
            dimensions.append(dict(label=col, values=df[col]))

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
                ticklen=4)

    figure["layout"] = go.Layout(title = args["title"],
                            xaxis = dict(axis),
                            yaxis = dict(axis),
                            dragmode='select',
                            width=1500,
                            height=1500,
                            autosize=True,
                            hovermode='closest',
                            plot_bgcolor='rgba(240,240,240, 0.95)',
                            )

    return dcc.Graph(id=identifier, figure=figure)

def get_scatterplot(data, identifier, args):
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
    figure["layout"] = go.Layout(title = args['title'],
                                xaxis= {"title": args['x_title']},
                                yaxis= {"title": args['y_title']},
                                margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest',
                                height=900,
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

    return dcc.Graph(id= identifier, figure = figure)

def get_volcanoplot(results, args):
    figures = []
    for identifier,title in results:
        result = results[(identifier,title)]
        figure = {"data":[],"layout":None}
        if "range_x" not in args:
            range_x = [-max(abs(result['x']))-0.1, max(abs(result['x']))+0.1]#if symmetric_x else []
        else:
            range_x = args["range_x"]
        if "range_y" not in args:
            range_y = [0,max(abs(result['y']))+2.5]
        else:
            range_y = args["range_y"]

        trace = Scattergl(x=result['x'],
                        y=result['y'],
                        mode='markers',
                        text=result['text'],
                        hoverinfo='text',
                        marker={'color': result['color'], 'colorscale': args["colorscale"], 'showscale': args['showscale'], 'size': args['marker_size']}
                        )

        figure["data"].append(trace)
        figure["layout"] = go.Layout(title=title,
                                        xaxis={'title': args['x_title'], 'range': range_x},
                                        yaxis={'title': args['y_title'], 'range': range_y},
                                        hovermode='closest',
                                        shapes=[
                                                {'type': 'line',
                                                'x0': np.log10(args['fc']),
                                                'y0': 0,
                                                'x1': np.log10(args['fc']),
                                                'y1': max(abs(result['y']))+2.5,
                                                'line': {
                                                    'color': 'grey',
                                                    'width': 2,
                                                    'dash':'dashdot'
                                                    },
                                                    },
                                                {'type': 'line',
                                                'x0': -np.log10(args['fc']),
                                                'y0': 0,
                                                'x1': -np.log10(args['fc']),
                                                'y1': max(abs(result['y']))+2.5,
                                                'line': {
                                                    'color': 'grey',
                                                    'width': 2,
                                                    'dash': 'dashdot'
                                                    },
                                                    },
                                                {'type': 'line',
                                                'x0': -max(abs(result['x']))-0.1,
                                                'y0': result['pvalue'],
                                                'x1': max(abs(result['x']))+0.1,
                                                'y1': result['pvalue'],
                                                'line': {
                                                    'color': 'grey',
                                                    'width': 1,
                                                    'dash': 'dashdot'
                                                    },
                                                    }
                                                ],
                                        showlegend=False)

        figures.append(dcc.Graph(id= identifier, figure = figure))
    return figures

def run_volcano(data, identifier, args):
    # Loop through signature

    volcano_plot_results = {}
    grouping = data.groupby(['group1','group2'])
    for group in grouping.groups:
        signature = grouping.get_group(group)
        color = []
        text = []
        gidentifier = identifier + "_".join(group)
        title = 'Comparison: '+str(group[0])+' vs '+str(group[1])
        min_sig_pval = 0
        for index, row in signature.iterrows():
            # Text
            text.append('<b>'+str(row['identifier'])+": "+str(index)+'<br>Comparison: '+str(row['group1'])+' vs '+str(row['group2'])+'<br>logFC = '+str(round(row['logFC'], ndigits=2))+'<br>p = '+'{:.2e}'.format(row['pvalue'])+'<br>FDR = '+'{:.2e}'.format(row['padj']))

            # Color
            if row['padj'] < args['alpha']:
                min_sig_pval = row['pvalue'] if row['pvalue'] > min_sig_pval else min_sig_pval
                if row['FC'] < -args['fc']:
                    color.append('#2c7bb6')
                elif row['FC'] > args['fc']:
                    color.append('#d7191c')
                elif row['FC'] < -1.:
                    color.append('#abd9e9')
                elif row['FC'] > 1.:
                    color.append('#fdae61')
                else:
                    color.append('silver')
            else:
                color.append('silver')

        # Return
        volcano_plot_results[(gidentifier, title)] = {'x': signature['logFC'].values, 'y': signature['-Log pvalue'].values, 'text':text, 'color': color, 'pvalue':-np.log10(min_sig_pval)}

    figures = get_volcanoplot(volcano_plot_results, args)

    return figures

def get_heatmapplot(data, identifier, args):
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
    if args['format'] == "edgelist":
        df = df.set_index(args['source'])
        df = df.pivot_table(values=args['values'], index=df.index, columns=args['target'], aggfunc='first')
        df = df.fillna(0)
    figure = {}
    figure["data"] = []
    figure["layout"] = {"title":args['title'],
                        "height": 500,
                        "width": 700}
    figure['data'].append(go.Heatmap(z=df.values.tolist(),
                                    x = list(df.columns),
                                    y = list(df.index)))

    return dcc.Graph(id = identifier, figure = figure)

def get_complex_heatmapplot(data, identifier, args):
    df = data.copy()
    figure = {'data':[], 'layout':{}}

    if args['format'] == "edgelist":
        df = df.set_index(args['source'])
        df = df.pivot_table(values=args['values'], index=df.index, columns=args['target'], aggfunc='first')
        df = df.fillna(0)

    dendro_up = FF.create_dendrogram(df.values, orientation='bottom', labels=df.columns)
    for i in range(len(dendro_up['data'])):
        dendro_up['data'][i]['yaxis'] = 'y2'

    dendro_side = FF.create_dendrogram(df.values, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    figure['data'].extend(dendro_up['data'])
    figure['data'].extend(dendro_side['data'])

    if args['dist']:
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


    return dcc.Graph(id=identifier, figure=figure)

def get_network(data, identifier, args):
    net = None
    if not data.empty:
        data = data.rename(index=str, columns={args['values']: "width"})
        edge_prop_columns = [c for c in data.columns if c not in [args['source'], args['target']]]
        edge_properties = [str(d) for d in data.to_dict(orient='index').values()]
        graph = nx.from_pandas_edgelist(data, args['source'], args['target'], edge_prop_columns)
        betweenness = nx.betweenness_centrality(graph)
        degrees = dict(graph.degree())
        if args['node_size'] == 'betweenness':
            nx.set_node_attributes(graph, betweenness, 'radius')
        elif args['node_size'] == 'degree':
            nx.set_node_attributes(graph, degrees, 'radius')

        jgraph = json_graph.node_link_data(graph)

        net = Network(id=identifier, data=jgraph, width=args['width'], height=args['height'], maxLinkWidth=args['maxLinkWidth'], maxRadius=args['maxRadius'])

    return net

def get_3d_network(data, identifier, args):
    '''This function generates a 3D network in plotly
        --> Input:
            - data: Pandas DataFrame with the format: source    target  edge_property1 ...
            - sourceCol: name of the column with the source node
            - targetCol: name of the column with the target node
            - node_properties: dictionary with the properties for each node: {node:{color:..,size:..}}
            - identifier: identifier used to label the div that contains the network figure
            - Title of the plot
    '''
    edge_prop_columns = [c for c in data.columns if c not in [args['source'], args['target']]]
    edge_properties = [str(d) for d in data.to_dict(orient='index').values()]
    graph = nx.from_pandas_edgelist(data, args['source'], args['target'], edge_prop_columns)
    pos=nx.spring_layout(graph, dim=3)
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

    weight = np.abs(data[args['values']].tolist())
    degrees = dict(graph.degree())
    colors = []
    sizes = []
    labels = []
    annotations = []
    for node in graph.nodes():
        #colors.append(node_properties[node]["color"])
        size = degrees[node]
        if node in args["node_properties"]:
            if "size" in args["node_properties"][node]:
                size= args["node_properties"][node]["size"]

        label = "{} (degree:{})".format(node, size)
        labels.append(label)
        annotations.append(dict(text=node,
                                x=pos[node][0],
                                y=pos[node][1],
                                z=pos[node][2],
                                ax=0, ay=-0.,
                                font=dict(color= 'black', size=10),
                                showarrow=False))

        if size > 50:
            size = 50
        if size < 3:
            size = 5
        sizes.append(size)


    trace1=Scatter3d(x=Xed,
               y=Yed,
               z=Zed,
               mode='lines',
               opacity = 0.5,
               line=dict(color='rgb(155,155,155)', width=1),
               text=edge_properties,
               hoverinfo='text'
               )
    trace2=Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='proteins',
               marker=scatter3d.Marker(symbol='circle',
                             size=sizes,
                             #color=colors,
                             colorscale='Viridis',
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
         title=args['title'] + "(3D visualization)",
         width=1500,
         height=1500,
         showlegend=True,
         scene=Scene(
                    xaxis=XAxis(axis),
                    yaxis=YAxis(axis),
                    zaxis=ZAxis(axis),
                    annotations= annotations,
                    ),
        margin=Margin(t=1),
        hovermode='closest',
    )

    data=Data([trace1, trace2])
    figure=Figure(data=data, layout=layout)

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
                    align = ['center','center']),
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

def get_violinplot(data, identifier, args):
    df = data.copy()
    graphs = []

    for c in df.columns.unique():
        if c != args['group']:
            traces = create_violinplot(df, c, args['group'])
            figure = {"data": traces,
                    "layout":{
                            "title": "Violinplot per group for variable: "+c,
                            "yaxis": {
                                "zeroline":False,
                                }
                        }
                    }
            graphs.append(dcc.Graph(id=identifier+"_"+c, figure=figure))

    return graphs

def create_violinplot(df, variable, group_col='group'):
    traces = []
    for group in np.unique(df[group_col].values):
        violin = {"type": 'violin',
                    "x": df[group_col][df[group_col] == group],
                    "y": df[variable][df[group_col] == group],
                    "name": group,
                    "box": {
                        "visible": True
                    },
                    "meanline": {
                        "visible": True
                    }
                }
        traces.append(violin)

    return traces

def plot_dendrogram(Z_dendrogram, cutoff_line=True, value=15, orientation='bottom', hang=30, hide_labels=False, labels=None,
                    colorscale=None, hovertext=None, color_threshold=None):

    dendrogram = Dendrogram(Z_dendrogram, orientation, hang, hide_labels, labels,
                             colorscale, hovertext=hovertext, color_threshold=color_threshold)

    if cutoff_line == True:
        dendrogram.layout.update(add_line(dendrogram, value))

    return go.Figure(data=dendrogram.data, layout=dendrogram.layout)

def add_line(plotly_fig, value):
    plotly_fig.layout.update({'shapes':[{'type':'line',
                             'xref':'paper',
                             'yref':'y',
                             'x0':0, 'y0':value,
                             'x1':1, 'y1':value,
                             'line':{'color':'red'}}]})
    return plotly_fig.layout

def get_WGCNAPlots(data, identifier):

    data_exp, data_cli, dissTOM, moduleColors, Features_per_Module, MEs,\
    moduleTraitCor, textMatrix, MM, MMPvalue, FS, FSPvalue, METDiss, METcor = data
    plots = []
    #plot: sample dendrogram and clinical variables heatmap; input: data_exp, data_cli
    plots.append(plot_complex_dendrogram(data_exp, data_cli, title='Clinical variables variation by sample', dendro_labels=data_exp.index, distfun='euclidean', linkagefun='average', hang=40, subplot='heatmap', color_missingvals=True, width=1000, height=800))
    #plot: gene tree dendrogram and module colors; input: dissTOM, moduleColors
    plots.append(plot_complex_dendrogram(dissTOM, moduleColors, title='Co-expression: dendrogram and module colors', dendro_labels=dissTOM.columns, distfun=None, linkagefun='average', hang=0.1, subplot='module colors', col_annotation=True, width=1000, height=800))
    #plot: table with features per module; input: df
    plots.append(getBasicTable(Features_per_Module, identifier='', title='Proteins/Genes module color', colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':1500, 'font':12}, subplot = False))
    #plot: module-traits correlation with annotations; input: moduleTraitCor, textMatrix
    plots.append(plot_labeled_heatmap(moduleTraitCor, textMatrix, title='Module-Clinical variable relationships', colorscale=[[0,'rgb(0,255,0)'],[0.5,'rgb(255,255,255)'],[1,'rgb(255,0,0)']], row_annotation=True, width=1000, height=800))
    #plot: FS vs. MM correlation per trait/module scatter matrix; input: MM, FS, Features_per_Module
    plots.append(plot_intramodular_correlation(MM, FS, Features_per_Module, title='Intramodular analysis: Feature Significance vs. Module Membership', width=2500, height=4000))
    #input: METDiss, METcor
    plots.append(plot_complex_dendrogram(METDiss, METcor, title='Eigengene network and clinical data associations', dendro_labels=METDiss.index, distfun=None, linkagefun='average', hang=0.9,
                             subplot='heatmap', subplot_colorscale=[[0, 'rgb(0,255,0)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']],
                             color_missingvals=False, row_annotation=True, col_annotation=True, width=1000, height=800))

    graphs = []
    for i, j in enumerate(plots):
        graphs.append(dcc.Graph(id=identifier+'_'+str(i), figure=j))

    return graphs


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
