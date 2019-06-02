import os
import numpy as np
import pandas as pd
import ast
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as FF
import math
import random
import dash_table
from plotly import tools
import plotly.io as pio
from scipy.spatial.distance import pdist, squareform
from plotly.graph_objs import *
from kmapper import plotlyviz
import networkx as nx
from clustergrammer2 import net as clustergrammer_net
from pyvis.network import Network as visnet
from webweb import Web
from networkx.readwrite import json_graph
from dash_network import Network
from report_manager import utils, analyses
from wordcloud import WordCloud, STOPWORDS
from report_manager.plots import wgcnaFigures
import dash_cytoscape as cyto

def getPlotTraces(data, key='full', type = 'lines', div_factor=float(10^10000), horizontal=False):
    """
    This function returns traces for different kinds of plots.
    
    Args:
        data: is a Pandas DataFrame with one variable as data.index (i.e. 'x') and all others as columns (i.e. 'y').
        type: 'lines', 'scaled markers', 'bars'.
        div_factor: relative size of the markers.
        horizontal: bar orientation.
    
    Returns:
        List of traces.
    """
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
        fig['layout'].update(height=600, width=1000, title='Distribution plot '+i, annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')])
        graphs.append(dcc.Graph(id=identifier+"_"+i, figure=fig))

    return graphs

def get_barplot(data, identifier, args):
    """
    This function plots a simple barplot.
    
    Args:
        data: is a Pandas DataFrame with three columns: "name" of the bars, 'x' values and 'y' values to plot.
        identifier: is the id used to identify the div where the figure will be generated.
        title: The title of the figure.
    Returns:
        Barplot figure within the <div id="_dash-app-content"> .
    """
    figure = {}
    figure["data"] = []
    if "group" in args:
        for g in data[args["group"]].unique():
            color = None
            if 'colors' in args:
                if g in args['colors']:
                    color = args['colors'][g]
            trace = go.Bar(
                        x = data.loc[data[args["group"]] == g,args['x']], # assign x as the dataframe column 'x'
                        y = data.loc[data[args["group"]] == g, args['y']],
                        name = g,
                        marker = dict(color=color)
                        )
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
                            height = args['height'],
                            width = args['width'],
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')]
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
    figure['layout'] = dict(title = args['title'].title(),
                            paper_bgcolor = None,
                            legend = None,
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')])

    return dcc.Graph(id= identifier, figure = figure)

def get_ranking_plot(data, identifier, args):
    num_cols = 3
    fig = {}
    layouts = []
    num_groups = len(data.index.unique())
    num_rows = math.ceil(num_groups/num_cols)
    if 'group' in args:
        group=args['group']
    #subplot_title = "Ranking of proteins in {} samples"
    #subplot_titles = [subplot_title.format(index.title()) for index in data.index.unique()]
    fig = tools.make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=True,print_grid=False)
    if 'index' in args and args['index']:
        r = 1
        c = 1
        range_y = [data['y'].min(), data['y'].max()+1]
        for index in data.index.unique():
            gdata = data.loc[index, :].dropna().groupby('name', as_index=False).mean().sort_values(by='y', ascending=False)
            gdata = gdata.reset_index().reset_index()
            cols = ['x', 'group', 'name', 'y']
            cols.extend(gdata.columns[4:])
            gdata.columns = cols
            gfig = get_simple_scatterplot(gdata, identifier+'_'+str(index), args)
            trace = gfig.figure['data'].pop()
            glayout = gfig.figure['layout']['annotations']

            for l in glayout:
                nlayout = dict(x = l.x,
                            y = l.y,
                            xref = 'x'+str(c),
                            yref = 'y'+str(r),
                            text = l.text,
                            showarrow = True,
                            ax = l.ax,
                            ay = l.ay,
                            font = l.font,
                            align='center',
                            arrowhead=1,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='#636363')
                layouts.append(nlayout)
            trace.name = index
            fig.append_trace(trace, r, c)
            
            if c >= num_cols:
                r += 1
                c = 1
            else:
                c += 1
        fig['layout'].update(dict(height = args['height'], 
                                width=args['width'],  
                                title=args['title'], 
                                xaxis= {"title": args['x_title'], 'autorange':True}, 
                                yaxis= {"title": args['y_title'], 'range':range_y}))
        fig['layout'].annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')] + layouts 
    else:
        fig = get_simple_scatterplot(data, identifier+'_'+group, args).figure
    return dcc.Graph(id=identifier, figure=fig)
    

def get_scatterplot_matrix(data, identifier, args):
    df = data.copy()
    if "format" in args:
        if args["format"] == "long":
            columns = [args["variables"], args["values"]]
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
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')]
                            )

    return dcc.Graph(id=identifier, figure=figure)

def get_simple_scatterplot(data, identifier, args):
    figure = {}
    m = {'size': 15, 'line': {'width': 0.5, 'color': 'grey'}}
    text = data.name
    if 'colors' in data.columns:
        m.update({'color':data['colors'].tolist()})
    if 'size' in data.columns:
        m.update({'size':data['size'].tolist()})
    if 'symbol' in data.columns:
        m.update({'symbol':data['symbol'].tolist()})
    
    annots=[]
    if 'annotations' in args:
        for index, row in data.iterrows():
            name = row['name'].split(' ')[0]
            if name in args['annotations']:
                annots.append({'x': row['x'], 
                            'y': row['y'], 
                            'xref':'x', 
                            'yref': 'y', 
                            'text': name, 
                            'showarrow': False, 
                            'ax': 55, 
                            'ay': -1,
                            'font': dict(size = 8)})
    figure['data'] = [go.Scatter(x = data.x,
                                y = data.y,
                                text = text,
                                mode = 'markers',
                                opacity=0.7,
                                marker= m,
                                )]
                                
    figure["layout"] = go.Layout(title = args['title'],
                                xaxis= {"title": args['x_title']},
                                yaxis= {"title": args['y_title']},
                                margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest',
                                height=args['height'],
                                width=args['width'],
                                annotations = annots + [dict(xref='paper', yref='paper', showarrow=False, text='')],
                                showlegend=False
                                )
    
    return dcc.Graph(id= identifier, figure = figure)


def get_scatterplot(data, identifier, args):
    """
    This function plots a simple Scatterplot.
    
    Args:
        data: is a Pandas DataFrame with four columns: "name", x values and y values (provided as variables) to plot.
        identifier: is the id used to identify the div where the figure will be generated.
        title: The title of the figure.
    
    Returns:
        Scatterplot figure within the <div id="_dash-app-content"> .
    """
    figure = {}
    figure["data"] = []
    figure["layout"] = go.Layout(title = args['title'],
                                xaxis= {"title": args['x_title']},
                                yaxis= {"title": args['y_title']},
                                margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': -.4, 'y': 1.2},
                                hovermode='closest',
                                height=args['height'],
                                width=args['width'],
                                annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')]
                                )
    for name in data.name.unique():
        m = {'size': 25, 'line': {'width': 0.5, 'color': 'grey'}}
        if 'colors' in args:
            if name in args['colors']:
                m.update({'color' : args['colors'][name]})
        figure["data"].append(go.Scatter(x = data.loc[data["name"] == name, "x"],
                                        y = data.loc[data['name'] == name, "y"],
                                        text = name,
                                        mode = 'markers',
                                        opacity=0.7,
                                        marker= m,
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
            range_y = [0,max(abs(result['y']))+0.8]
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
                                                'x0': np.log2(args['fc']),
                                                'y0': 0,
                                                'x1': np.log2(args['fc']),
                                                'y1': range_y[1],
                                                'line': {
                                                    'color': 'grey',
                                                    'width': 2,
                                                    'dash':'dashdot'
                                                    },
                                                    },
                                                {'type': 'line',
                                                'x0': -np.log2(args['fc']),
                                                'y0': 0,
                                                'x1': -np.log2(args['fc']),
                                                'y1': range_y[1],
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
                                        annotations = result['annotations']+[dict(xref='paper', yref='paper', showarrow=False, text='')],
                                        showlegend=False)

        figures.append(dcc.Graph(id= identifier, figure = figure))
    return figures

def run_volcano(data, identifier, args={'alpha':0.05, 'fc':2, 'colorscale':'Blues', 'showscale': False, 'marker_size':6, 'x_title':'log2FC', 'y_title':'-log10(pvalue)', 'num_annotations':10}):
    # Loop through signature

    volcano_plot_results = {}
    grouping = data.groupby(['group1','group2'])
    for group in grouping.groups:
        signature = grouping.get_group(group)
        color = []
        text = []
        annotations = []
        num_annotations = args['num_annotations'] if 'num_annotations' in args else 10
        gidentifier = identifier + "_".join(map(str,group))
        title = 'Comparison: '+str(group[1])+' vs '+str(group[0])
        sig_pval = False
        signature = signature.sort_values(by="padj",ascending=True)
        for index, row in signature.iterrows():
            # Text
            text.append('<b>'+str(row['identifier'])+": "+str(index)+'<br>Comparison: '+str(row['group2'])+' vs '+str(row['group1'])+'<br>log2FC = '+str(round(row['log2FC'], ndigits=2))+'<br>p = '+'{:.2e}'.format(row['pvalue'])+'<br>FDR = '+'{:.2e}'.format(row['padj']))

            # Color
            if row['padj'] <= args['alpha']:
                sig_pval = True
                if row['FC'] <= -args['fc']:
                    annotations.append({'x': row['log2FC'], 
                                    'y': row['-log10 pvalue'], 
                                    'xref':'x', 
                                    'yref': 'y', 
                                    'text': str(row['identifier']), 
                                    'showarrow': False, 
                                    'ax': 0, 
                                    'ay': -10,
                                    'font': dict(color = "#2c7bb6", size = 7)})
                    color.append('#2c7bb6')
                elif row['FC'] >= args['fc']:
                    annotations.append({'x': row['log2FC'], 
                                    'y': row['-log10 pvalue'], 
                                    'xref':'x', 
                                    'yref': 'y', 
                                    'text': str(row['identifier']), 
                                    'showarrow': False, 
                                    'ax': 0, 
                                    'ay': -10,
                                    'font': dict(color = "#d7191c", size = 7)})
                    color.append('#d7191c')
                elif row['FC'] < -1.:
                    color.append('#abd9e9')
                elif row['FC'] > 1.:
                    color.append('#fdae61')
                else:
                    color.append('lightblue')
            else:
                color.append('silver')

        if not sig_pval:
            alpha = 0.0000000000001
        else:
            alpha = args['alpha']
        
        if len(annotations) < num_annotations:
            num_annotations = len(annotations)

        # Return
        volcano_plot_results[(gidentifier, title)] = {'x': signature['log2FC'].values, 'y': signature['-log10 pvalue'].values, 'text':text, 'color': color, 'pvalue':-np.log10(alpha), 'annotations':annotations[0:num_annotations]}

    figures = get_volcanoplot(volcano_plot_results, args)

    return figures

def get_heatmapplot(data, identifier, args):
    """
    This function plots a simple Heatmap.
    
    Args:
        data: is a Pandas DataFrame with the shape of the heatmap where index corresponds to rows
              and column names corresponds to columns, values in the heatmap corresponds to the row values.
        identifier: is the id used to identify the div where the figure will be generated.
        title: The title of the figure.
    
    Returns:
        Heatmap figure within the <div id="_dash-app-content">.
    """
    df = data.copy()
    if args['format'] == "edgelist":
        df = df.set_index(args['source'])
        df = df.pivot_table(values=args['values'], index=df.index, columns=args['target'], aggfunc='first')
        df = df.fillna(0)
    figure = {}
    figure["data"] = []
    figure["layout"] = {"title":args['title'],
                        "height": 500,
                        "width": 700,
                        "annotations" : [dict(xref='paper', yref='paper', showarrow=False, text='')]}
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
                                       'ticks':""}, 
                                       'annotations': [dict(xref='paper', yref='paper', showarrow=False, text='')]
                                       })


    return dcc.Graph(id=identifier, figure=figure)

def get_notebook_network_pyvis(graph, args):
    notebook_net = visnet(args['width'], args['height'], notebook=True)
    notebook_net.barnes_hut(overlap=0.8)
    notebook_net.from_nx(graph)
    notebook_net.show_buttons(['nodes', 'edges', 'physics'])
    notebook_net.generate_html(notebook=True)

    return notebook_net

def get_notebook_network_web(graph, args):
    notebook_net = Web(nx.to_numpy_matrix(graph).tolist())
    notebook_net.display.scaleLinkWidth = True

    return notebook_net

def network_to_tables(graph):
    edges_table = nx.to_pandas_edgelist(graph)
    nodes_table = pd.DataFrame.from_dict(dict(graph.nodes(data=True))).transpose().reset_index()

    return nodes_table, edges_table


def get_network(data, identifier, args):
    net = None
    if 'cutoff_abs' not in args:
        args['cutoff_abs'] = False
    if not data.empty:
        if 'cutoff' in args:
            if args['cutoff_abs']:
                data = data[np.abs(data[args['values']]) > args['cutoff']]
            else:
                data = data > arg['cutoff']
        data = data.rename(index=str, columns={args['values']: "width"})
        edge_prop_columns = [c for c in data.columns if c not in [args['source'], args['target']]]
        edge_properties = [str(d) for d in data.to_dict(orient='index').values()]
        graph = nx.from_pandas_edgelist(data, args['source'], args['target'], edge_attr=True)
        

        degrees = dict(graph.degree())
        nx.set_node_attributes(graph, degrees, 'degree')
        betweenness = None
        ev_centrality = None
        if data.shape[0] < 100:
            betweenness = nx.betweenness_centrality(graph, weight='width')
            ev_centrality = nx.eigenvector_centrality_numpy(graph)
            nx.set_node_attributes(graph, betweenness, 'betweenness centrality')
            nx.set_node_attributes(graph, ev_centrality, 'eigenvector centrality')

        if args['node_size'] == 'betweenness' and betweeness is not None:
            nx.set_node_attributes(graph, betweenness, 'radius')
        elif args['node_size'] == 'ev_centrality' and ev_centrality is not None:
            nx.set_node_attributes(graph, ev_centrality, 'radius')
        elif args['node_size'] == 'degree':
            nx.set_node_attributes(graph, degrees, 'radius')

        clusters = analyses.basicAnalysis.get_network_communities(graph, args)
        col = utils.get_hex_colors(len(set(clusters.values())))
        colors = {n:col[clusters[n]] for n in clusters}
        nx.set_node_attributes(graph, colors, 'color')
        nx.set_node_attributes(graph, clusters, 'cluster')
        notebook_net = get_notebook_network_pyvis(graph, args)
        nodes_table, edges_table = network_to_tables(graph)
        nodes_fig_table = getBasicTable(nodes_table, identifier=identifier+"_nodes_table", title=args['title']+" nodes table")
        edges_fig_table = getBasicTable(edges_table, identifier=identifier+"_edges_table", title=args['title']+" edges table")
        
        stylesheet, layout = get_network_style(colors, args['color_weight'])
        args['stylesheet'] = stylesheet
        args['layout'] = layout
        
        cy_graph = json_graph.cytoscape_data(graph)
        cy_nodes = cy_graph['elements']['nodes']
        cy_edges = cy_graph['elements']['edges']
        cy_elements = cy_nodes
        cy_elements.extend(cy_edges)

        net = {"notebook":notebook_net, "app":get_cytoscape_network(cy_elements, identifier, args), "net_tables":(nodes_fig_table, edges_fig_table)}
    return net

def get_network_style(node_colors, color_edges):
    color_selector = "{'selector': '[name = \"KEY\"]', 'style': {'background-color': 'VALUE'}}"
    stylesheet=[{'selector': 'node', 'style': {'label': 'data(name)'}}, 
                {'selector':'edge','style':{'curve-style': 'bezier'}}]

    layout = {'name': 'cose',
                'idealEdgeLength': 100,
                'nodeOverlap': 20,
                'refresh': 20,
                'fit': True,
                'padding': 30,
                'randomize': False,
                'componentSpacing': 100,
                'nodeRepulsion': 400000,
                'edgeElasticity': 100,
                'nestingFactor': 5,
                'gravity': 80,
                'numIter': 1000,
                'initialTemp': 200,
                'coolingFactor': 0.95,
                'minTemp': 1.0}

    if color_edges:
        stylesheet.extend([{'selector':'[width < 0]', 'style':{'line-color':'#3288bd'}},{'selector':'[width > 0]', 'style':{'line-color':'#d73027'}}])

    for k,v in node_colors.items():
        stylesheet.append(ast.literal_eval(color_selector.replace("KEY", k).replace("VALUE",v)))

    return stylesheet, layout

def get_pca_plot(data, identifier, args):
    pca_data, loadings = data    
    figure = {}
    traces = []
    annotations = []
    sct = get_scatterplot(pca_data, identifier, args).figure
    traces.extend(sct['data'])
    figure['layout'] = sct['layout']
    for index in list(loadings.index)[0:args['loadings']]:
        x = loadings.loc[index,'x'] * 5 
        y = loadings.loc[index, 'y'] * 5
        value = loadings.loc[index, 'value']

        trace = go.Scatter(x= [0,x],
                        y = [0,y],
                        mode='markers+lines',
                        text=index+" loading: {0:.2f}".format(value),
                        name = index,
                        marker= dict(size=3,
                                    symbol= 1,
                                    color = 'darkgrey', #set color equal to a variable
                                    showscale=False,
                                    opacity=0.7,
                                    ),
                        showlegend=False,
                        )
        annotation = dict( x=x,
                        y=y,
                        xref='x',
                        yref='y',
                        text=index,
                        showarrow=False,
                        font=dict(
                                size=10,
                                color='darkgrey'
                            ),
                            align='center',
                            ax=20,
                            ay=-30,
                            )
        annotations.append(annotation) 
        traces.append(trace)

    figure['data'] = traces   
    figure['layout'].annotations = annotations

    return  dcc.Graph(id = identifier, figure = figure)


def getSankeyPlot(data, identifier, args={'source':'source', 'target':'target', 'weight':'weight','source_colors':'source_colors', 'target_colors':'target_colors', 'orientation': 'h', 'valueformat': '.0f', 'width':800, 'height':800, 'font':12, 'title':'Sankey plot'}):
    """
    This function generates a Sankey plot in Plotly.
    
    Args:
        data: Pandas DataFrame with the format: source  target  weight.
        identifier: id for the web app.
            args: dictionary with the following items:
                - source: name of the column containing the source 
                - target: name of the column containing the target 
                - weight: name of the column containing the weight 
                - source_colors: name of the column in data that contains the colors of each source item
                - target_colors: name of the column in data that contains the colors of each target item
                - title: Plot title 
                - orientation: whether to plot horizontal ('h') or vertical ('v')
                - valueformat: how to show the value ('.0f')
                - width: plot width
                - height: plot height
                - font: font size
    
    Returns:
        dcc.Graph.
    """
    nodes = list(set(data[args['source']].tolist() + data[args['target']].tolist()))
    if 'source_colors' in args:
        node_colors = dict(zip(data[args['source']],data[args['source_colors']]))
    else:
        scolors = ['blue']*len(data['source'].tolist())
        node_colors = dict(zip(data[args['source']],scolors))
        args['source_colors'] = 'source_colors'
        data['source_colors'] = scolors
    if 'target_colors' in args:
        node_colors.update(dict(zip(data[args['target']],data[args['target_colors']])))
    else:
        node_colors.update(dict(zip(data[args['target']],['red']*len(data['target'].tolist()))))
    data_trace = dict(type='sankey',
                        #domain = dict(x =  [0,1], y =  [0,1]),
                        orientation = 'h' if 'orientation' not in args else args['orientation'],
                        valueformat = ".0f" if 'valueformat' not in args else args['valueformat'],
                        arrangement = 'freeform',
                        node = dict(pad = 25 if 'pad' not in args else args['pad'],
                                    thickness = 25 if 'thickness' not in args else args['thickness'],
                                    line = dict(color = "black", width = 0.3),
                                    label =  nodes,
                                    color =  ["rgba"+str(utils.hex2rgb(node_colors[c])) if node_colors[c].startswith('#') else node_colors[c] for c in nodes]
                                    ),
                        link = dict(source = [list(nodes).index(i) for i in data[args['source']].tolist()],
                                    target = [list(nodes).index(i) for i in data[args['target']].tolist()],
                                    value =  data[args['weight']].tolist(),
                                    color = ["rgba"+str(utils.hex2rgb(c)) if c.startswith('#') else c for c in data[args['source_colors']].tolist()]
                                    ))
    layout =  dict(
        width= 800 if 'width' not in args else args['width'],
        height= 800 if 'height' not in args else args['height'],
        title = args['title'],
        annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
        font = dict(
            size = 12 if 'font' not in args else args['font'],
        )
    )

    figure = dict(data=[data_trace], layout=layout)

    return dcc.Graph(id = identifier, figure = figure)

def getBasicTable(data, identifier, title, colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':2500, 'font':12}, subplot = False):
    if subset is not None:
        data = data[subset]

    #booleanDictionary = {True: 'TRUE', False: 'FALSE'}
    #if 'rejected' in data.columns:
    #    data['rejected'] = data['rejected'].replace(booleanDictionary)
    
    data_trace = dash_table.DataTable(id='table_'+identifier,
                                        data=data.to_dict("rows"),
                                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in data.columns],
                                        css=[{
                                            'selector': '.dash-cell div.dash-cell-value',
                                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                        }],
                                        style_data={'whiteSpace': 'normal'},
                                        style_cell={
                                            'minWidth': '50px', 'maxWidth': '180px',
                                            'textAlign': 'left', 'padding': '1px', 'vertical-align': 'top'
                                        },
                                        style_table={
                                            'maxHeight': '800',
                                            'overflowY': 'scroll',
                                            'overflowX': 'scroll'
                                        },
                                        style_header={
                                            'backgroundColor': '#2b8cbe',
                                            'fontWeight': 'bold',
                                            'position': 'sticky'
                                        },
                                        style_data_conditional=[{
                                            "if": 
                                                {"column_id": "rejected", "filter": 'rejected eq "TRUE"'},
                                                "backgroundColor": "#3B8861",
                                                'color': 'white'
                                            },
                                            ],
                                        n_fixed_rows=1,
                                        sorting=True,
                                        )

    return html.Div([html.H2(title),data_trace])

def get_violinplot(data, identifier, args):
    df = data.copy()
    graphs = []

    for c in df.columns.unique():
        if c != args['group']:
            traces = create_violinplot(df, c, args['group'])
            figure = {"data": traces,
                    "layout":{
                            "title": "Violinplot per group for variable: "+c,
                            "annotations": [dict(xref='paper', yref='paper', showarrow=False, text='')],
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


def get_clustergrammer_plot(df, identifier, args):
    div = None
    if not df.empty:
        if 'format' in args:
            if args['format'] == 'edgelist':
                df = df[['node1', 'node2', 'weight']].pivot(index='node1', columns='node2') 
        clustergrammer_net.load_df(df)

        link = utils.get_clustergrammer_link(clustergrammer_net, filename=None)

        iframe = html.Iframe(src=link, width=1000, height=900)

        div = html.Div([html.H2(args['title']),iframe])
    return div

def get_parallel_plot(data, identifier, args):
    lines = []
    for col in data:
        for group in data[col]:

            pass


    trace = [go.Parcoords(
                line = dict(color = df[args['group']],
                #colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']]),
                dimensions = list([
                    dict(range = [0,8],
                        constraintrange = [4,8],
                        label = 'Sepal Length', values = df['sepal_length']),
                    dict(range = [0,8],
                        label = 'Sepal Width', values = df['sepal_width']),
                    dict(range = [0,8],
                        label = 'Petal Length', values = df['petal_length']),
                    dict(range = [0,8],
                        label = 'Petal Width', values = df['petal_width'])        
                    ])
                ))
    ]
    
    layout = go.Layout(
        plot_bgcolor = '#E5E5E5',
        paper_bgcolor = '#E5E5E5'
        )
    
    fig = go.Figure(data = data, layout = layout)

    return dcc.Graph(id=identifier, figure=fig)

def get_WGCNAPlots(data, identifier):
    """ 
    Takes data from runWGCNA function and builds WGCNA plots: 

    Args:
        data: tuple with multiple pandas dataframes.
        identifier: is the id used to identify the div where the figure will be generated.
    
    Returns:
        List of dcc.Graph.
    """
    graphs = []
    if data is not None:
        data_exp, data_cli, dissTOM, moduleColors, Features_per_Module, MEs,\
        moduleTraitCor, textMatrix, MM, MMPvalue, FS, FSPvalue, METDiss, METcor = data
        plots = []
        #plot: sample dendrogram and clinical variables heatmap; input: data_exp, data_cli
        plots.append(wgcnaFigures.plot_complex_dendrogram(data_exp, data_cli, title='Clinical variables variation by sample', dendro_labels=data_exp.index, distfun='euclidean', linkagefun='average', hang=40, subplot='heatmap', color_missingvals=True, width=1000, height=800))
        #plot: gene tree dendrogram and module colors; input: dissTOM, moduleColors
        plots.append(wgcnaFigures.plot_complex_dendrogram(dissTOM, moduleColors, title='Co-expression: dendrogram and module colors', dendro_labels=dissTOM.columns, distfun=None, linkagefun='average', hang=0.1, subplot='module colors', col_annotation=True, width=1000, height=800))
        #plot: table with features per module; input: df
        plots.append(getBasicTable(Features_per_Module, identifier='', title='Proteins/Genes module color', colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':1500, 'font':12}, subplot = False))
        #plot: module-traits correlation with annotations; input: moduleTraitCor, textMatrix
        plots.append(wgcnaFigures.plot_labeled_heatmap(moduleTraitCor, textMatrix, title='Module-Clinical variable relationships', colorscale=[[0,'#67a9cf'],[0.5,'#f7f7f7'],[1,'#ef8a62']], row_annotation=True, width=1000, height=800))
        #plot: FS vs. MM correlation per trait/module scatter matrix; input: MM, FS, Features_per_Module
        plots.append(wgcnaFigures.plot_intramodular_correlation(MM, FS, Features_per_Module, title='Intramodular analysis: Feature Significance vs. Module Membership', width=1500, height=4000))
        #input: METDiss, METcor
        plots.append(wgcnaFigures.plot_complex_dendrogram(METDiss, METcor, title='Eigengene network and clinical data associations', dendro_labels=METDiss.index, distfun=None, linkagefun='average', hang=0.9,
                                 subplot='heatmap', subplot_colorscale=[[0,'#67a9cf'],[0.5,'#f7f7f7'],[1,'#ef8a62']],
                                 color_missingvals=False, row_annotation=True, col_annotation=True, width=1000, height=800))

        graphs = []
        for i, j in enumerate(plots):
            if isinstance(j, html.Div):
                graphs.append(j)
            else:
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

def get_2_venn_diagram(data, identifier, cond1, cond2, args):
    figure = {}
    figure["data"] = []
    total = len(set(data[cond1].dropna().index).union(set(data[cond2].dropna().index)))
    unique1 = len(set(data[cond1].dropna().index).difference(data[cond2].dropna().index))#/total
    unique2 = len(set(data[cond2].dropna().index).difference(data[cond1].dropna().index))#/total
    intersection12 = len(set(data[cond1].dropna().index).intersection(data[cond2].dropna().index))#/total

    figure["data"] = [go.Scatter(
        x=[1, 1.75, 2.5],
        y=[1, 1, 1],
        text=[cond1+": "+str(unique1), str(intersection12), cond2+": "+str(unique2)],
        mode='text',
        textfont=dict(
            color='black',
            size=18,
            family='Arial',
        )
    )]


    figure["layout"] = {
        'xaxis': {
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
        },
        'yaxis': {
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
        },
        'shapes': [
            {
                'opacity': 0.3,
                'xref': 'x',
                'yref': 'y',
                'fillcolor': args['colors'][cond1],
                'x0': 0,
                'y0': 0,
                'x1': 2,
                'y1': 2,
                'type': 'circle',
                'line': {
                    'color': args['colors'][cond1],
                },
            },
            {
                'opacity': 0.3,
                'xref': 'x',
                'yref': 'y',
                'fillcolor': args['colors'][cond2],
                'x0': 1.5,
                'y0': 0,
                'x1': 3.5,
                'y1': 2,
                'type': 'circle',
                'line': {
                    'color': args['colors'][cond2],
                },
            }
        ],
        'margin': {
            'l': 20,
            'r': 20,
            'b': 100
        },
        'height': 600,
        'width': 800,
    }

    return dcc.Graph(id = identifier, figure=figure)

def get_wordcloud(text, identifier, args={'stopwords':[], 'max_words': 400, 'max_font_size': 100, 'width':700, 'height':700, 'margin': 1}):
    stopwords = set(STOPWORDS)
    if 'stopwords' in args:
        stopwords = stopwords.union(args['stopwords'])
    wc = WordCloud(stopwords = stopwords,
                   max_words = args['max_words'],
                   max_font_size = args['max_font_size'],
                   background_color='white',
                   margin=args['margin'])
    wc.generate(text)

    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x=[]
    y=[]
    j = 0
    for i in position_list:
        x.append(i[1]+fontsize_list[j]+10)
        y.append(i[0]+5)
        j += 1

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*70)
    new_freq_list

    trace = go.Scatter(x=x,
                       y=y,
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hoverinfo='text',
                       hovertext=['{0} freq: {1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode="text",
                       text=word_list
                      )

    layout = go.Layout(
                       xaxis=dict(showgrid=False,
                                  showticklabels=False,
                                  zeroline=False,
                                  automargin=True),
                       yaxis=dict(showgrid=False,
                                  showticklabels=False,
                                  zeroline=False,
                                  automargin=True),
                      width=args['width'],
                      height=args['height'],
                      title=args['title'],
                      annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                      )

    figure = go.Figure(data=[trace], layout=layout)

    return dcc.Graph(id = identifier, figure=figure)


def get_cytoscape_network(net, identifier, args):
    cytonet = html.Div([html.H2(args['title']), cyto.Cytoscape(id=identifier,
                                    stylesheet=args['stylesheet'],
                                    elements=net,
                                    layout=args['layout'],
                                    minZoom = 0.2,
                                    maxZoom = 1.8,
                                    style={'width': '100%', 'height': '500px'}
                                    )
                    ])


    return cytonet

def save_DASH_plot(plot, name, plot_format='svg', directory='.'):
    if not os.path.exists(directory):
        os.mkdir(directory)
    if plot_format in ['svg', 'pdf', 'png', 'jpeg', 'jpg']:
        plot_file = os.path.join(directory, str(name)+'.'+str(plot_format))
        if hasattr(plot, 'figure'):
            pio.write_image(plot.figure, plot_file)
        else:
            pio.write_image(plot, plot_file)
        
    
    
    
