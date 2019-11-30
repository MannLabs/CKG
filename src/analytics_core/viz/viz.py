import os
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as FF
import math
import random
import dash_table
import plotly.subplots as tools
import plotly.io as pio
from scipy.spatial.distance import pdist, squareform
# import dash_bio as dashbio
from scipy.stats import zscore
import networkx as nx
from cyjupyter import Cytoscape
from pyvis.network import Network as visnet
from webweb import Web
from networkx.readwrite import json_graph
from analytics_core import utils
from analytics_core.analytics import analytics
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import nltk


from analytics_core.analytics import wgcnaAnalysis
from analytics_core.viz import wgcnaFigures, Dendrogram
import dash_cytoscape as cyto

def getPlotTraces(data, key='full', type = 'lines', div_factor=float(10^10000), horizontal=False):
    """
    This function returns traces for different kinds of plots.

    :param data: Pandas DataFrame with one variable as data.index (i.e. 'x') and all others as columns (i.e. 'y').
    :param str type: 'lines', 'scaled markers', 'bars'.
    :param float div_factor: relative size of the markers.
    :param bool horizontal: bar orientation.
    :return: list of traces.

    Exmaple 1::

        result = getPlotTraces(data, key='full', type = 'lines', horizontal=False)

    Example 2::

        result = getPlotTraces(data, key='full', type = 'scaled markers', div_factor=float(10^3000), horizontal=True)
    """
    if type == 'lines':
        traces = [go.Scattergl(x=data.index, y=data[col], name = col+' '+key, mode='markers+lines') for col in data.columns]

    elif type == 'scaled markers':
        traces = [go.Scattergl(x = data.index, y = data[col], name = col+' '+key, mode = 'markers', marker = dict(size = data[col].values/div_factor, sizemode = 'area')) for col in data.columns]

    elif type == 'bars':
        traces = [go.Bar(x = data.index, y = data[col], orientation = 'v', name = col+' '+key) for col in data.columns]
        if horizontal == True:
            traces = [go.Bar(x = data[col], y = data.index, orientation = 'h', name = col+' '+key) for col in data.columns]

    else: return 'Option not found'

    return traces

def get_markdown(text, args={}):
    """
    Converts a given text into a Dash Markdown component. It includes a syntax for things like bold text and italics, links, inline code snippets, lists, quotes, and more.
    For more information visit https://dash.plot.ly/dash-core-components/markdown.

    :param str text: markdown string (or array of strings) that adhreres to the CommonMark spec.
    :param dict args: dictionary with items from https://dash.plot.ly/dash-core-components/markdown.
    :return: dash Markdown component.
    """
    mkdown = dcc.Markdown(text)

    return mkdown

def get_pieplot(data, identifier, args):
    """
    This function plots a simple Pie plot.

    :param data: pandas DataFrame with values to plot as columns and labels as index.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **valueCol** (str) -- name of the column with the values to be plotted.
        * **textCol** (str) -- name of the column containing information for the hoverinfo parameter.
        * **height** (str) -- height of the plot.
        * **width** (str) -- width of the plot.
    :return: Pieplot figure within the <div id="_dash-app-content">.
    """
    figure = {}
    figure['data'] = []
    figure['data'].append(go.Pie(labels=data.index, values=data[args['valueCol']], hovertext=data[args['textCol']], hoverinfo='label+text+percent'))
    figure["layout"] = go.Layout(height = args['height'],
                            width = args['width'],
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            template='plotly_white')

    return dcc.Graph(id = identifier, figure = figure)

def get_distplot(data, identifier, args):
    """

    :param data:
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **group** (str) -- name of the column containing the group.

    """
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
        fig['layout'].update(height=600, width=1000, title='Distribution plot '+i, annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')], template='plotly_white')
        graphs.append(dcc.Graph(id=identifier+"_"+i, figure=fig))

    return graphs

def get_barplot(data, identifier, args):
    """
    This function plots a simple barplot.

    :param data: pandas DataFrame with three columns: 'name' of the bars, 'x' values and 'y' values to plot.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **title** (str) -- plot title.
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **height** (str) -- plot height.
        * **width** (str) -- plot width.
    :return: barplot figure within the <div id="_dash-app-content">.

    Example::

        result = get_barplot(data, identifier='barplot', args={'title':'Figure with Barplot'})
    """
    figure = {}
    figure["data"] = []
    if "group" in args:
        for g in data[args["group"]].unique():
            color = None
            if 'colors' in args:
                if g in args['colors']:
                    color = args['colors'][g]
            errors = []
            if 'errors' in args:
                errors = data.loc[data[args["group"]] == g, args['errors']]
            #errors = data.groupby(args["group"]).agg({args['y']:'std'})
            trace = go.Bar(
                        x = data.loc[data[args["group"]] == g,args['x']], # assign x as the dataframe column 'x'
                        y = data.loc[data[args["group"]] == g, args['y']],
                        error_y = dict(type='data',array=errors),
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
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            template='plotly_white'
                        )

    return dcc.Graph(id= identifier, figure = figure)

##ToDo
def get_facet_grid_plot(data, identifier, args):
    """
    This function plots a scatterplot matrix where we can plot one variable against another to form a regular scatter plot, and we can pick a third faceting variable
    to form panels along the columns to segment the data even further, forming a bunch of vertical panels. For more information visit https://plot.ly/python/facet-trellis/.

    :param data: pandas dataframe with format: 'group', 'name', 'type', and 'x' and 'y' values to be plotted.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param ditc args: see below.
    :Arguments:
        * **x** (str) -- name of the column containing values to plot in the x axis.
        * **y** (str) -- name of the column containing values to plot in the y axis.
        * **group** (str) -- name of the column containing the group.
        * **class** (str) -- name of the column to be used as 'facet' column.
        * **plot_type** (str) -- decides the type of plot to appear in the facet grid. The options are 'scatter', 'scattergl', 'histogram', 'bar', and 'box'.
        * **title** (str) -- plot title.
    :return: facet grid figure within the <div id="_dash-app-content">.

    Example::

        result = get_facet_grid_plot(data, identifier='facet_grid', args={'x':'a', 'y':'b', 'group':'group', 'class':'type', 'plot_type':'bar', 'title':'Facet Grid Plot'})
    """
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
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            template='plotly_white')

    return dcc.Graph(id= identifier, figure = figure)

def get_ranking_plot(data, identifier, args):
    """
    Creates abundance multiplots (one per sample group).

    :param data: long-format pandas dataframe with group as index, 'name' (protein identifiers) and 'y' (LFQ intensities) as columns.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below
    :Arguments:
        * **group** (str) -- name of the column containing the group.
        * **index** (bool) -- set to True when multi samples per group. Calculates the mean intensity for each protein in each group.
        * **x_title** (str) -- title of plot x axis.
        * **y_title** (str) -- title of plot y axis.
        * **title** (str) -- plot title.
        * **width** (int) -- plot width.
        * **height** (int) -- plot height.
        * **annotations** (dict, optional) -- dictionary where data points names are the keys and descriptions are the values.
    :return: multi abundance plot figure within the <div id="_dash-app-content">.

    Example::

        result = get_ranking_plot(data, identifier='ranking', args={'group':'group', 'index':'', 'x_title':'x_axis', 'y_title':'y_axis', \
                                    'title':'Ranking Plot', 'width':100, 'height':150, 'annotations':{'GPT~P24298': 'liver disease', 'CP~P00450': 'Wilson disease'}})
    """
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
        i = 0
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
                            xref = 'x'+str(i+1),
                            yref = 'y'+str(i+1),
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
            i += 1
        fig['layout'].update(dict(height = args['height'],
                                width=args['width'],
                                title=args['title'],
                                xaxis= {"title": args['x_title'], 'autorange':True},
                                yaxis= {"title": args['y_title'], 'range':range_y},
                                template='plotly_white'))
        fig['layout'].annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')] + layouts
    else:
        fig = get_simple_scatterplot(data, identifier+'_'+group, args).figure
    return dcc.Graph(id=identifier, figure=fig)

def get_scatterplot_matrix(data, identifier, args):
    """
    This function pltos a multi scatterplot (one for each unique element in args['group']).

    :param data: pandas dataframe with four columns: 'name' of the data points, 'x' and 'y' values to plot, and 'group' they belong to.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below
    :Arguments:
        * **group** (str) -- name of the column containing the group.
        * **title** (str) -- plot title.
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **height** (int) -- plot height.
        * **width** (int) -- plot width.
        * **annotations** (dict, optional) -- dictionary where data points names are the keys and descriptions are the values.
    :return: multi scatterplot figure within the <div id="_dash-app-content">.

    Example::

        result = get_scatterplot_matrix(data, identifier='scatter matrix', args={'group':'group', 'title':'Scatter Plot Matrix', 'x_title':'x_axis', \
                                        'y_title':'y_axis', 'height':100, 'width':100, 'annotations':{'GPT~P24298': 'liver disease', 'CP~P00450': 'Wilson disease'}})
    """
    num_cols = 3
    fig = {}
    if 'group' in args:
        group=args['group']

    num_groups = len(data[group].unique())
    num_rows = math.ceil(num_groups/num_cols)
    fig = tools.make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=True,print_grid=False)
    r = 1
    c = 1
    range_y = [data['y'].min(), data['y'].max()+1]
    for g in data[group].unique():
        gdata = data[data[group] == g].dropna()
        gfig = get_simple_scatterplot(gdata, identifier+'_'+str(g), args)
        trace = gfig.figure['data'].pop()
        trace.name = g
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
                            yaxis= {"title": args['y_title'], 'range':range_y},
                            template='plotly_white'))

    fig['layout'].annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')]

    return dcc.Graph(id=identifier, figure=fig)

def get_scatterplot_matrix_old(data, identifier, args):
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
    if 'name' in data.columns:
        text = data.name
    else:
        text= data[args['group']]

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
                            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            template='plotly_white'
                            )

    return dcc.Graph(id=identifier, figure=figure)

def get_simple_scatterplot(data, identifier, args):
    """
    Plots a simple scatterplot with the possibility of including in-plot annotations of data points.

    :param data: long-format pandas dataframe with columns: 'x' (ranking position), 'group' (original dataframe position), \
                    'name' (protein identifier), 'y' (LFQ intensity), 'symbol' (data point shape) and 'size' (data point size).
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **annotations** (dict) -- dictionary where data points names are the keys and descriptions are the values.
        * **title** (str) -- plot title.
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **height** (int) -- plot height.
        * **width** (int) -- plot width.
    :return: annotated scatterplot figure within the <div id="_dash-app-content">.

    Example::

        result = get_scatterplot_matrix(data, identifier='scatter plot', args={'annotations':{'GPT~P24298': 'liver disease', 'CP~P00450': 'Wilson disease'}', \
                                        'title':'Scatter Plot', 'x_title':'x_axis', 'y_title':'y_axis', 'height':100, 'width':100})
    """
    figure = {}
    m = {'size': 15, 'line': {'width': 0.5, 'color': 'grey'}}
    text = data.name
    if 'colors' in data.columns:
        m.update({'color':data['colors'].tolist()})
    elif 'colors' in args:
        m.update({'color':args['colors'].tolist()})
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
    figure['data'] = [go.Scattergl(x = data.x,
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
                                showlegend=False,
                                template='plotly_white'
                                )

    return dcc.Graph(id= identifier, figure = figure)


def get_scatterplot(data, identifier, args):
    """
    This function plots a simple Scatterplot.

    :param data: is a Pandas DataFrame with four columns: "name", x values and y values (provided as variables) to plot.
    :param str identifier: is the id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **title** (str) -- title of the figure.
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **height** (int) -- plot height.
        * **width** (int) -- plot width.
    :return: scatterplot figure within the <div id="_dash-app-content">.

    Example::

        result = get_scatteplot(data, identifier='scatter plot', 'title':'Scatter Plot', 'x_title':'x_axis', 'y_title':'y_axis', 'height':100, 'width':100}))
    """
    figure = {}
    figure["data"] = []
    figure["layout"] = go.Layout(title = args['title'],
                                xaxis= {"title": args['x_title']},
                                yaxis= {"title": args['y_title']},
                                #margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
                                legend={'x': -.4, 'y': 1.2},
                                hovermode='closest',
                                height=args['height'],
                                width=args['width'],
                                annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
                                template='plotly_white'
                                )
    for name in data.name.unique():
        m = {'size': 25, 'line': {'width': 0.5, 'color': 'grey'}}
        if 'colors' in args:
            if name in args['colors']:
                m.update({'color' : args['colors'][name]})
        figure["data"].append(go.Scattergl(x = data.loc[data["name"] == name, "x"],
                                        y = data.loc[data['name'] == name, "y"],
                                        text = name,
                                        mode = 'markers',
                                        opacity=0.7,
                                        marker= m,
                                        name=name))

    return dcc.Graph(id= identifier, figure = figure)

def get_volcanoplot(results, args):
    """
    This function plots volcano plots for each internal dictionary in a nested dictionary.

    :param dict[dict] results: nested dictionary with pairwise group comparisons as keys and internal dictionaries containing 'x' (log2FC values), \
                                'y' (-log10 p-values), 'text', 'color', 'pvalue' and 'annotations' (number of hits to be highlighted).
    :param dict args: see below.
    :Arguments:
        * **fc** (float) -- fold change threshold.
        * **range_x** (list) -- list with minimum and maximum values for x axis.
        * **range_y** (list) -- list with minimum and maximum values for y axis.
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **colorscale** (str) -- string for predefined plotly colorscales or dict containing one or more of the keys listed in \
                                    https://plot.ly/python/reference/#layout-colorscale.
        * **showscale** (bool) -- determines whether or not a colorbar is displayed for a trace.
        * **marker_size** (int) -- sets the marker size (in px).
    :return: list of volcano plot figures within the <div id="_dash-app-content">.

    Example::

        result = get_volcanoplot(results, args={'fc':2.0, 'range_x':[0, 1], 'range_y':[-1, 1], 'x_title':'x_axis', 'y_title':'y_title', 'colorscale':'Blues', \
                                'showscale':True, 'marker_size':7})
    """
    figures = []
    for identifier,title in results:
        result = results[(identifier,title)]
        figure = {"data":[],"layout":None}
        if "range_x" not in args:
            range_x = [-max(abs(result['x']))-0.1, max(abs(result['x']))+0.1]#if symmetric_x else []
        else:
            range_x = args["range_x"]
        if "range_y" not in args:
            range_y = [0,max(abs(result['y']))+1.]
        else:
            range_y = args["range_y"]
        trace = go.Scatter(x=result['x'],
                        y=result['y'],
                        mode='markers',
                        text=result['text'],
                        hoverinfo='text',
                        marker={'color':result['color'], 
                                'colorscale': args["colorscale"], 
                                'showscale': args['showscale'], 
                                'size': args['marker_size'],
                                'line': {'color':result['color'], 'width':2}
                                }
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
                                        template='plotly_white',
                                        showlegend=False)

        figures.append(dcc.Graph(id= identifier, figure = figure))
    return figures

def run_volcano(data, identifier, args={'alpha':0.05, 'fc':2, 'colorscale':'Blues', 'showscale': False, 'marker_size':8, 'x_title':'log2FC', 'y_title':'-log10(pvalue)', 'num_annotations':10}):
    """ 
    This function parsers the regulation data from statistical tests and creates volcano plots for all distinct group comparisons. Significant hits with lowest adjusted p-values are highlighed.

    :param data: pandas dataframe with format: 'identifier', 'group1', 'group2', 'mean(group1', 'mean(group2)', 'log2FC', 'std_error', 'tail', 't-statistics', 'padj_THSD', \
                                                'effsize', 'efftype', 'FC', 'rejected', 'F-statistics', 'pvalue', 'padj', 'correction', '-log10 pvalue' and 'Method'.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **alpha** (float) -- adjusted p-value threshold for significant hits.
        * **fc** (float) -- fold change threshold.
        * **colorscale** (str or dict) -- name of predefined plotly colorscale or dictionary containing one or more of the keys listed in \
                                    https://plot.ly/python/reference/#layout-colorscale.
        * **showscale** (bool) -- determines whether or not a colorbar is displayed for a trace.
        * **marker_size** (int) -- sets the marker size (in px).
        * **x_title** (str) -- plot x axis title.
        * **y_title** (str) -- plot y axis title.
        * **num_annotations** (int) -- number of hits to be highlighted (if num_annotations = 10, highlights 10 hits with lowest adjusted p-value).
    :return: list of volcano plot figures within the <div id="_dash-app-content">.

    Example::

        result = run_volcano(data, identifier='volvano data', args={'alpha':0.05, 'fc':2.0, 'colorscale':'Blues', 'showscale':False, 'marker_size':6, 'x_title':'log2FC', \
                            'y_title':'-log10(pvalue)', 'num_annotations':10})
    """
    # Loop through signature
    volcano_plot_results = {}
    grouping = data.groupby(['group1','group2'])

    for group in grouping.groups:
        signature = grouping.get_group(group)
        color = []
        line_colors = []
        text = []
        annotations = []
        num_annotations = args['num_annotations'] if 'num_annotations' in args else 10
        gidentifier = identifier + "_".join(map(str,group))
        title = 'Comparison: '+str(group[0])+' vs '+str(group[1])
        sig_pval = False
        signature = signature.sort_values(by="padj",ascending=True)
        pvals = []
        for index, row in signature.iterrows():
            # Text
            text.append('<b>'+str(row['identifier'])+": "+str(index)+'<br>Comparison: '+str(row['group1'])+' vs '+str(row['group2'])+'<br>log2FC = '+str(round(row['log2FC'], ndigits=2))+'<br>p = '+'{:.2e}'.format(row['pvalue'])+'<br>FDR = '+'{:.2e}'.format(row['padj']))

            # Color
            if row['padj'] < args['alpha']:
                pvals.append(row['-log10 pvalue'])
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
                                    'font': dict(color = "#2c7bb6", size = 10)})
                    color.append('rgba(44, 123, 182, 0.2)')
                    line_colors.append('#2c7bb6')
                elif row['FC'] >= args['fc']:
                    annotations.append({'x': row['log2FC'],
                                    'y': row['-log10 pvalue'],
                                    'xref':'x',
                                    'yref': 'y',
                                    'text': str(row['identifier']),
                                    'showarrow': False,
                                    'ax': 0,
                                    'ay': -10,
                                    'font': dict(color = "#d7191c", size = 10)})
                    color.append('rgba(215, 25, 28, 0.2)')
                    line_colors.append('#d7191c')
                elif row['FC'] < -1.:
                    color.append('rgba(171, 217, 233, 0.1)')
                    line_colors.append('#abd9e9')
                elif row['FC'] > 1.:
                    color.append('rgba(253, 174, 97, 0.1)')
                    line_colors.append('#fdae61')
                else:
                    color.append('rgba(153, 153, 153, 0.1)')
                    line_colors.append('#999999')
            else:
                color.append('rgba(153, 153, 153, 0.1)')
                line_colors.append('#999999')

        if len(annotations) < num_annotations:
            num_annotations = len(annotations)

        if len(pvals) > 0:
            pvals.sort()
            min_pval_sign = pvals[0]
        else:
            min_pval_sign = 0

        volcano_plot_results[(gidentifier, title)] = {'x': signature['log2FC'].values, 'y': signature['-log10 pvalue'].values, 'text':text, 'color': color, 'line_color':line_colors, 'pvalue':min_pval_sign, 'annotations':annotations[0:num_annotations]}

    figures = get_volcanoplot(volcano_plot_results, args)

    return figures

def get_heatmapplot(data, identifier, args):
    """
    This function plots a simple Heatmap.

    :param data: is a Pandas DataFrame with the shape of the heatmap where index corresponds to rows \
                and column names corresponds to columns, values in the heatmap corresponds to the row values.
    :param str identifier: is the id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **format** (str) -- defines the format of the input dataframe.
        * **source** (str) -- name of the column containing the source.
        * **target** (str) -- name of the column containing the target.
        * **values** (str) -- name of the column containing the values to be plotted.
        * **title** (str) -- title of the figure.
    :return: heatmap figure within the <div id="_dash-app-content">.

    Example::

        result = get_heatmapplot(data, identifier='heatmap', args={'format':'edgelist', 'source':'node1', 'target':'node2', 'values':'score', 'title':'Heatmap Plot'})
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
                        "annotations" : [dict(xref='paper', yref='paper', showarrow=False, text='')],
                        "template":'plotly_white'}
    figure['data'].append(go.Heatmap(z=df.values.tolist(),
                                    x = list(df.columns),
                                    y = list(df.index)))

    return dcc.Graph(id = identifier, figure = figure)

# def get_complex_heatmapplot(df, identifier, args):
#     figure = {}
#     if args['format'] == "edgelist":
#         df = df.set_index(args['source'])
#         df = df.pivot_table(values=args['values'], index=df.index, columns=args['target'], aggfunc='first')
#         df = df.dropna(how='all', axis=1)
#         df = df.fillna(0)

#     figure = dashbio.Clustergram(width=1600,
#                                  color_threshold={'row': 150, 'col': 700},
#                                  color_map='BuPu',
#                                  data=df.values,
#                                  row_labels=list(df.index),
#                                  column_labels=list(df.columns.values),
#                                  hide_labels=['row'],
#                                  height=1800)

#     return dcc.Graph(id=identifier, figure=figure)

def get_complex_heatmapplot_old(data, identifier, args):
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
                             "template":'plotly_white'
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


    return dcc.Graph(id=identifier, figure=figure,)

def get_notebook_network_pyvis(graph, args={}):
    """
    This function converts a Networkx graph into a PyVis graph supporting Jupyter notebook embedding.

    :param graph graph: networkX graph.
    :param dict args: see below.
    :Arguments:
        * **height** (int) -- network canvas height.
        * **width** (int) -- network canvas width.
    :return: PyVis graph.

    Example::

        result = get_notebook_network_pyvis(graph, args={'height':100, 'width':100})
    """
    if 'width' not in args:
        args['width'] = 800
    if 'height' not in args:
        args['height'] = 850
    notebook_net = visnet(args['width'], args['height'], notebook=True)
    notebook_net.barnes_hut(overlap=0.8)
    notebook_net.from_nx(graph)
    notebook_net.show_buttons(['nodes', 'edges', 'physics'])
    utils.generate_html(notebook_net)

    return notebook_net

def get_notebook_network_web(graph, args):
    """
    This function converts a networkX graph into a webweb interactive network in a browser.

    :param graph graph: networkX graph.
    :return: web network.
    """
    notebook_net = Web(nx.to_numpy_matrix(graph).tolist())
    notebook_net.display.scaleLinkWidth = True

    return notebook_net

def network_to_tables(graph):
    """
    Creates the graph edge list and node list and returns them as separate Pandas DataFrames.

    :param graph: networkX graph used to construct the Pandas DataFrame.
    :return: two Pandas DataFrames.
    """
    edges_table = nx.to_pandas_edgelist(graph)
    nodes_table = pd.DataFrame.from_dict(dict(graph.nodes(data=True))).transpose().reset_index()

    return nodes_table, edges_table

def generate_configuration_tree(report_pipeline, dataset_type):
    """
    This function retrieves the analysis pipeline from a dataset .yml file and creates a Cytoscape network, organized hierarchically.

    :param dict report_pipeline: dictionary with dataset type analysis and visualization pipeline (conversion of .yml files to python dictionary).
    :param str dataset_type: type of dataset ('clinical', 'proteomics', 'DNAseq', 'RNAseq', 'multiomics').
    :return: new Dash div with title and Cytoscape network, summarizing analysis pipeline.
    """
    nodes = []
    edges = []
    args = {}
    conf_plot = None
    if len(report_pipeline) >=1:
        root = dataset_type.title() + " default analysis pipeline"
        nodes.append({'data':{'id':0, 'label':root}, 'classes': 'root'})
        i = 0
        for section in report_pipeline:
            if section == "args":
                continue
            nodes.append({'data':{'id':i+1, 'label':section.title()}, 'classes': 'section'})
            edges.append({'data':{'source':0, 'target':i+1}})
            i += 1
            k = i
            for subsection in report_pipeline[section]:
                nodes.append({'data':{'id':i+1, 'label':subsection.title()}, 'classes': 'subsection'})
                edges.append({'data':{'source':k, 'target':i+1}})
                i += 1
                j = i
                conf = report_pipeline[section][subsection]
                data_names = conf['data']
                analysis_types = conf['analyses']
                arguments = conf['args']
                if isinstance(data_names, dict):
                    for d in data_names:
                        nodes.append({'data':{'id':i+1, 'label':d+':'+data_names[d]}, 'classes': 'data'})
                        edges.append({'data':{'source':j, 'target':i+1}})
                        i += 1
                else:
                    nodes.append({'data':{'id':i+1, 'label':data_names}, 'classes': 'data'})
                    edges.append({'data':{'source':j, 'target':i+1}})
                    i += 1
                for at in analysis_types:
                    nodes.append({'data':{'id':i+1, 'label':at},'classes': 'analysis'})
                    edges.append({'data':{'source':j, 'target':i+1}})
                    i += 1
                    f = i
                if len(analysis_types):
                    for a in arguments:
                        nodes.append({'data':{'id':i+1, 'label':a+':'+str(arguments[a])},'classes': 'argument'})
                        edges.append({'data':{'source':f, 'target':i+1}})
                        i += 1
        config_stylesheet = [
                        # Group selectors
                        {
                            'selector': 'node',
                            'style': {
                                'content': 'data(label)'
                                }
                        },
                        # Class selectors
                        {
                            'selector': '.root',
                            'style': {
                                'background-color': '#66c2a5',
                                'line-color': 'black',
                                'font-size': '14'
                            }
                        },
                        {
                            'selector': '.section',
                            'style': {
                                'background-color': '#a6cee3',
                                'line-color': 'black',
                                'font-size': '12'
                            }
                        },
                        {
                            'selector': '.subsection',
                            'style': {
                                'background-color': '#1f78b4',
                                'line-color': 'black',
                                'font-size': '12'
                            }
                        },
                        {
                            'selector': '.data',
                            'style': {
                                'background-color': '#b2df8a',
                                'line-color': 'black',
                                'font-size': '12'
                            }
                        },
                        {
                            'selector': '.analysis',
                            'style': {
                                'background-color': '#33a02c',
                                'line-color': 'black',
                                'font-size': '12'
                            }
                        },
                        {
                            'selector': '.argument',
                            'style': {
                                'background-color': '#fb9a99',
                                'line-color': 'black',
                                'font-size': '12'
                            }
                        },
                    ]
        net = []
        net.extend(nodes)
        net.extend(edges)
        args['stylesheet'] = config_stylesheet
        args['title'] = 'Analysis Pipeline'
        args['layout'] = {'name': 'breadthfirst', 'roots': '#0'}
        #args['mouseover_node'] = {}
        conf_plot = get_cytoscape_network(net, dataset_type, args)

    return conf_plot

def get_network(data, identifier, args):
    """
    This function filters an input dataframe based on a threshold score and builds a cytoscape network. For more information on \
    'node_size' parameter, visit https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html and \
    https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.eigenvector_centrality_numpy.html.

    :param data: long-format pandas dataframe with at least three columns: source node, target node and value (e.g. weight, score).
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **source** (str) -- name of the column containing the source.
        * **target** (str) -- name of the column containing the target.
        * **cutoff** (float) -- value threshold for network building.
        * **cutoff_abs** (bool) -- if True will take both positive and negative sides of the cutoff value.
        * **values** (str) -- name of the column containing the values to be plotted.
        * **node_size** (str) -- method used to determine node radius ('betweenness', 'ev_centrality', 'degree').
        * **title** (str) -- plot title.
        * **color_weight** (bool) -- if True, edges in network are colored red if score > 0 and blue if score < 0.
    :return: dictionary with the network in multiple formats: jupyetr-notebook compatible, web brower compatibles, data table, and json.

    Example::

        result = get_network(data, identifier='network', args={'source':'node1', 'target':'node2', 'cutoff':0.5, 'cutoff_abs':True, 'values':'weight', \
                            'node_size':'ev_centrality', 'title':'Network Figure', 'color_weight': True})
    """
    net = None
    if 'cutoff_abs' not in args:
        args['cutoff_abs'] = False
    if not data.empty:
        if 'cutoff' in args:
            if args['cutoff_abs']:
                data = data[np.abs(data[args['values']]) > args['cutoff']]
            else:
                data = data > args['cutoff']
               
        data[args["source"]] = [str(n).replace("'","") for n in data[args["source"]]]
        data[args["target"]] = [str(n).replace("'","") for n in data[args["target"]]]
        
        if "values" not in args:
            args["values"] = 'width'
            data[args["values"]] = 1 
        
        data = data.rename(index=str, columns={args['values']: "width"})
        data['edge_width'] = data['width'].apply(np.abs)
        min_edge_value = data['edge_width'].min()
        max_edge_value = data['edge_width'].max()
        graph = nx.from_pandas_edgelist(data, args['source'], args['target'], edge_attr=True)

        degrees = dict(graph.degree())
        nx.set_node_attributes(graph, degrees, 'degree')
        betweenness = None
        ev_centrality = None
        if data.shape[0] < 100 and data.shape[0] > 5:
            betweenness = nx.betweenness_centrality(graph, weight='width')
            ev_centrality = nx.eigenvector_centrality_numpy(graph)
            nx.set_node_attributes(graph, betweenness, 'betweenness')
            nx.set_node_attributes(graph, ev_centrality, 'eigenvector')

        min_node_size = 0
        max_node_size = 0
        if args['node_size'] == 'betweenness' and betweenness is not None:
            min_node_size = min(betweenness.values())
            max_node_size = max(betweenness.values())
            nx.set_node_attributes(graph, betweenness, 'radius')
        elif args['node_size'] == 'ev_centrality' and ev_centrality is not None:
            min_node_size = min(ev_centrality.values())
            max_node_size = max(ev_centrality.values())
            nx.set_node_attributes(graph, ev_centrality, 'radius')
        elif args['node_size'] == 'degree':
            min_node_size = min(degrees.values())
            max_node_size = max(degrees.values())
            nx.set_node_attributes(graph, degrees, 'radius')

        clusters = analytics.get_network_communities(graph, args)
        col = utils.get_hex_colors(len(set(clusters.values())))
        colors = {n:col[clusters[n]] for n in clusters}
        nx.set_node_attributes(graph, colors, 'color')
        nx.set_node_attributes(graph, clusters, 'cluster')

        vis_graph = graph
        if len(vis_graph.edges()) > 500:
            max_nodes = 100
            cluster_members = defaultdict(list)
            cluster_nums = {}
            for n in clusters:
                if clusters[n] not in cluster_nums:
                    cluster_nums[clusters[n]] = 0
                cluster_members[clusters[n]].append(n)
                cluster_nums[clusters[n]] += 1
            valid_clusters = [c for c,n in sorted(cluster_nums.items() ,  key=lambda x: x[1])]
            valid_nodes = []
            for c in valid_clusters:
                valid_nodes.extend(cluster_members[c])
                if len(valid_nodes) >= max_nodes:
                    valid_nodes = valid_nodes[0:max_nodes]
                    break
            vis_graph = vis_graph.subgraph(valid_nodes)

        nodes_table, edges_table = network_to_tables(graph)
        nodes_fig_table = get_table(nodes_table, identifier=identifier+"_nodes_table", title=args['title']+" nodes table")
        edges_fig_table = get_table(edges_table, identifier=identifier+"_edges_table", title=args['title']+" edges table")

        stylesheet, layout = get_network_style(colors, args['color_weight'])
        stylesheet.append({'selector':'edge','style':{'width':'mapData(edge_width,'+ str(min_edge_value) +','+ str(max_edge_value) +', .5, 8)'}})
        if min_node_size > 0 and max_node_size >0:
            mapper = 'mapData(radius,'+ str(min_node_size) +','+ str(max_node_size) +', 15, 50)'
            stylesheet.append({'selector':'node','style':{'width':mapper, 'height':mapper}})
        args['stylesheet'] = stylesheet
        args['layout'] = layout
        
        cy_elements, mouseover_node = utils.networkx_to_cytoscape(vis_graph)
        
        app_net = get_cytoscape_network(cy_elements, identifier, args)
        
        #args['mouseover_node'] = mouseover_node

        net = {"notebook":[cy_elements, stylesheet,layout], "app": app_net, "net_tables":(nodes_fig_table, edges_fig_table), "net_json":json_graph.node_link_data(graph)}
    return net

def get_network_style(node_colors, color_edges):
    '''
    This function uses a dictionary of nodes and colors and creates a stylesheet and layout for a network.

    :param dict node_colors: dictionary with node names as keys and colors as values.
    :param bool color_edges: if True, add edge coloring to stylesheet (red for positive width, blue for negative).
    :return: stylesheet (list of dictionaries specifying the style for a group of elements, a class of elements, or a single element) and \
                layout (dictionary specifying how the nodes should be positioned on the canvas).
    '''

    color_selector = "{'selector': '[name = \"KEY\"]', 'style': {'background-color': 'VALUE'}}"
    stylesheet=[{'selector': 'node', 'style': {'label': 'data(name)'}},
                {'selector':'edge','style':{'curve-style': 'bezier'}}]

    layout = {'name': 'cose',
                'idealEdgeLength': 100,
                'nodeOverlap': 20,
                'refresh': 20,
                #'fit': True,
                #'padding': 30,
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
        stylesheet.extend([{'selector':'[width < 0]', 'style':{'line-color':'#4dc3d6'}},{'selector':'[width > 0]', 'style':{'line-color':'#d6604d'}}])


    for k,v in node_colors.items():
        stylesheet.append(ast.literal_eval(color_selector.replace("KEY", k).replace("VALUE",v)))

    return stylesheet, layout

def visualize_notebook_network(network, notebook_type='jupyter', layout={'width':'100%', 'height':'700px'}):
    """
    This function returns a Cytoscape network visualization for Jupyter notebooks

    :param tuple network: tuple with two dictionaries: network data and stylesheet (see get_network(data, identifier, args)).
    :param str notebook_type: the type of notebook where the network will be visualized (currently only jupyter notebook is supported)
    :param dict layout: specific layout properties (see https://dash.plot.ly/cytoscape/layout)
    :return: cyjupyter.cytoscape.Cytoscape object

    Example::
    
        net = get_network(clincorr.dropna(), identifier='corr', args={'source':'node1', 'target':'node2',
                                                            'cutoff':0, 'cutoff_abs':True,
                                                            'values':'weight','node_size':'degree',
                                                            'title':'Network Figure', 'color_weight': True})
        visualize_notebook_network(net['notebook'], notebook_type='jupyter', layout={'width':'100%', 'height':'700px'})
    """
    net = None
    if notebook_type == 'jupyter':
        net = Cytoscape(data={'elements':network[0]}, visual_style=network[1], layout=layout)
    elif notebook_type == 'jupyterlab':
        pass

    return net

def get_pca_plot(data, identifier, args):
    """
    This function creates a pca plot with scores and top "args['loadings']" loadings.

    :param tuple data: tuple with two pandas dataframes: scores and loadings.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below
    :Arguments:
        * **loadings** (int) -- number of features with highest loading values to be displayed in the pca plot
        * **title** (str) -- title of the figure
        * **x_title** (str) -- plot x axis title
        * **y_title** (str) -- plot y axis title
        * **height** (int) -- plot height
        * **width** (int) -- plot width
    :return: PCA figure within the <div id="_dash-app-content">.

    Example::

        result = get_pca_plot(data, identifier='pca', args={'loadings':15, 'title':'PCA Plot', 'x_title':'PC1', 'y_title':'PC2', 'height':100, 'width':100})
    """
    pca_data, loadings = data
    figure = {}
    traces = []
    annotations = []
    sct = get_scatterplot(pca_data, identifier, args).figure
    traces.extend(sct['data'])
    figure['layout'] = sct['layout']
    figure['layout'].template='plotly_white'
    for index in list(loadings.index)[0:args['loadings']]:
        x = loadings.loc[index,'x'] * 5
        y = loadings.loc[index, 'y'] * 5
        value = loadings.loc[index, 'value']

        trace = go.Scattergl(x= [0,x],
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
    figure['layout']['template'] = 'plotly_white'

    return  dcc.Graph(id = identifier, figure = figure)


def get_sankey_plot(data, identifier, args={'source':'source', 'target':'target', 'weight':'weight','source_colors':'source_colors', 'target_colors':'target_colors', 'orientation': 'h', 'valueformat': '.0f', 'width':800, 'height':800, 'font':12, 'title':'Sankey plot'}):
    """
    This function generates a Sankey plot in Plotly.

    :param data: Pandas DataFrame with the format: source  target  weight.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below
    :Arguments:
        * **source** (str) -- name of the column containing the source
        * **target** (str) -- name of the column containing the target
        * **weight** (str) -- name of the column containing the weight
        * **source_colors** (str) -- name of the column in data that contains the colors of each source item
        * **target_colors** (str) -- name of the column in data that contains the colors of each target item
        * **title** (str) -- plot title
        * **orientation** (str) -- whether to plot horizontal ('h') or vertical ('v')
        * **valueformat** (str) -- how to show the value ('.0f')
        * **width** (int) -- plot width
        * **height** (int) -- plot height
        * **font** (int) -- font size
    :return: dcc.Graph

    Example::

        result = get_sankey_plot(data, identifier='sankeyplot', args={'source':'source', 'target':'target', 'weight':'weight','source_colors':'source_colors', \
                                'target_colors':'target_colors', 'orientation': 'h', 'valueformat': '.0f', 'width':800, 'height':800, 'font':12, 'title':'Sankey plot'})
    """
    figure = {}
    if data is not None and not data.empty:
        nodes = list(set(data[args['source']].tolist() + data[args['target']].tolist()))
        if 'source_colors' in args:
            node_colors = dict(zip(data[args['source']],data[args['source_colors']]))
        else:
            scolors = ['#045a8d']*len(data[args['source']].tolist())
            node_colors = dict(zip(data[args['source']],scolors))
            args['source_colors'] = 'source_colors'
            data['source_colors'] = scolors

        hover_data = []
        if 'hover' in args:
            hover_data = [t.upper() for t in data[args['hover']].tolist()]

        if 'target_colors' in args:
            node_colors.update(dict(zip(data[args['target']],data[args['target_colors']])))
        else:
            node_colors.update(dict(zip(data[args['target']],['#a6bddb']*len(data[args['target']].tolist()))))
        data_trace = dict(type='sankey',
                            #domain = dict(x =  [0,1], y =  [0,1]),
                            orientation = 'h' if 'orientation' not in args else args['orientation'],
                            valueformat = ".0f" if 'valueformat' not in args else args['valueformat'],
                            arrangement = 'freeform',
                            node = dict(pad = 10 if 'pad' not in args else args['pad'],
                                        thickness = 10 if 'thickness' not in args else args['thickness'],
                                        line = dict(color = "black", width = 0.3),
                                        label =  nodes,
                                        color =  ["rgba"+str(utils.hex2rgb(node_colors[c])) if node_colors[c].startswith('#') else node_colors[c] for c in nodes]
                                        ),
                            link = dict(source = [list(nodes).index(i) for i in data[args['source']].tolist()],
                                        target = [list(nodes).index(i) for i in data[args['target']].tolist()],
                                        value =  data[args['weight']].tolist(),
                                        color = ["rgba"+str(utils.hex2rgb(c)) if c.startswith('#') else c for c in data[args['source_colors']].tolist()],
                                        label = hover_data
                                        ))
        layout =  dict(
            width= 800 if 'width' not in args else args['width'],
            height= 800 if 'height' not in args else args['height'],
            title = args['title'],
            annotations = [dict(xref='paper', yref='paper', showarrow=False, text='')],
            font = dict(
                size = 12 if 'font' not in args else args['font'],
            ),
            template='plotly_white'
            )

        figure = dict(data=[data_trace], layout=layout)

    return dcc.Graph(id = identifier, figure = figure)

def get_table(data, identifier, title, colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':2500, 'font':12}, subplot = False):
    """
    This function converts a pandas dataframe into an interactive table for viewing, editing and exploring large datasets. For more information visit https://dash.plot.ly/datatable.

    :param data: pandas dataframe.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param str title: table title.
    :param subset: selects columns from dataframe to be used. If None, the entire dataframe is used.
    :type subest: str, list or None
    :return: new Dash div containing title and interactive table.

    Example::

        result = get_table(data, identifier='table', title='Table Figure', subset = None)
    """
    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
        if subset is not None:
            data = data[subset]

        #booleanDictionary = {True: 'TRUE', False: 'FALSE'}
        #if 'rejected' in data.columns:
        #    data['rejected'] = data['rejected'].replace(booleanDictionary)

        list_cols = data.applymap(lambda x: isinstance(x, list)).all()
        list_cols = list_cols.index[list_cols].tolist()

        for c in list_cols:
            data[c] = data[c].apply(lambda x: ";".join(x))

        data_trace = dash_table.DataTable(id='table_'+identifier,
                                            data=data.to_dict("rows"),
                                            columns=[{"name": str(i).replace('_', ' ').title(), "id": i} for i in data.columns],
                                            css=[{
                                                'selector': '.dash-cell div.dash-cell-value',
                                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                            }],
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                            style_cell={
                                                'minWidth': '50px', 'maxWidth': '500px',
                                                'textAlign': 'left', 'padding': '1px', 'vertical-align': 'top'
                                            },
                                            style_table={
                                                "height": "fit-content",
                                            #    "max-height": "500px",
                                                "width": "fit-content",
                                            #    "max-width": "1500px",
                                            #    'overflowY': 'scroll',
                                            #    'overflowX': 'scroll'
                                            },
                                            style_header={
                                                'backgroundColor': '#2b8cbe',
                                                'fontWeight': 'bold',
                                                'position': 'sticky'
                                            },
                                            style_data_conditional=[{
                                                "if": 
                                                    {"column_id": "Rejected", "filter_query": 'Rejected eq "True"'},
                                                    "backgroundColor": "#3B8861",
                                                    'color': 'white'
                                                },
                                                ],
                                            fixed_rows={ 'headers': True },
                                            filter_action='native',
                                            page_current= 0,
                                            page_size = 25,
                                            page_action='native',
                                            sort_action='custom',
                                            )
        table = [html.H2(title),data_trace]
    else:
        table = None

    return html.Div(table)

def get_multi_table(data,identifier, title):
    tables = [html.H2(title)]
    if data is not None and isinstance(data, dict):
        for subtitle in data:
            df = data[subtitle]
            if len(df.columns) > 10:
                df = df.transpose()
            table = get_table(df, identifier=identifier+"_"+subtitle, title=subtitle)
            if table is not None:
                tables.append(table)
    
    return html.Div(tables)
            

def get_violinplot(data, identifier, args):
    """
    This function creates a violin plot for all columns in the input dataframe.

    :param data: pandas dataframe with samples as rows and dependent variables as columns.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below
    :Arguments:
        * **drop_cols** (list) -- column labels to be dropped from the dataframe.
        * **group** (str) -- name of the column containing the group.
    :return: list of violion plots within the <div id="_dash-app-content">.

    Example::

        result = get_violinplot(data, identifier='violinplot, args={'drop_cols':['sample', 'subject'], 'group':'group'})
    """
    df = data.copy()
    graphs = []
    if 'drop_cols' in args:
        if set(args['drop_cols']).intersection(df.columns) == len(args['drop_cols']):
            df = df.drop(args['drop_cols'], axis=1)
    for c in df.columns.unique():
        if c != args['group']:
            traces = create_violinplot(df, c, args['group'])
            figure = {"data": traces,
                    "layout":{
                            "title": "Violinplot per group for variable: "+c,
                            "annotations": [dict(xref='paper', yref='paper', showarrow=False, text='')],
                            "template":'plotly_white',
                            "yaxis": {
                                "zeroline":False,
                                }
                        }
                    }
            graphs.append(dcc.Graph(id=identifier+"_"+c, figure=figure))

    return graphs

def create_violinplot(df, variable, group_col='group'):
    """
    This function creates traces for a simple violin plot.

    :param df: pandas dataframe with samples as rows and dependent variables as columns.
    :param (str) variable: name of the column with the dependent variable.
    :pram (str) group_col: name of the column containing the group.
    :return: list of traces to be used as data for plotly figure.

    Example::

        result = create_violinplot(df, 'prptein a', group_col='group')
    """
    traces = []
    for group in np.unique(df[group_col].values):
        violin = {"type": 'violin',
                    "x": df[group_col][df[group_col] == group].values,
                    "y": df[variable][df[group_col] == group].values,
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


def get_clustergrammer_plot(data, identifier, args):
    """
    This function takes a pandas dataframe, calculates clustering, and generates the visualization json.
    For more information visit https://github.com/MaayanLab/clustergrammer-py.

    :param data: long-format pandas dataframe with columns 'node1' (source), 'node2' (target) and 'weight'
    :param str identifier: id used to identify the div where the figure will be generated
    :param dict args: see below
    :Arguments:
        * **format** (str) -- defines if dataframe needs to be converted from 'edgelist' to matrix
        * **title** (str) -- plot title
    :return: Dash Div with heatmap plot from Clustergrammer web-based tool
    """
    from clustergrammer2 import net as clustergrammer_net
    div = None
    if not data.empty:
        if 'format' in args:
            if args['format'] == 'edgelist':
                data = data[['node1', 'node2', 'weight']].pivot(index='node1', columns='node2')
        clustergrammer_net.load_df(data)

        link = utils.get_clustergrammer_link(clustergrammer_net, filename=None)

        iframe = html.Iframe(src=link, width=1000, height=900)

        div = html.Div([html.H2(args['title']),iframe])
    return div

def get_parallel_plot(data, identifier, args):
    """
    This function creates a parallel coordinates plot, with sample groups as the different dimensions.

    :param data: pandas dataframe with groups as rows and dependent variables as columns.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **group** (str) -- name of the column containing the groups.
        * **zscore** (bool) -- if True, calculates the z score of each values in the row, relative to the row mean and standard deviation.
        * **color** (str) -- line color.
        * **title** (str) -- plot title.
    :return: parallel plot figure within <div id="_dash-app-content"> .

    Example::

        result = get_parallel_plot(data, identifier='parallel plot', args={'group':'group', 'zscore':True, 'color':'blue', 'title':'Parallel Plot'})
    """
    if 'group' in args:
        group = args['group']
        if 'zscore' in args:
            if args['zscore']:
                data = data.set_index(group).apply(zscore)
        color = '#de77ae'
        if 'color' in args:
            color = args['color']
        min_val = round(data._get_numeric_data().min().min())
        max_val = round(data._get_numeric_data().max().max())
        df = data.reset_index().groupby(group)
        dims = []
        for i in df.groups:
            values = df.get_group(i).values.tolist()[0][1:]

            dim = dict(label=i, range=[min_val, max_val], values=values)
            dims.append(dim)

        fig_data = [
                go.Parcoords(
                    line = dict(color = color),
                    dimensions = dims)
                ]
        layout = go.Layout(
            title=args['title'],
            annotations= [dict(xref='paper', yref='paper', showarrow=False, text='')],
            template='plotly_white'
        )

        fig = dict(data = fig_data, layout = layout)

    return dcc.Graph(id=identifier, figure=fig)

def get_WGCNAPlots(data, identifier):
    """
    Takes data from runWGCNA function and builds WGCNA plots.

    :param data: tuple with multiple pandas dataframes.
    :param str identifier: is the id used to identify the div where the figure will be generated.
    :return: list of dcc.Graph.
    """
    graphs = []
    data = tuple(data[k] for k in data)

    if data is not None:
        data_exp, data_cli, dissTOM, moduleColors, Features_per_Module, MEs,\
        moduleTraitCor, textMatrix, MM, MMPvalue, FS, FSPvalue, METDiss, METcor = data
        plots = []
        # plot: sample dendrogram and clinical variables heatmap; input: data_exp, data_cli
        #plots.append(wgcnaFigures.plot_complex_dendrogram(data_exp, data_cli, title='Clinical variables variation by sample', dendro_labels=data_exp.index, distfun='euclidean', linkagefun='average', hang=40, subplot='heatmap', color_missingvals=True, width=1000, height=800))

        # plot: gene tree dendrogram and module colors; input: dissTOM, moduleColors
        plots.append(wgcnaFigures.plot_complex_dendrogram(dissTOM, moduleColors, title='Co-expression: dendrogram and module colors', dendro_labels=dissTOM.columns, distfun=None, linkagefun='ward', hang=0.1, subplot='module colors', col_annotation=True, width=1000, height=800))

        # plot: table with features per module; input: df
        plots.append(get_table(Features_per_Module, identifier='', title='Proteins/Genes module color', colors = ('#C2D4FF','#F5F8FF'), subset = None,  plot_attr = {'width':1500, 'height':1500, 'font':12}, subplot = False))

        #plot: module-traits correlation with annotations; input: moduleTraitCor, textMatrix
        plots.append(wgcnaFigures.plot_labeled_heatmap(moduleTraitCor, textMatrix, title='Module-Clinical variable relationships', colorscale=[[0,'#67a9cf'],[0.5,'#f7f7f7'],[1,'#ef8a62']], row_annotation=True, width=1000, height=800))

        #plot: FS vs. MM correlation per trait/module scatter matrix; input: MM, FS, Features_per_Module
        #plots.append(wgcnaFigures.plot_intramodular_correlation(MM, FS, Features_per_Module, title='Intramodular analysis: Feature Significance vs. Module Membership', width=1000, height=2000))

        #input: METDiss, METcor
        # plots.append(wgcnaFigures.plot_complex_dendrogram(METDiss, METcor, title='Eigengene network and clinical data associations', dendro_labels=METDiss.index, distfun=None, linkagefun='ward', hang=0.9,
        #                          subplot='heatmap', subplot_colorscale=[[0,'#67a9cf'],[0.5,'#f7f7f7'],[1,'#ef8a62']],
        #                          color_missingvals=False, row_annotation=True, col_annotation=True, width=1000, height=800))

        dendro_tree = wgcnaAnalysis.get_dendrogram(METDiss, METDiss.index, distfun=None, linkagefun='ward', div_clusters=False)
        dendrogram = Dendrogram.plot_dendrogram(dendro_tree, hang=0.9, cutoff_line=False)

        layout = go.Layout(width=900, height=900, showlegend=False, title='',
                            xaxis=dict(domain=[0, 1], range=[np.min(dendrogram['layout']['xaxis']['tickvals'])-6,np.max(dendrogram['layout']['xaxis']['tickvals'])+4], showgrid=False,
                                        zeroline=True, ticks='', automargin=True, anchor='y'),
                            yaxis=dict(domain=[0.7, 1], autorange=True, showgrid=False, zeroline=False, ticks='outside', title='Height', automargin=True, anchor='x'),
                            xaxis2=dict(domain=[0, 1], autorange=True, showgrid=True, zeroline=False, ticks='', showticklabels=False, automargin=True, anchor='y2'),
                            yaxis2=dict(domain=[0, 0.64], autorange=True, showgrid=False, zeroline=False, automargin=True, anchor='x2'))

        if all(list(METcor.columns.map(lambda x: METcor[x].between(-1,1, inclusive=True).all()))) != True:
            df = wgcnaAnalysis.get_percentiles_heatmap(METcor, dendro_tree, bydendro=True, bycols=False).T
        else:
            df = wgcnaAnalysis.df_sort_by_dendrogram(wgcnaAnalysis.df_sort_by_dendrogram(METcor, dendro_tree).T, dendro_tree)

        heatmap = wgcnaFigures.get_heatmap(df, colorscale=[[0,'#67a9cf'],[0.5,'#f7f7f7'],[1,'#ef8a62']], color_missing=False)

        figure = tools.make_subplots(rows=2, cols=1, print_grid=False)

        for i in list(dendrogram['data']):
            figure.append_trace(i, 1, 1)
        for j in list(heatmap['data']):
            figure.append_trace(j, 2, 1)

        figure['layout'] = layout
        figure['layout']['template'] = 'plotly_white'
        figure['layout'].update({'xaxis':dict(domain=[0, 1], ticks='', showticklabels=False, anchor='y'),
                                 'xaxis2':dict(domain=[0, 1], ticks='', showticklabels=True, anchor='y2'),
                                 'yaxis':dict(domain=[0.635, 1], anchor='x'),
                                 'yaxis2':dict(domain=[0., 0.635], ticks='', showticklabels=True, anchor='x2')})

        plots.append(figure)

        graphs = []
        for i, j in enumerate(plots):
            if isinstance(j, html.Div):
                graphs.append(j)
            else:
                graphs.append(dcc.Graph(id=identifier+'_'+str(i), figure=j))

    return graphs


def getMapperFigure(data, identifier, title):
    """
    This function uses the KeplerMapper python package to visualize high-dimensional data and generate a FigureWidget that can be shown or editted.
    This method is suitable for use in Jupyter notebooks. For more information visit https://kepler-mapper.scikit-tda.org/reference/stubs/kmapper.plotlyviz.plotlyviz.html.

    :param data: dictionary. Simplicial complex output from the KeplerMapper map method.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param str title: plot title.
    :return: plotly FigureWidget within <div id="_dash-app-content"> .
    """
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
                        hist_left=25, hist_right=25, member_textbox_width=800)
    return  dcc.Graph(id = identifier, figure=figure)

def get_2_venn_diagram(data, identifier, cond1, cond2, args):
    """
    This function extracts the exlusive features in cond1 and cond2 and their common features, and build a two-circle venn diagram.

    :param data: pandas dataframe with features as rows and group identifiers as columns.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param str cond1: identifier of first group.
    :param str cond2: identifier of second group.
    :param dict args: see below.
    :Arguments:
        * **colors** (dict) -- dictionary with cond1 and cond2 as keys, and color codes as values.
        * **title** (str) -- plot title.
    :return: two-circle venn diagram figure within <div id="_dash-app-content">.

    Example::

        result = get_2_venn_diagram(data, identifier='venn2', cond1='group1', cond2='group2', args={'color':{'group1':'blue', 'group2':'red'}, \
                                    'title':'Two-circle Venn diagram'})

    """
    figure = {}
    figure["data"] = []
    unique1 = len(set(data[cond1].dropna().index).difference(data[cond2].dropna().index))#/total
    unique2 = len(set(data[cond2].dropna().index).difference(data[cond1].dropna().index))#/total
    intersection = len(set(data[cond1].dropna().index).intersection(data[cond2].dropna().index))#/total

    return plot_2_venn_diagram(cond1, cond2, unique1, unique2, intersection, identifier, args)

def plot_2_venn_diagram(cond1, cond2, unique1, unique2, intersection, identifier, args):
    """
    This function creates a simple non area-weighted two-circle venn diagram.

    :param str cond1: label of the first circle.
    :param str cond2: label of the second circle.
    :param int unique1: number of features exclusive to cond1.
    :param int unique2: number of features exclusive to cond2.
    :parm int intersection: number of features common to cond1 and cond2.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **colors** (dict) -- dictionary with cond1 and cond2 as keys, and color codes as values.
        * **title** (str) -- plot title.
    :return: two-circle venn diagram figure within <div id="_dash-app-content">.

    Example::

        result = plot_2_venn_diagram(cond1='group1', cond2='group2', unique1=10, unique2=15, intersection=8, identifier='vennplot', \
                                    args={'color':{'group1':'blue', 'group2':'red'}, 'title':'Two-circle Venn diagram'})

    """
    figure = {}
    figure["data"] = []

    figure["data"] = [go.Scattergl(
        x=[1, 1.75, 2.5],
        y=[1, 1, 1],
        text=[str(unique1), str(intersection), str(unique2)],
        mode='text',
        textfont=dict(
            color='black',
            size=14,
            family='Arial',
        )
    )]

    if 'colors' not in args:
        args['colors'] = {cond1:'#a6bddb', cond2:'#045a8d'}


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
        'title':args['title'],
        "template":'plotly_white'
    }

    return dcc.Graph(id = identifier, figure=figure)

def get_wordcloud(data, identifier, args={'stopwords':[], 'max_words': 400, 'max_font_size': 100, 'width':700, 'height':700, 'margin': 1}):
    """
    This function generates a Wordcloud based on the natural text in a pandas dataframe column.

    :param data: pandas dataframe with columns: 'PMID', 'abstract', 'authors', 'date', 'journal', 'keywords', 'title', 'url', 'Proteins', 'Diseases'.
    :param str identifier: id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **text_col** (str) -- name of column containing the natural text used to generate the wordcloud.
        * **stopwords** (list) -- list of words that will be eliminated.
        * **max_words** (int) -- maximum number of words.
        * **max_font_size** (int) -- maximum font size for the largest word.
        * **margin** (int) -- plot margin size.
        * **width** (int) -- width of the plot.
        * **height** (int) -- height of the plot.
        * **title** (str) -- plot title.
    :return: wordcloud figure within <div id="_dash-app-content">.

    Example::

        result = get_wordcloud(data, identifier='wordcloud', args={'stopwords':['BACKGROUND','CONCLUSION','RESULT','METHOD','CONCLUSIONS','RESULTS','METHODS'], \
                                'max_words': 400, 'max_font_size': 100, 'width':700, 'height':700, 'margin': 1})
    """
    figure=None
    if data is not None:
        nltk.download('stopwords')
        sw = set(stopwords.words('english')).union(set(STOPWORDS))
        if 'stopwords' in args:
            sw = sw.union(args['stopwords'])

        if isinstance(data, pd.DataFrame):
            if not data.empty:
                if "text_col" in args:
                    text = ''.join(str(a) for a in data[args["text_col"]].unique().tolist())
                else:
                    return None
            else:
                return None
        else:
            text = data

        wc = WordCloud(stopwords = sw,
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

        trace = go.Scattergl(x=x,
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
                          template='plotly_white'
                          )

        figure = dict(data=[trace], layout=layout)

    return dcc.Graph(id = identifier, figure=figure)


def get_cytoscape_network(net, identifier, args):
    """
    This function creates a Cytoscpae network in dash. For more information visit https://dash.plot.ly/cytoscape.

    :param dict net: dictionary in which each element (key) is defined by a dictionary with 'id' and 'label' \
                    (if it is a node) or 'source', 'target' and 'label' (if it is an edge).
    :param str identifier: is the id used to identify the div where the figure will be generated.
    :param dict args: see below.
    :Arguments:
        * **title** (str) -- title of the figure.
        * **stylesheet** (list[dict]) -- specifies the style for a group of elements, a class of elements, or a single element \
                                        (accepts two keys 'selector' and 'style').
        * **layout** (dict) -- specifies how the nodes should be positioned on the screen.
    :return: network figure within <div id="_dash-app-content">.
    """
    cytonet = html.Div([html.H2(args['title']), cyto.Cytoscape(id=identifier,
                                    stylesheet=args['stylesheet'],
                                    elements=net,
                                    layout=args['layout'],
                                    minZoom = 0.2,
                                    maxZoom = 1.5,
                                    #mouseoverNodeData=args['mouseover_node'],
                                    style={'width': '100%', 'height': '700px'}
                                    )
                    ])


    return cytonet

def save_DASH_plot(plot, name, plot_format='svg', directory='.'):
    """
    This function saves a plotly figure to a specified directory, in a determined format.

    :param plot: plotly figure (dictionary with data and layout)
    :param str name: name of the figure
    :param str plot_format: suffix of the saved file ('svg', 'pdf', 'png', 'jpeg', 'jpg')
    :param str directory: folder where figure is to be saved
    :return: figure saved in directory

    Example::

        result = save_DASH_plot(plot, name='Plot example', plot_format='svg', directory='/data/plots')
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    if plot_format in ['svg', 'pdf', 'png', 'jpeg', 'jpg']:
        plot_file = os.path.join(directory, str(name)+'.'+str(plot_format))
        if hasattr(plot, 'figure'):
            pio.write_image(plot.figure, plot_file)
        else:
            pio.write_image(plot, plot_file)
