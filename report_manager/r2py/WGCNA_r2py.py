#R module for CKG

import os
import pandas as pd
import numpy as np
import scipy as scp
from scipy.cluster.hierarchy import distance, linkage, dendrogram, fcluster
from collections import OrderedDict, defaultdict
from natsort import natsorted, index_natsorted, order_by_index
import cufflinks as cf
import color_list_up
import urllib.request

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as FF
#from plotly_dendrogram import plot_dendrogram
#from plotly.offline import plot, iplot, , download_plotlyjs

from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FloatVector
import rpy2.robjects.packages as rpacks
import R2Py
from plot_dendrogram import plot_dendrogram
pandas2ri.activate()

R = ro.r

#Import R packages
base = R2Py.call_Rpackage("package", "base")
stats = R2Py.call_Rpackage("package", "stats")
WGCNA = R2Py.call_Rpackage("package", "WGCNA")
dynamicTreeCut = R2Py.call_Rpackage("function", "cutreeDynamic")



def get_preproc_data(filename):
    df = pd.read_hdf(filename, 'preprocessed')
    df = df.drop('group', axis=1)
    df = df.set_index('sample').reindex(natsorted(df['sample'])).reset_index()
    return df

def get_clinical_data(filename, df_exp):
    df = pd.read_hdf(filename, 'preprocessed')
    df = df[df['subject'].isin(df_exp.sample)]
    df = df.set_index('subject').reindex(natsorted(df['subject'])).reset_index()
    return df

def check_datasets(df_exp, df_traits):
    align = df_traits.index == df_exp.index
    unique, counts = np.unique(align, return_counts=True)
    if dict(zip(unique, counts))[True] == len(datTraits.index) and dict(zip(unique, counts))[True] == len(datExpr0.index):
        return 'Datasets OK'
    else:
        return 'Error: Check datasets'

def get_goodSamplesGenes(df, verbose = 1):
    df_r = pandas2ri.py2ri_pandasdataframe(df)
    gsg = WGCNA.goodSamplesGenes(df_r, verbose = verbose)

    if bool(gsg.rx2("allOK")) != True:
        samples = pd.DataFrame(np.array(gsg.rx2("goodSamples")), columns=[gsg.names[1]])
        samples['samples'] = R.rownames(df_r)
        features = pd.DataFrame(np.array(gsg.rx2("goodGenes")), columns=[gsg.names[0]])
        features['features'] = R.names(df_r)

        goodSamples = samples[samples['goodSamples'] == 1]['samples'].tolist()
        goodFeatures = features[features['goodGenes'] == 1]['features'].tolist()

        df2 = R2Py.R_matrix2Py_matrix(df_r)
        df2 = df2.loc[(df2.index.isin(goodSamples)), (df2.columns.isin(goodGenes))]
    else:
        df2 = R2Py.R_matrix2Py_matrix(df_r)

    return df2

def get_clusters_elements(linkage_matrix, fcluster_method, value, labels):
    clust = fcluster(linkage_matrix, fcluster_method, value)
    clusters = defaultdict(list)
    for i, j in zip(clust, labels):
        clusters[i].append(j)

    return clusters

def get_dendrogram(df, labels, distfun='euclidean', linkagefun='average', div_clusters=True, fcluster_method='distance', fcluster_cutoff=15):
    if distfun is None:
        dist = np.asarray(stats.as_dist(df))
    else:
        dist = np.asarray(stats.dist(df, method = distfun))

    Z = linkage(dist, method = linkagefun)
    Z_dendrogram = dendrogram(Z, no_plot = True, labels = labels)

    if div_clusters == True:
        clusters = get_clusters_elements(Z, fcluster_method, fcluster_cutoff, labels)
        return Z_dendrogram, clusters
    else:
        return Z_dendrogram

def filter_df_by_cluster(df, clusters, number):
    df2 = df[df.index.isin(clusters[number])]
    return df2

def df_sort_by_dendrogram(df, Z_dendrogram):
    df2 = df.copy()
    df2.index = pd.CategoricalIndex(df2.index, categories=Z_dendrogram['ivl'])
    df2.sort_index(level=0, inplace=True)
    df2 = df2.T
    return df2

def get_percentiles_heatmap(df, Z_dendrogram, bydendro= True, bycols=True):
    if bydendro == True:
        df2 = df_sort_by_dendrogram(df, Z_dendrogram)
    else:
        df2 = df

    p = pd.DataFrame(index=df2.index, columns=df2.columns)

    if bycols == True:
        for j in df2.index:
            for i in df2.columns:
                pct = (df2.loc[j,i] - np.nanmin(df2.loc[j,:])) / ((np.nanmax(df2.loc[j, :]) - np.nanmin(df2.loc[j, :])) * 1.)
                pct = pct - (pct - 0.5) * 1. / 40 #have to rescale it to account for endpoints of cmaps
                p.loc[j,i] = pct
    else:
        for i in df2.index:
            for j in df2.columns:
                pct = (df2.loc[i,j] - np.nanmin(df2.loc[:,j])) / ((np.nanmax(df2.loc[:,j]) - np.nanmin(df2.loc[:,j])) * 1.)
                pct = pct - (pct - 0.5) * 1. / 40 #have to rescale it to account for endpoints of cmaps
                p.loc[i,j] = pct
    return p

def get_miss_values_df(df):
    df2 = df.copy()
    df2 = df2.isnull().astype('int')
    df2 = df2.replace(0, np.nan)
    return df2

def plot_heatmap(df_percentiles, colorscale=None , color_missing=True, outliers=True):
    if colorscale:
        colors = colorscale
    else:
        colors = [[0, 'rgb(255,255,255)'], [1, 'rgb(255,51,0)']]

    trace_valid = go.Heatmap(z=df_percentiles.values, y=df_percentiles.index, colorscale=colors, showscale=True,
                             colorbar=dict(x=1, y=0.475, xanchor='left', yanchor='middle', len=0.35, thickness=15))

    outlier_df = df_percentiles[df_percentiles.index == 'outlierC']
    trace_outlier = go.Heatmap(z=outlier_df.values, y=outlier_df.index, colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,51,0)']], showscale=False)

    df_missing = get_miss_values_df(df_percentiles)
    trace_missing = go.Heatmap(z=df_missing.values, y=df_percentiles.index, colorscale=[[0, 'rgb(201,201,201)'], [1, 'rgb(201,201,201)']], showscale=False)


    if color_missing == True and outliers==True:
        data = [trace_valid, trace_outlier, trace_missing]
    elif color_missing == True and outliers==False:
        data = [trace_valid, trace_missing]
    elif color_missing == False and outliers==True:
        data = [trace_valid, trace_outlier]
    else: data = [trace_valid]

    layout = go.Layout(yaxis=dict(autorange='reversed', automargin=True))
    fig = go.Figure(data=data, layout=layout)
    return fig

def paste_matrices(matrix1, sigint1, matrix2, sigint2):
    R_function = R(''' text <- function(matrix1, sigint1, matrix2, sigint2) {
                                    paste(signif(matrix1, sigint1), signif(matrix2, sigint2), sep = "<br>")} ''')

    paste_matrix = R_function(matrix1, sigint1, matrix2, sigint2)
    df = R2Py.R_matrix2Py_matrix(paste_matrix, matrix1.rownames, matrix1.colnames)
    return df

def cutreeDynamic(distmatrix, linkagefun, minModuleSize=30, deepSplit=2, pamRespectsDendro=False, distfun=None):
    if distfun is None:
        dist = stats.as_dist(distmatrix)
    else:
        dist = stats.dist(distmatrix, method = distfun)

    R_function = R(''' clusters <- function(dist, distmatrix, linkagefun, minModuleSize, deepSplit, pamRespectsDendro) {
                                        cutreeDynamic(dendro=hclust(dist, method = linkagefun),distM=distmatrix,
                                        deepSplit=deepSplit, pamRespectsDendro= pamRespectsDendro, minClusterSize= minModuleSize)} ''')

    cutree = R_function(dist, distmatrix, linkagefun=linkagefun, minModuleSize=minModuleSize, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro)
    return np.array(cutree)

def plot_complex_dendrogram(df_exp, df_traits, dendro_labels=[], distfun='euclidean', linkagefun='average', subplot='module colors', modColors=dynamicColors, color_missingvals=True):
    dendro_tree = get_dendrogram(df_exp, dendro_labels, distfun=distfun, linkagefun=linkagefun, div_clusters=False)
    plot_dendro = plot_dendrogram(dendro_tree, hang=0.04, cutoff_line=False)

    fig = tls.make_subplots(rows=2, cols=1)

    for i in list(plot_dendro.data):
        fig.append_trace(i, 1, 1)

    layout = go.Layout(width=2000, height=1200, showlegend=False,
                       xaxis=dict(domain=[0, 1], range=[np.min(plot_dendro.layout.xaxis.tickvals)-6,np.max(plot_dendro.layout.xaxis.tickvals)+4], showgrid=False,
                                  zeroline=True, ticks='',automargin=True),
                       yaxis=dict(domain=[0.7, 1], autorange=True, showgrid=False, zeroline=False, ticks='outside', title='Height', automargin=True),
                       xaxis2=dict(domain=[0, 1], autorange=True, showgrid=True, zeroline=False, ticks='', showticklabels=False, automargin=True),
                       yaxis2=dict(domain=[0, 0.64], autorange=True, showgrid=False, zeroline=False, automargin=True))

    if subplot == 'module colors':
        df_col, colors = get_module_color_annotation(dendro_labels, modColors, plot_dendro)
        shapes = plot_dendrogram_guidelines(dendro_tree, plot_dendro)
        trace = go.Heatmap(z=df_col.vals, y=df_col.y, x=df_col.labels, showscale=False, colorscale=colors)
        fig.append_trace(trace, 2, 1)
        fig.layout.update({'shapes':shapes,
                           'yaxis':dict(domain=[0.2, 1]),
                           'yaxis2':dict(domain=[0, 0.19], title='Module colors', ticks='', showticklabels=False)})

    elif subplot == 'heatmap':
        df_percentiles = get_percentiles_heatmap(df_traits, dendro_tree, bydendro=True, bycols=True)
        plot_heat = plot_heatmap(df_percentiles, color_missing=color_missingvals, outliers=False)
        for j in list(plot_heat.data):
            fig.append_trace(j, 2, 1)
            fig.layout.update({'xaxis':dict(ticktext=np.array(plot_dendro.layout.xaxis.ticktext), tickvals=list(plot_dendro.layout.xaxis.tickvals)),
                               'yaxis2':dict(autorange='reversed')})

    return fig













def build_network(df_exp, RsquaredCut=0.8, networkType='unsigned', verbose=2):
    powers = np.arange(1,20,1)
    sft = WGCNA.pickSoftThreshold(df_exp, RsquaredCut=RsquaredCut, powerVector=powers, networkType=networkType, verbose=verbose)

    softPower = sft.rx2('powerEstimate')[0]
    adjacency = WGCNA.adjacency(df_exp, power=softPower, type=networkType)

    TOM = WGCNA.TOMsimilarity(adjacency)
    dissTOM = R("1") - TOM
    return dissTOM

def identify_modules_and_colors(network, minModuleSize=30, deepSplit=2, pamRespectsDendro=False):
    dynamicMods = cutreeDynamic(network, distfun=None, linkagefun="average", minModuleSize=minModuleSize, deepSplit=deepSplit,
                                pamRespectsDendro=pamRespectsDendro)

    unique, counts = np.unique(dynamicMods, return_counts=True)
    module_size = dict(zip(unique, counts))
    dynamicColors= np.array(WGCNA.labels2colors(dynamicMods))
    return module_size, dynamicColors

def get_module_color_annotation(gene_list, module_colors, dendrogram):
    gene_colors = dict(zip(gene_list, module_colors))
    colors_dict = color_list_up.make_color_dict()

    n = len(gene_colors.keys())
    val = 1/(n-1)
    number = 0

    colors = []
    vals = []
    for i in gene_colors.keys():
        name = gene_colors[i]
        color = colors_dict[name]
        n = number
        colors.append([round(n,4), color])
        vals.append((i, round(n,4)))
        number = n+val

    labels = list(dendrogram.layout.xaxis.ticktext)
    y = [1]*len(labels)

    df = pd.DataFrame([labels, y], index=['labels', 'y']).T
    df['vals'] = df['labels'].map(dict(vals))
    return df, colors

def plot_dendrogram_guidelines(Z_tree, dendrogram):
    tickvals = list(dendrogram.layout.xaxis.tickvals)

    keys = ['type', 'x0', 'y0', 'x1', 'y1', 'line']
    line_keys = ['color', 'width', 'dash']
    line_vals = ['rgb(192,192,192)', 0.1, 'dot']
    line = dict(zip(line_keys,line_vals))

    values = []
    for i in tickvals[70::70]:
        values.append(('line', i, 0.3, i, np.max(Z_tree['dcoord'])))

    values = [list(i)+[line] for i in values]
    shapes = []
    for i in values:
        d = dict(zip(keys, i))
        shapes.append(d)
    return shapes
















def plot_complex_dendrogram(dendro_df=datExpr0, subplot_df=datTraits, dendro_labels=[], distfun='euclidean', linkagefun='average', hang=0.04, subplot='module colors', modColors=dynamicColors, color_missingvals=True):
    dendro_tree = get_dendrogram(dendro_df, dendro_labels, distfun=distfun, linkagefun=linkagefun, div_clusters=False)
    plot_dendro = plot_dendrogram(dendro_tree, hang=hang, cutoff_line=False)

    fig = tls.make_subplots(rows=2, cols=1)

    for i in list(plot_dendro.data):
        fig.append_trace(i, 1, 1)

    layout = go.Layout(width=2000, height=1200, showlegend=False,
                       xaxis=dict(domain=[0, 1], range=[np.min(plot_dendro.layout.xaxis.tickvals)-6,np.max(plot_dendro.layout.xaxis.tickvals)+4], showgrid=False,
                                  zeroline=True, ticks='',automargin=True),
                       yaxis=dict(domain=[0.7, 1], autorange=True, showgrid=False, zeroline=False, ticks='outside', title='Height', automargin=True),
                       xaxis2=dict(domain=[0, 1], autorange=True, showgrid=True, zeroline=False, ticks='', showticklabels=False, automargin=True),
                       yaxis2=dict(domain=[0, 0.64], autorange=True, showgrid=False, zeroline=False, automargin=True))

    fig['layout'] = layout

    if subplot == 'module colors':
        df_col, colors = get_module_color_annotation(dendro_labels, bygene=True, module_colors=modColors, dendrogram=plot_dendro)
        shapes = plot_dendrogram_guidelines(dendro_tree, plot_dendro)
        trace = go.Heatmap(z=df_col.vals, y=df_col.y, x=df_col.labels, showscale=False, colorscale=colors)
        fig.append_trace(trace, 2, 1)
        fig.layout.update({'shapes':shapes,
                           'xaxis':dict(showticklabels=False),
                           'yaxis':dict(domain=[0.2, 1]),
                           'yaxis2':dict(domain=[0, 0.19], title='Module colors', ticks='', showticklabels=False)})

    elif subplot == 'heatmap':
        if all(list(subplot_df.columns.map(lambda x: subplot_df[x].between(-1,1, inclusive=True).all()))) != True:
            df_percentiles = get_percentiles_heatmap(subplot_df, dendro_tree, bydendro=True, bycols=True)
        else:
            df_percentiles = df_sort_by_dendrogram(subplot_df, dendro)

        plot_heat = plot_heatmap(df_percentiles, color_missing=color_missingvals, outliers=False)
        for j in list(plot_heat.data):
            fig.append_trace(j, 2, 1)
            fig.layout.update({'xaxis':dict(ticktext=np.array(plot_dendro.layout.xaxis.ticktext), tickvals=list(plot_dendro.layout.xaxis.tickvals)),
                               'yaxis2':dict(autorange='reversed')})

    return fig
