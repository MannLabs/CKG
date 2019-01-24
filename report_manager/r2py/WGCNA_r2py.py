%load_ext rpy2.ipython
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
utils = R2Py.call_Rpackage("package", "utils")
stats = R2Py.call_Rpackage("package", "stats")
biocm = R2Py.call_Rpackage("package", "BiocManager")
WGCNA = R2Py.call_Rpackage("package", "WGCNA")



def get_preproc_data(filename):
    df = pd.read_hdf(filename, 'preprocessed')
    df_py = df.drop('group', axis=1)
    df_py.set_index('sample', inplace = True)
    df_py = df_py.reindex(index = natsorted(df_py.index))

    return df_py

def get_goodSamplesGenes(df_py, verbose = 1):
    df_r = pandas2ri.py2ri_pandasdataframe(df_py)
    gsg = WGCNA.goodSamplesGenes(df_r, verbose = verbose)

    if bool(gsg.rx2("allOK")) != True:
        samples = pd.DataFrame(np.array(gsg.rx2("goodSamples")), columns=[gsg.names[1]])
        samples['samples'] = R.rownames(df_r)
        genes = pd.DataFrame(np.array(gsg.rx2("goodGenes")), columns=[gsg.names[0]])
        genes['genes'] = R.names(df_r)

        goodSamples = samples[samples['goodSamples'] == 1]['samples'].tolist()
        goodGenes = genes[genes['goodGenes'] == 1]['genes'].tolist()

        df2_py = R2Py.R_matrix2Py_matrix(df_r)
        df2_py = data_py.loc[(df2_py.index.isin(goodSamples)), (df2_py.columns.isin(goodGenes))]

    else:
        df2_py = R2Py.R_matrix2Py_matrix(df_r)

    return df2_py

def get_clusters_elements(linkage_matrix, fcluster_method, value, labels):
    clust = fcluster(linkage_matrix, fcluster_method, value)
    clusters = defaultdict(list)
    for i, j in zip(clust, labels):
        clusters[i].append(j)

    return clusters

def get_dendrogram(df_py, dist_method='euclidean', link_methods='average', labels=df_py.index, dendro_clusters=True, fcluster_method='distance', value=15):
    dist = np.asarray(stats.dist(df_py, method = dist_method))
    Z = linkage(dist, method = linkage_methods)
    Z_dendrogram = dendrogram(Z, no_plot = True, labels = labels)

    if dendro_clusters == True:
        clust = fcluster(Z, fcluster_method, value)
        clusters = defaultdict(list)
        for i, j in zip(clust, labels):
            clusters[i].append(j)

        return Z_dendrogram, clusters

    else: return Z_dendrogram

def filter_df_by_cluster(df_py, clusters, number):
    df2_py = df_py[df_py.index.isin(clusters[number])]

    return df2_py

def get_percentiles_heatmap(df_py, bycols=False):
    p =pd.DataFrame(index=df_py.index,columns=df_py.columns)

    if bycols != True:
        for j in df_py.index:
            for i in df_py.columns:
                pct = (df_py.loc[j,i] - np.nanmin(df_py.loc[j,:])) / ((np.nanmax(df_py.loc[j, :]) - np.nanmin(df_py.loc[j, :])) * 1.)
                pct = pct - (pct - 0.5) * 1. / 40 #have to rescale it to account for endpoints of cmaps
                p.loc[j,i] = pct
    else:
        for i in df_py.index:
            for j in df_py.columns:
                pct = (df_py.loc[i,j] - np.nanmin(df_py.loc[:,j])) / ((np.nanmax(df_py.loc[:,j]) - np.nanmin(df_py.loc[:,j])) * 1.)
                pct = pct - (pct - 0.5) * 1. / 40 #have to rescale it to account for endpoints of cmaps
                p.loc[i,j] = pct

    return p

def get_miss_values_df(df_py):
    df = df_py.copy()
    df = df.isnull().astype('int')
    df = df.replace(0, np.nan)

    return df

def plot_heatmap(df_percentiles, color_missing=True, width, height):
    trace_valid = go.Heatmap(z=df_percentiles.values, y=df_percentiles.index, colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,51,0)']], showscale=False)

    if color_missing:
        df_missing = get_miss_values_df(df_percentiles)
        trace_missing = go.Heatmap(z=df_missing.values, y=df_percentiles.index, colorscale=[[0, 'rgb(201,201,201)'], [1, 'rgb(201,201,201)']], showscale=False)
        data = [trace_valid, trace_missing]
    else: data = [trace_valid]

    fig = go.Figure(data=data, layout=go.Layout(width=width, height=height))

    return fig

def get_blockwise_network_modules(df_py, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro, saveTOMs, saveTOMFileBase, verbose):
    network = R(''' net <- function(df_py, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro,
                                    saveTOMs, saveTOMFileBase, verbose) {
                                    blockwiseModules(df_py, power=power, minModuleSize=minModuleSize, reassignThreshold = reassignThreshold,
                                    mergeCutHeight = mergeCutHeight, numericLabels = numericLabels, pamRespectsDendro = pamRespectsDendro,
                                    saveTOMs = saveTOMs, saveTOMFileBase = saveTOMFileBase, verbose = verbose)}''')

    moduleLabels_r = network.rx2('colors')
    moduleGenes_r = network.rx2('blockGenes')[0]
    geneTree_r = network.rx2('dendrograms')[0]
    MEs_r = network.rx2('MEs')
    moduleColors_r = WGCNA.labels2colors(network.rx2('colors'))

    return moduleLabels_r, moduleGenes_r, geneTree_r, MEs_r, moduleColors_r

def paste_matrices(matrix1, sigint1, matrix2, sigint2):
    function = R(''' text <- function(matrix1, sigint1, matrix2, sigint2) {
                                    paste(signif(matrix1, sigint1), signif(matrix2, sigint2), sep = "<br>")} ''')
    return function

def plot_modules_traits_correlation(df_py, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro, saveTOMs, saveTOMFileBase, verbose):
    nGenes = len(df_py.columns)
    nSamples = len(df_py.index)

    network = get_blockwise_network_modules(df_py, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro, saveTOMs, saveTOMFileBase, verbose)


def get_modules_colorscale(MEs_r):
    MEs_py = R2Py.R_matrix2Py_matrix(MEs_r)

    n = len(MEs_py.columns)
    if n % 2 == 0: pass
    else: n = n-1

    val = 1/n
    number = 0

    colorscale = []
    vals = []
    for i in MEs_py.columns:
        name = i.split('ME')[1]
        color = colors_dict[name]
        n = number
        colorscale.append([round(n,2), color])
        vals.append(round(n,2))
        number = n+val

    return colorscale

def get_eigengenes_modules(df_py):
    MEs0 = WGCNA.moduleEigengenes(df_py, moduleColors).rx2('eigengenes')
