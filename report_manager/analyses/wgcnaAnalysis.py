import os
import numpy as np
import pandas as pd
import scipy as scp
from scipy.cluster.hierarchy import distance, linkage, dendrogram, fcluster
from collections import OrderedDict, defaultdict
from natsort import natsorted, index_natsorted, order_by_index
import urllib.request

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FloatVector
import rpy2.robjects.packages as rpacks
from report_manager import R2Py
pandas2ri.activate()

R = ro.r
R('options(stringsAsFactors = FALSE)')
#R('source("http://bioconductor.org/biocLite.R")')
#R('biocLite(c("GO.db", "preprocessCore", "impute"))')


base = R2Py.call_Rpackage("package", "base")
stats = R2Py.call_Rpackage("package", "stats")
WGCNA = R2Py.call_Rpackage("package", "WGCNA")


def get_data(data, drop_cols_exp=['group', 'sample'], drop_cols_cli=['group', 'biological_sample']):
    wgcna_data = {}
    for i in data:
        df = data[i]
        if i == 'clinical':
            df.drop_duplicates(keep='first', inplace=True)
            df = df.reset_index()
            df.set_index(['subject'], inplace=True)
            df = df.reindex(index=natsorted(df.index))
            df = df.drop(drop_cols_cli, axis=1)
        else:
            df.set_index(['subject'], inplace=True)
            df = df.reindex(index=natsorted(df.index))
            df = df.drop(drop_cols_exp, axis=1)

        wgcna_data[i] = df

    return wgcna_data


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
    data = df2.T

    return data

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

def paste_matrices(matrix1, matrix2):
    a = pandas2ri.ri2py(matrix1)
    b = pandas2ri.ri2py(matrix2)
    text = []
    for i, j in zip(a, b):
        for x, y in zip(i, j):
            text.append(('{:0.2}<br>{:.0e}'.format(x, y)))

    text = np.array(text)
    text.shape = (len(matrix1.rownames), len(matrix1.colnames))
    textMatrix = pd.DataFrame(text, index=matrix1.rownames, columns=matrix1.colnames)
    return textMatrix

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


def build_network(data, softPower=6, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose_merge=3):
    #softPower = pick_softThreshold(data, RsquaredCut=RsquaredCut, networkType=networkType, verbose=network_verbose)
    adjacency = WGCNA.adjacency(data, power=softPower, type=networkType)

    TOM = WGCNA.TOMsimilarity(adjacency)
    dissTOM = pd.DataFrame(R("1") - TOM)
    dissTOM.columns = data.columns
    dissTOM.index = data.columns

    moduleColors = identify_module_colors(dissTOM, minModuleSize=minModuleSize, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro)
    if merge_modules == True:
        MEs, moduleColors = merge_similar_modules(data, moduleColors, MEDissThres=MEDissThres, verbose=verbose_merge)
    else: pass

    return dissTOM, moduleColors

def pick_softThreshold(data, RsquaredCut=0.8, networkType='unsigned', verbose=2):
    powers = np.arange(1,20,1)
    sft = WGCNA.pickSoftThreshold(data, RsquaredCut=RsquaredCut, powerVector=powers, networkType=networkType, verbose=verbose)
    softPower = sft.rx2('powerEstimate')[0]

    return softPower

def identify_module_colors(network, minModuleSize=30, deepSplit=2, pamRespectsDendro=False):
    dynamicMods = cutreeDynamic(network, distfun=None, linkagefun="average", minModuleSize=minModuleSize, deepSplit=deepSplit,
                                pamRespectsDendro=pamRespectsDendro)

    dynamicColors= np.array(WGCNA.labels2colors(dynamicMods))

    return dynamicColors

def calculate_module_eigengenes(df_exp, modColors, softPower=6, dissimilarity=True):
    MEList = WGCNA.moduleEigengenes(df_exp, modColors, softPower=softPower)
    MEs0 = MEList.rx2('eigengenes')
    MEs = WGCNA.orderMEs(MEs0)
    if dissimilarity:
        MEcor = WGCNA.cor(MEs)
        MEcor = R2Py.R_matrix2Py_matrix(MEcor, MEcor.rownames, MEcor.colnames)
        MEDiss = 1 - MEcor
        return MEs, MEDiss
    else:
        return MEs

def merge_similar_modules(df_exp, modColors, MEDissThres=0.25, verbose=3):
    merge = WGCNA.mergeCloseModules(df_exp, modColors, cutHeight=MEDissThres, verbose=verbose)
    mergedColors = merge.rx2('colors')
    mergedMEs = merge.rx2('newMEs')

    return mergedMEs, mergedColors

def calculate_ModuleTrait_correlation(df_exp, df_traits, MEs):
    nFeatures = len(df_exp.columns)
    nSamples = len(df_exp.index)

    df_traits_r = df_traits.copy()
    df_traits_r.columns = df_traits_r.columns.str.replace(' ', 'space')
    df_traits_r.columns = df_traits_r.columns.str.replace('(', 'parentheses1')
    df_traits_r.columns = df_traits_r.columns.str.replace(')', 'parentheses2')

    moduleTraitCor_r = WGCNA.cor(MEs, df_traits_r, use='p')
    moduleTraitPvalue_r = WGCNA.corPvalueStudent(moduleTraitCor_r, nSamples)
    textMatrix = paste_matrices(moduleTraitCor_r, moduleTraitPvalue_r)

    moduleTraitCor = R2Py.R_matrix2Py_matrix(moduleTraitCor_r, moduleTraitCor_r.rownames, moduleTraitCor_r.colnames)
    moduleTraitPvalue = R2Py.R_matrix2Py_matrix(moduleTraitPvalue_r, moduleTraitPvalue_r.rownames, moduleTraitPvalue_r.colnames)

    moduleTraitCor.columns = moduleTraitCor.columns.str.replace('space', ' ')
    moduleTraitPvalue.columns = moduleTraitPvalue.columns.str.replace('space', ' ')
    textMatrix.columns = textMatrix.columns.str.replace('space', ' ')
    moduleTraitCor.columns = moduleTraitCor.columns.str.replace('parentheses1', '(')
    moduleTraitCor.columns = moduleTraitCor.columns.str.replace('parentheses2', ')')
    moduleTraitPvalue.columns = moduleTraitPvalue.columns.str.replace('parentheses1', '(')
    moduleTraitPvalue.columns = moduleTraitPvalue.columns.str.replace('parentheses2', ')')
    textMatrix.columns = textMatrix.columns.str.replace('parentheses1', '(')
    textMatrix.columns = textMatrix.columns.str.replace('parentheses2', ')')

    return moduleTraitCor, textMatrix

def calculate_ModuleMembership(df_exp, MEs):
    nSamples=len(df_exp.index)

    df_exp_r = df_exp.copy()
    df_exp_r.columns = df_exp_r.columns.str.replace('-', 'dash')

    modLabels = [i[2:] for i in list(MEs.colnames)]
    FeatureModuleMembership = base.as_data_frame(WGCNA.cor(df_exp_r, MEs, use='p'))
    MMPvalue = base.as_data_frame(WGCNA.corPvalueStudent(base.as_matrix(FeatureModuleMembership), nSamples))

    FeatureModuleMembership.colnames = ['MM'+str(col) for col in modLabels]
    MMPvalue.colnames = ['p.MM'+str(col) for col in modLabels]

    FeatureModuleMembership = R2Py.R_matrix2Py_matrix(FeatureModuleMembership, FeatureModuleMembership.rownames, FeatureModuleMembership.colnames)
    MMPvalue = R2Py.R_matrix2Py_matrix(MMPvalue, MMPvalue.rownames, MMPvalue.colnames)
    FeatureModuleMembership.index = FeatureModuleMembership.index.str.replace('dash', '-')
    MMPvalue.index = MMPvalue.index.str.replace('dash', '-')

    return FeatureModuleMembership, MMPvalue

def calculate_FeatureTraitSignificance(df_exp, df_trait):
    nSamples=len(df_exp.index)

    df_exp_r = df_exp.copy()
    df_exp_r.columns = df_exp_r.columns.str.replace('-', 'dash')
    df_cli_r = df_trait.copy()
    df_cli_r.columns = df_cli_r.columns.str.replace(' ', 'space')
    df_cli_r.columns = df_cli_r.columns.str.replace('(', 'parentheses1')
    df_cli_r.columns = df_cli_r.columns.str.replace(')', 'parentheses2')

    FeatureTraitSignificance = base.as_data_frame(WGCNA.cor(df_exp_r, df_cli_r, use='p'))
    FSPvalue = base.as_data_frame(WGCNA.corPvalueStudent(base.as_matrix(FeatureTraitSignificance), nSamples))

    FeatureTraitSignificance.colnames = ['GS.'+str(col) for col in FeatureTraitSignificance.colnames]
    FSPvalue.colnames = ['p.GS.'+str(col) for col in FSPvalue.colnames]

    FeatureTraitSignificance = R2Py.R_matrix2Py_matrix(FeatureTraitSignificance, FeatureTraitSignificance.rownames, FeatureTraitSignificance.colnames)
    FSPvalue = R2Py.R_matrix2Py_matrix(FSPvalue, FSPvalue.rownames, FSPvalue.colnames)

    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('space', ' ')
    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('parentheses1', '(')
    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('parentheses2', ')')
    FeatureTraitSignificance.index = FeatureTraitSignificance.index.str.replace('dash', '-')
    FSPvalue.columns = FSPvalue.columns.str.replace('space', ' ')
    FSPvalue.columns = FSPvalue.columns.str.replace('parentheses1', '(')
    FSPvalue.columns = FSPvalue.columns.str.replace('parentheses2', ')')
    FSPvalue.index = FSPvalue.index.str.replace('dash', '-')

    return FeatureTraitSignificance, FSPvalue

def get_FeaturesPerModule(df_exp, modColors, mode='dictionary'):
    if mode == 'dataframe':
        features_per_module = dict(zip(df_exp.columns, modColors))
        features_per_module = pd.DataFrame(list(features_per_module.items()), columns=['name', 'modColor'])

    elif mode == 'dictionary':
        features_per_module = defaultdict(list)
        for k, v in zip(modColors, df_exp.columns):
            features_per_module[k].append(v)
        features_per_module = dict((k, v) for k, v in features_per_module.items())

    return features_per_module

def get_ModuleFeatures(df_exp, modColors, modules=[]):
    allfeatures = get_FeaturesPerModule(df_exp, modColors, mode='dictionary')

    modules = modules
    selectfeatures = [allfeatures[x] for x in modules]
    return selectfeatures

def get_EigengenesTrait_correlation(module_eigengenes, df_trait):
    df_trait_r =  df_trait.copy()
    df_trait_r.columns = df_trait_r.columns.str.replace(' ', 'space')
    df_trait_r.columns = df_trait_r.columns.str.replace('(', 'parentheses1')
    df_trait_r.columns = df_trait_r.columns.str.replace(')', 'parentheses2')
    df_trait_r.columns = df_trait_r.columns.str.replace('/', 'slash')

    MET = WGCNA.orderMEs(base.cbind(module_eigengenes, df_trait_r))
    METcor = WGCNA.cor(MET, use='p')
    METcor = R2Py.R_matrix2Py_matrix(METcor, METcor.rownames, METcor.colnames)

    METcor.columns = METcor.columns.str.replace('space', ' ')
    METcor.columns = METcor.columns.str.replace('parentheses1', '(')
    METcor.columns = METcor.columns.str.replace('parentheses2', ')')
    METcor.columns = METcor.columns.str.replace('slash', '/')
    METcor.index = METcor.index.str.replace('space', ' ')
    METcor.index = METcor.index.str.replace('parentheses1', '(')
    METcor.index = METcor.index.str.replace('parentheses2', ')')
    METcor.index = METcor.index.str.replace('slash', '/')

    METDiss = 1 - METcor

    return METDiss, METcor
