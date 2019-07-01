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

#Call R
R = ro.r
R('options(stringsAsFactors = FALSE)')
#R('source("http://bioconductor.org/biocLite.R")')
#R('biocLite(c("GO.db", "preprocessCore", "impute"))')


#Call R packages
base = R2Py.call_Rpackage("package", "base")
stats = R2Py.call_Rpackage("package", "stats")
WGCNA = R2Py.call_Rpackage("package", "WGCNA")


def get_data(data, drop_cols_exp=['group', 'sample'], drop_cols_cli=['group', 'biological_sample', 'index']):
    """ 
    This function cleanes up and formats experimental and clinical data into similarly shaped dataframes.
    
    Args:
        data: dictionary with processed clinical and proteomics datasets
        drop_cols_exp: list of columns to drop from processed experimental (protemics/rna-seq/dna-seq) dataframe
        drop_cols_cli: list of columns to drop from processed clinical dataframe
    
    Returns:
        Dictionary with experimental and clinical dataframes (keys are the same as in the input dictionary).  
    """
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
            df = df.reset_index()
            df.set_index(['subject'], inplace=True)
            df = df.reindex(index=natsorted(df.index))
            df = df.drop(drop_cols_exp, axis=1)

        wgcna_data[i] = df

    return wgcna_data


def get_dendrogram(df, labels, distfun='euclidean', linkagefun='average', div_clusters=False, fcluster_method='distance', fcluster_cutoff=15):
    """ 
    This function calculates the distance matrix and performs hierarchical cluster analysis on a set of dissimilarities and methods for analyzing it.
   
    Args:
        df: pandas dataframe with samples/subjects as index and features as columns
        labels: list of labels for the leaves of the tree
        distfun: distance measure to be used ('euclidean', 'maximum', 'manhattan', 'canberra', 'binary', 'minkowski' or 'jaccard')
        linkagefun: hierarchical/agglomeration method to be used ('single', 'complete', 'average', 'weighted', 'centroid', 'median' or 'ward')
        div_clusters: dividing dendrogram leaves into clusters (True or False)
        fcluster_method: criterion to use in forming flat clusters
        fcluster_cutoff: maximum cophenetic distance between observations in each cluster
    
    Returns:
        Dictionary of data structures computed to render the dendrogram. Keys: 'icoords', 'dcoords', 'ivl' and 'leaves'. If div_clusters is used, it will also return a dictionary of each cluster and respective leaves.
    """
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


def get_clusters_elements(linkage_matrix, fcluster_method, fcluster_cutoff, labels):
    """ 
    This function implements the generation of flat clusters from an hierarchical clustering with the same interface as scipy.cluster.hierarchy.fcluster.
    
    Args:
        linkage_matrix: hierarchical clustering encoded with a linkage matrix.
        fcluster_method: criterion to use in forming flat clusters ('inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit')
        fcluster_cutoff: maximum cophenetic distance between observations in each cluster.
        labels: list of labels for the leaves of the dendrogram.
    
    Returns:
        A dictionary where keys are the cluster numbers and values are the dendrogram leaves.
    """
    clust = fcluster(linkage_matrix, fcluster_cutoff, fcluster_method)
    clusters = defaultdict(list)
    for i, j in zip(clust, labels):
        clusters[i].append(j)
    return clusters

def filter_df_by_cluster(df, clusters, number):
    """ 
    Select only the members of a defined cluster.
    
    Args:
        df: pandas dataframe with samples/subjects as index and features as columns
        clusters: clusters dictionary from get_dendrogram function if div_clusters option was True.
        number: cluster number (key)
    
    Returns:
        Pandas dataframe with all the features (columns) and samples/subjects belonging to the defined cluster (index).
    """
    return df[df.index.isin(clusters[number])]

def df_sort_by_dendrogram(df, Z_dendrogram):
    """ 
    Reorders pandas dataframe by index and according to the dendrogram list of leaf nodes labels.

    Args:
        df: pandas dataframe with the labels to be reordered as index
        Z_dendrogram: dictionary of data structures computed to render the dendrogram. Keys: 'icoords', 'dcoords', 'ivl' and 'leaves'.
    
    Returns:
        Reordered pandas dataframe.
    """
    data = df.copy()
    data.index = pd.CategoricalIndex(data.index, categories=Z_dendrogram['ivl'])
    data.sort_index(level=0, inplace=True)
    return data

def get_percentiles_heatmap(df, Z_dendrogram, bydendro= True, bycols=False):
    """ 
    This function transforms the absolute values in each row or column (option 'bycols') into relative values.
    
    Args:
        df: pandas dataframe with samples/subjects as index and features as columns.
        Z_dendrogram: dictionary of data structures computed to render the dendrogram. Keys: 'icoords', 'dcoords', 'ivl' and 'leaves'.
        bydendro: if labels should be ordered according to dendrogram list of leaf nodes labels set to True, otherwise set to False.
        bycols: relative values calculated across rows (samples) then set to False. Calculation performed across columns (features) set to True.
    
    Returns:
        Pandas dataframe.
    """
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

def get_miss_values_df(data):
    """ 
    Proccesses pandas dataframe so missing values can be plotted in heatmap with specific color.

    Args:
        data: pandas dataframe.
    
    Returns:
        Pandas dataframe with missing values as integer 1, and originally valid values as NaN.
    """
    df = data.copy()
    df = df.isnull().astype('int')
    df = df.replace(0, np.nan)
    return df

def paste_matrices(matrix1, matrix2, rows, cols):
    """ 
    Takes two R matrices with analog shapes and concatenates each value in matrix 1 with corresponding one in matrix 2, returning a single pandas dataframe.

    Args:
        matrix1: R matrix
        matrix2: R matrix
    
    Returns:
        Pandas dataframe.
    """
    #a = pandas2ri.ri2py(matrix1)
    #b = pandas2ri.ri2py(matrix2)
    
    text = []
    for i, j in zip(matrix1, matrix2):
        for x, y in zip(i, j):
            text.append(('{:0.2}<br>{:.0e}'.format(x, y)))
    
    text = np.array(text)
    text.shape = (matrix1.shape[0], matrix1.shape[1])
    textMatrix = pd.DataFrame(text, index=rows, columns=cols)
    return textMatrix

def cutreeDynamic(distmatrix, linkagefun='average', minModuleSize=30, method='hybrid', deepSplit=2, pamRespectsDendro=False, distfun=None):
    """
    This function implements the R cutreeDynamic wrapper in Python, provinding an access point for methods of adaptive branh pruning of hierarchical clustering dendrograms.

    Args:
        data: pandas dataframe.
        distfun: distance measure to be used ('euclidean', 'maximum', 'manhattan', 'canberra', 'binary', 'minkowski' or 'jaccard').
        linkagefun: hierarchical/agglomeration method to be used ('single', 'complete', 'average', 'weighted', 'centroid', 'median' or 'ward').
        minModuleSize: minimum module size.
        method: method to use ('hybrid' or 'tree').
        deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
        pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.

    Returns:
        Numpy array of numerical labels giving assignment of objects to modules. Unassigned objects are labeled 0, the largest module has label 1, next largest 2 etc.
    """
    #if distfun is None:
    #    dist = stats.as_dist(distmatrix)
    #else:
    #    dist = stats.dist(distmatrix, method = distfun)
          
    R_function = R(''' clusters <- function(distmatrix, linkagefun, minModuleSize, method, deepSplit, pamRespectsDendro) {
                                        cutreeDynamic(dendro=hclust(as.dist(distmatrix), method = linkagefun), method=method, distM=distmatrix, 
                                        deepSplit=deepSplit, pamRespectsDendro= pamRespectsDendro, minClusterSize= minModuleSize)} ''')
    
    cutree = R_function(distmatrix, linkagefun=linkagefun, minModuleSize=minModuleSize, method=method, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro)
    
    return np.array(cutree)


def build_network(data, softPower=6, networkType='unsigned', minModuleSize=30, method='hybrid', deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0):
    """ 
    Weighted gene network construction and module detection. Calculates co-expression similarity and adjacency, topological overlap matrix (TOM) and clusters features in modules.

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        softPower: soft-thresholding power.
        networkType: network type ('unsigned', 'signed', 'signed hybrid', 'distance').
        minModuleSize: minimum module size.
        deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
        pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.
        merge_modules: if True, very similar modules are merged.
        MEDissThres: maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging.
        verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    
    Returns:
         Tuple with TOM dissimilarity pandas dataframe, and R/rpy2 vector array with module colors per experimental feature.
    """

    #Calculate adjacencies
    adjacency = WGCNA.adjacency(data, power=softPower, type=networkType)

    #Transforms the adjacency into topological overlap matrix (TOM)
    TOM = WGCNA.TOMsimilarity(adjacency, verbose = verbose)
    #Calculates the corresponding dissimilarity matrix
    dissTOM = pd.DataFrame(R("1") - TOM)
    dissTOM.columns = data.columns
    dissTOM.index = data.columns

    #Identify co-expression modules
    moduleColors = identify_module_colors(dissTOM, minModuleSize=minModuleSize, method=method, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro)

    #Merge modules whose expression profiles are very similar
    if merge_modules == True:
        MEs, moduleColors = merge_similar_modules(data, moduleColors, MEDissThres=MEDissThres, verbose=verbose)
    else: pass

    return dissTOM, moduleColors

def pick_softThreshold(data, RsquaredCut=0.8, networkType='unsigned', verbose=0):
    """ 
    Analysis of scale free topology for multiple soft thresholding powers. Aids the user in choosing a proper soft-thresholding power for network construction.
    
    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        RsquaredCut: desired minimum scale free topology fitting index R^2.
        networkType: network type ('unsigned', 'signed', 'signed hybrid', 'distance').
        verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    
    Returns:
        Estimated appropriate soft-thresholding power: the lowest power for which the scale free topology fit R^2 exceeds RsquaredCut.
    """
    powers = np.arange(1,20,1)
    sft = WGCNA.pickSoftThreshold(data, RsquaredCut=RsquaredCut, powerVector=powers, networkType=networkType, verbose=verbose)
    softPower = sft.rx2('powerEstimate')[0]
    return softPower

def identify_module_colors(matrix, minModuleSize=30, method='hybrid', deepSplit=2, pamRespectsDendro=False):
    """
    Identifies co-expression modules and converts the numeric labels into colors.
    
    Args:
        matrix: dissimilarity structure as produced by R.stats dist.
        minModuleSize: minimum module size.
        deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
        pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.
    
    Returns:
        Numpy array of strings with module color of each experimental feature.
    """
    dynamicMods = cutreeDynamic(matrix, distfun=None, linkagefun="average", minModuleSize=minModuleSize, method=method, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro)
    dynamicColors= np.array(WGCNA.labels2colors(dynamicMods))
    return dynamicColors

def calculate_module_eigengenes(data, modColors, softPower=6, dissimilarity=True):
    """ 
    Calculates modules eigengenes to quantify co-expression similarity of entire modules.

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        modColors: vector (numeric, character or a factor) attributing module colors to each feature in the experimental dataframe.
        softPower: soft-thresholding power.
        dissimilarity: calculates dissimilarity of module eigengenes.
    
    Returns:
        R/rpy2 dataframe with calculated module eigengenes. If dissimilarity is set to True, returns a tuple with two dataframes, the first in R/rpy2 with the module eigengenes and the second in pandas with the eigengenes dissimilarity. 
    """
    MEList = WGCNA.moduleEigengenes(data, modColors, softPower=softPower)
    MEs0 = MEList.rx2('eigengenes')
    MEs = WGCNA.orderMEs(MEs0, verbose=0)
    if dissimilarity:
        MEcor = WGCNA.cor(MEs, verbose=0)
        MEcor = R2Py.R_matrix2Py_matrix(MEcor, MEcor.rownames, MEcor.colnames)
        MEDiss = 1 - MEcor
        return MEs, MEDiss
    else:
        return MEs

def merge_similar_modules(data, modColors, MEDissThres=0.25, verbose=0):
    """ 
    Merges modules in co-expression network that are too close as measured by the correlation of their eigengenes.

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        modColors: vector (numeric, character or a factor) attributing module colors to each feature in the experimental dataframe.
        MEDissThres: maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging.
        verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    
    Returns:
        Tuple containing R dataframe with eigengenes of the new merged modules, and R charater array with module colors of each expeirmental feature.
    """
    merge = WGCNA.mergeCloseModules(data, modColors, cutHeight=MEDissThres, verbose=verbose)
    mergedColors = merge.rx2('colors')
    mergedMEs = merge.rx2('newMEs')
    return mergedMEs, mergedColors

def calculate_ModuleTrait_correlation(df_exp, df_traits, MEs):
    """ 
    Correlates eigengenes with external traits in order to identify the most significant module-trait associations.

    Args:
        df_exp: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        df_traits: pandas dataframe containing clinical data, with samples/subjects as rows and clinical traits as columns.
        MEs: module eigengenes.
    
    Returns:
        Tuple with two pandas datafames, first the correlation between all module eigengenes and all clinical traits, second a dataframe with concatenated correlation and p-value used for heatmap annotation.
    """
    nFeatures = len(df_exp.columns)
    nSamples = len(df_exp.index)

    df_traits_r = df_traits.copy()
    df_traits_r.columns = df_traits_r.columns.str.replace(' ', 'space')
    df_traits_r.columns = df_traits_r.columns.str.replace('(', 'parentheses1')
    df_traits_r.columns = df_traits_r.columns.str.replace(')', 'parentheses2')
    moduleTraitCor_r = WGCNA.cor(MEs, df_traits_r, use='p', verbose=0)
    moduleTraitPvalue_r = WGCNA.corPvalueStudent(moduleTraitCor_r, nSamples)
    textMatrix = paste_matrices(moduleTraitCor_r, moduleTraitPvalue_r, MEs.columns, df_traits_r.columns)
    
    moduleTraitCor = pd.DataFrame(moduleTraitCor_r, index=MEs.columns, columns=df_traits_r.columns)
    moduleTraitPvalue = pd.DataFrame(moduleTraitPvalue_r, index=MEs.columns, columns=df_traits_r.columns)

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

def calculate_ModuleMembership(data, MEs):
    """ 
    For each module, calculates the correlation of the module eigengene and the feature expression profile (quantitative measure of module membership (MM)). 

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        MEs: module eigengenes.
    
    Returns:
        Tuple with two pandas dataframes, one with MM correlations and another with p-values.
    """
    nSamples=len(data.index)
    
    data_r = data.copy()
    data_r.columns = data_r.columns.str.replace('~', 'dash')

    modLabels = [i[2:] for i in list(MEs.columns)]
    FeatureModuleMembership = base.as_data_frame(WGCNA.cor(data_r, MEs, use='p', verbose=0))
    MMPvalue = base.as_data_frame(WGCNA.corPvalueStudent(base.as_matrix(FeatureModuleMembership), nSamples))

    FeatureModuleMembership.columns = ['MM'+str(col) for col in modLabels]
    MMPvalue.columns = ['p.MM'+str(col) for col in modLabels]

    #FeatureModuleMembership = R2Py.R_matrix2Py_matrix(FeatureModuleMembership, FeatureModuleMembership.index, FeatureModuleMembership.columns)
    #MMPvalue = R2Py.R_matrix2Py_matrix(MMPvalue, MMPvalue.rownames, MMPvalue.colnames)
    FeatureModuleMembership.index = data_r.columns.str.replace('dash', '~')
    MMPvalue.index = MMPvalue.index.str.replace('dash', '~')

    return FeatureModuleMembership, MMPvalue

def calculate_FeatureTraitSignificance(df_exp, df_traits):
    """ 
    Quantifies associations of individual experimental features with the measured clinical traits, by defining Feature Significance (FS) as the absolute value of the correlation between the feature and the trait.

    Args:
        df_exp: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        df_traits: pandas dataframe containing clinical data, with samples/subjects as rows and clinical traits as columns.
    
    Returns:
        Tuple with two pandas dataframes, one with FS correlations and another with p-values.
    """
    nSamples=len(df_exp.index)

    df_exp_r = df_exp.copy()
    df_exp_r.columns = df_exp_r.columns.str.replace('~', 'dash')
    df_cli_r = df_traits.copy()
    df_cli_r.columns = df_cli_r.columns.str.replace(' ', 'space')
    df_cli_r.columns = df_cli_r.columns.str.replace('(', 'parentheses1')
    df_cli_r.columns = df_cli_r.columns.str.replace(')', 'parentheses2')

    FeatureTraitSignificance = base.as_data_frame(WGCNA.cor(df_exp_r, df_cli_r, use='p', verbose=0))
    FSPvalue = base.as_data_frame(WGCNA.corPvalueStudent(base.as_matrix(FeatureTraitSignificance), nSamples))

    FeatureTraitSignificance.columns = ['GS.'+str(col) for col in df_cli_r.columns]
    FSPvalue.columns = ['p.GS.'+str(col) for col in df_cli_r.columns]

    #FeatureTraitSignificance = R2Py.R_matrix2Py_matrix(FeatureTraitSignificance, FeatureTraitSignificance.rownames, FeatureTraitSignificance.colnames)
    #FSPvalue = R2Py.R_matrix2Py_matrix(FSPvalue, FSPvalue.rownames, FSPvalue.colnames)

    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('space', ' ')
    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('parentheses1', '(')
    FeatureTraitSignificance.columns = FeatureTraitSignificance.columns.str.replace('parentheses2', ')')
    FeatureTraitSignificance.index = df_exp_r.columns.str.replace('dash', '~')
    FSPvalue.columns = df_cli_r.columns.str.replace('space', ' ')
    FSPvalue.columns = FSPvalue.columns.str.replace('parentheses1', '(')
    FSPvalue.columns = FSPvalue.columns.str.replace('parentheses2', ')')
    FSPvalue.index = df_exp_r.columns.str.replace('dash', '~')

    return FeatureTraitSignificance, FSPvalue

def get_FeaturesPerModule(data, modColors, mode='dictionary'):
    """ 
    Groups all experimental features by the co-expression module they belong to.

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        modColors: vector (numeric, character or a factor) attributing module colors to each feature in the experimental dataframe.
        mode: type of the value returned by the function ('dictionary' or 'dataframe').
    
    Returns:
        Depending on selected mode, returns a dictionary or dataframe with module color per experimental feature.
    """
    if mode == 'dataframe':
        features_per_module = dict(zip(data.columns, modColors))
        features_per_module = pd.DataFrame(list(features_per_module.items()), columns=['name', 'modColor'])
        
    elif mode == 'dictionary':
        features_per_module = defaultdict(list)
        for k, v in zip(modColors, data.columns):
            features_per_module[k].append(v)
        features_per_module = dict((k, v) for k, v in features_per_module.items())
    return features_per_module

def get_ModuleFeatures(data, modColors, modules=[]):
    """ 
    Groups and returns a list of the experimental features clustered in specific co-expression modules.

    Args:
        data: pandas dataframe containing experimental data, with samples/subjects as rows and features as columns.
        modColors: vector (numeric, character or a factor) attributing module colors to each feature in the experimental dataframe.
        modules: list of module colors of interest.
    
    Returns:
        List of lists with experimental features in each selected module.
    """
    allfeatures = get_FeaturesPerModule(data, modColors, mode='dictionary')

    modules = modules
    selectfeatures = [allfeatures[x] for x in modules]
    return selectfeatures

def get_EigengenesTrait_correlation(MEs, data):
    """ 
    Eigengenes are used as representative profiles of the co-expression modules, and correlation between them is used to quantify module similarity.
    Clinical traits are added to the eigengenes to see how the traits fir into the eigengen network.

    Args:
        MEs: module eigengenes.
        data: pandas dataframe containing clinical data, with samples/subjects as rows and clinical traits as columns.
    
    Returns:
        Tuple with two pandas dataframes, one with features and traits recalculates module eigengenes dissimilarity, and another with all the overall correlations.
    """
    df_traits_r =  data.copy()
    df_traits_r.columns = df_traits_r.columns.str.replace(' ', 'space')
    df_traits_r.columns = df_traits_r.columns.str.replace('(', 'parentheses1')
    df_traits_r.columns = df_traits_r.columns.str.replace(')', 'parentheses2')
    df_traits_r.columns = df_traits_r.columns.str.replace('/', 'slash')
    
    
    MET = WGCNA.orderMEs(base.cbind(MEs, df_traits_r), verbose=0)
    METcor = WGCNA.cor(MET, use='p', verbose=0)
    METcor = pd.DataFrame(METcor, MET.columns, MET.columns)

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
