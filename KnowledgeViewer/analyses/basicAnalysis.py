import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn import preprocessing
import statsmodels.stats.multitest as multi
from scipy import stats
import numpy as np
import math
from predictive_imputer import predictive_imputer

def extract_number_missing(df, conditions, missing_max):
    if conditions is None:
        groups = data.loc[:, data.notnull().sum(axis = 1) >= missing_max]
    else:
        groups = data.copy()
        groups = groups.drop(["sample"], axis = 1)
        groups = data.set_index("group").notnull().groupby(level=0).sum(axis = 1)
        groups = groups[groups>=missing_max]
    
    groups = groups.dropna(how='all', axis=1)

    return list(groups.columns)

def extract_percentage_missing(data, conditions, missing_max):
    if conditions is None:
        groups = data.loc[:, data.isnull().mean() <= missing_max].index
    else:
        groups = data.copy()
        groups = groups.drop(["sample"], axis = 1)
        groups = data.set_index("group")
        groups = groups.isnull().groupby(level=0).mean()
        groups = groups[groups<=missing_max]
    
    groups = groups.dropna(how='all', axis=1)
        
    return list(groups.columns)

def imputation_KNN(data):
    df = data.copy()
    X = np.array(df.loc[:,df.isnull().any()].values, dtype=np.float64)
    imp = predictive_imputer.PredictiveImputer(f_model="KNN")
    X_trans = imp.fit(X).transform(X.copy())

    missingdata_df = df.columns[df.isnull().any()].tolist()
    dfm = pd.DataFrame(X_trans, index =list(df.index), columns = missingdata_df)
    df.update(dfm)
    
    return df

def imputation_normal_distribution(data, shift = 1.8, nstd = 0.3):
    data_imputed = data.copy()
    for i in data_imputed.loc[:, data_imputed.isnull().any()]:
        missing = data_imputed[i].isnull()
        std = data_imputed[i].std()
        mean = data_imputed[i].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        data_imputed.loc[missing, i] = np.random.normal(mu, sigma, size=len(data_imputed[missing]))
    return data_imputed


def polish_median_normalization(data, max_iter = 10):
    mediandf = data.copy()
    for i in range(max_iter):
        col_median = mediandf.median(axis= 0)
        row_median = mediandf.median(axis = 1)
        if row_median.mean() == 0 and col_median.mean() ==0:
            break
            
        mediandf = mediandf.sub(row_median, axis=0)
        mediandf = mediandf.sub(col_median, axis=1)
        
    normData = data - mediandf
    
    return normData

def quantile_normalization(data):
    rank_mean = data.stack().groupby(data.rank(method='first').stack().astype(int)).mean()
    normdf = data.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    
    return normdf

def linear_normalization(data, method = "l1", axis = 0):
    normvalues = preprocessing.normalize(data.fillna(0).values, norm=method, axis=axis, copy=True, return_norm=False)
    normdf = pd.DataFrame(normvalues, index = data.index, columns = data.columns)
    
    return normdf

def remove_group(data):
    data.drop(['group'], axis=1)
    return data

def get_measurements_ready(data, imputation = True, method = 'distribution', missing_method = 'percentage', missing_max = 0.3):
    df = data.copy()
    conditions = df.group.unique()
    df = df.set_index(['group','sample'])
    df = df.pivot_table(values='LFQ_intensity', index=df.index, columns='identifier', aggfunc='first')
    df = df.reset_index()
    df[['group', 'sample']] = df["index"].apply(pd.Series)
    df = df.drop(["index"], axis=1)
    aux = ['group', 'sample']
    if missing_method == 'at_least_x_per_group':
        aux.extend(extract_number_missing(df, conditions, missing_max))
    elif missing_method == 'percentage':
        aux.extend(extract_percentage_missing(df, conditions, missing_max))  
    df = df[aux]
    if imputation:
        if method == "KNN":
            df = imputation_KNN(df)
        elif method == "distribution":
            df = imputation_normal_distribution(df, shift = 1.8, nstd = 0.3)
        else:
            sys.exit()

    return df


def runPCA(data, components = 2):
    result = {}
    df = data.copy()
    df = df.drop(['sample'], axis=1)
    df = df.set_index('group')
    X = df.values
    y = df.index
    pca = PCA(n_components=components)
    pca.fit(X)
    X = pca.transform(X)
    var_exp = pca.explained_variance_ratio_
    args = {"x_title":"PC1"+" ({0:.2f})".format(var_exp[0]),"y_title":"PC2"+" ({0:.2f})".format(var_exp[1])}
    if components == 2:
        resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
        resultDf = resultDf.reset_index()
        resultDf.columns = ["name", "x", "y"]
    if components > 2:
        args.update({"z_title":"PC3"+str(var_exp[2])})
        resultDf = pd.DataFrame(X, index = y)
        resultDf = resultDf.reset_index()
        cols = []
        if len(components)>3:
            cols = resultDf.columns[4:]
        resultDf.columns = ["name", "x", "y", "z"] + cols
    result['pca'] = resultDf
    return result, args

def runTSNE(data, components=2, perplexity=40, n_iter=1000, init='pca'):
    result = {}
    df = data.copy()
    df = df.drop(['sample'], axis=1)
    df = df.set_index('group')
    X = df.values
    y = df.index
    
    tsne = TSNE(n_components=components, verbose=2, perplexity=perplexity, n_iter=n_iter, init=init)
    X = tsne.fit_transform(X)
    args = {"x_title":"C1","y_title":"C2"}
    if components == 2:
        resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
        resultDf = resultDf.reset_index()
        resultDf.columns = ["name", "x", "y"]
    if components > 2:
        args.update({"z_title":"C3"})
        resultDf = pd.DataFrame(X, index = y)
        resultDf = resultDf.reset_index()
        cols = []
        if len(components)>4:
            cols = resultDf.columns[4:]
        resultDf.columns = ["name", "x", "y", "z"] + cols
    result['tsne'] = resultDf
    return result, args
    
def runUMAP(data, n_neighbors=10, min_dist=0.3, metric='cosine'):
    result = {}
    df = data.copy()
    df = df.drop(['sample'], axis=1)
    df = df.set_index('group')
    X = df.values
    y = df.index

    X = umap.UMAP(n_neighbors=10, min_dist=0.3, metric= metric).fit_transform(X)
    args = {"x_title":"C1","y_title":"C2"}
    resultDf = pd.DataFrame(X, index = y)
    resultDf = resultDf.reset_index()
    cols = []
    if len(resultDf.columns)>3:
            cols = resultDf.columns[3:]
    resultDf.columns = ["name", "x", "y"] + cols
    result['umap'] = resultDf
    return result, args


def calculate_ttest(df, condition1, condition2):
    group1 = df[condition1]
    group2 = df[condition2]
    
    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values
    
    t, pvalue = stats.ttest_ind(group1, group2, nan_policy='omit')
    log = -math.log(pvalue, 10)
        
    return (df.name, t, pvalue, log)

def calculate_fold_change(df, condition1, condition2):
    group1 = df[condition1]
    group2 = df[condition2]

    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values

    if np.isnan(group1).all() or np.isnan(group2).all():
        fold_change = np.nan
    else:
        fold_change = np.nanmedian(group1) - np.nanmedian(group2)

    return fold_change

def ttest(data, condition1, condition2, alpha = 0.05, drop_cols=["sample", "name"]):
    df = data.copy()
    df = df.set_index('group')
    df = df.drop(drop_cols, axis = 1)
    df = df.loc[[condition1, condition2],:].T
    columns = ['identifier', 't-statistics', 'pvalue', '-Log pvalue']
    scores = df.apply(func = calculate_ttest, axis = 1, result_type='expand', args =(condition1, condition2))
    scores.columns = columns
    scores = scores.set_index("identifier")
    scores = scores.dropna(how="all")
    
    #Fold change
    scores["log2FC"] = df.apply(func = calculate_fold_change, axis = 1, args =(condition1, condition2))
    
    #Cohen's d
    scores["cohen_d"] = df.apply(func = cohen_d, axis = 1, args =(condition1, condition2, 1))
    
    #Hedge's g
    scores["hedges_g"] = df.apply(func = hedges_g, axis = 1, args =(condition1, condition2, 1))

    #FDR correction
    reject, padj = multi.fdrcorrection(scores["pvalue"], alpha=alpha, method = 'indep')
    scores['padj'] = padj
    scores['reject'] = reject
    scores = scores.reset_index()
    
    return scores

def oneway_anova(df, grouping):
    df_anova = data_imputed.copy()
    df_anova['group'] = df_anova.index.map(grouping.get)
    group_list = [str(x) for x in input().split()]
    df_anova = df_anova[df_anova['group'].isin(group_list)]
    df_anova = df_anova.reset_index().set_index(['group', 'Samples']).T
    df_anova = df_anova.unstack().reset_index().rename(columns={"level_2":"Protein ID"})
    df_anova.rename(columns={df_anova.columns[3]: 'value'}, inplace=True)
    columns = ['identifier', 't-statistics', 'pvalue', '-Log pvalue']
    scores = []
    for protein_entry in df_anova.groupby('Protein ID'):
        protein = protein_entry[0]
        samples = [group[1] for group in protein_entry[1].groupby('group')['value']]
        t_val, p_val = stats.f_oneway(*samples)
        log = -math.log(p_val, 10)
        scores.append((protein, t_val, p_val, log))
    scores = pd.DataFrame(scores)
    scores.columns = columns
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['pvalue'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue

    return scores

def cohen_d(df, condition1, condition2, ddof = 0):
    group1 = df[condition1]
    group2 = df[condition2]

    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values

    ng1 = group1.size
    ng2 = group2.size
    dof = ng1 + ng2 - 2
    if np.isnan(group1).all() or np.isnan(group2).all():
        d = np.nan
    else:
        meang1 = np.nanmean(group1) 
        meang2 = np.nanmean(group2)
        sdg1 = np.nanstd(group1, ddof = ddof)
        sdg2 = np.nanstd(group2, ddof = ddof)
        d = (meang1 - meang2) / np.sqrt(((ng1-1)* sdg1 ** 2 + (ng2-1)* sdg2 ** 2) / dof)
    
    return d

def hedges_g(df, condition1, condition2, ddof = 0):
    group1 = df[condition1]
    group2 = df[condition2]
    
    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values


    ng1 = group1.size
    ng2 = group2.size
    dof = ng1 + ng2 - 2
    if np.isnan(group1).all() or np.isnan(group2).all():
        g = np.nan
    else:
        meang1 = np.nanmean(group1) 
        meang2 = np.nanmean(group2)
        sdpooled = np.nanstd(np.concatenate([group1, group2]), ddof = ddof)

        #Correct bias small sample size
        if ng1+ng2 < 50:
            g = ((meang1 - meang2) / sdpooled) * ((ng1+ng2-3) / (ng1+ng2-2.25)) * np.sqrt((ng1+ng2-2) / (ng1+ng2))
        else:
            g = ((meang1 - meang2) / sdpooled)
        
    return g

def runVolcano(signature, cutoff = 1, alpha = 0.05):
    # Loop through signature
    color = []
    text = []
    for index, rowData in signature.iterrows():
        # Text
        text.append('<b>'+str(rowData['Leading razor protein'])+": "+str(index)+'<br>Gene = '+str(rowData['Gene names'])+'<br>log2FC = '+str(round(rowData['log2FC'], ndigits=2))+'<br>p = '+'{:.2e}'.format(rowData['pvalue'])+'<br>FDR = '+'{:.2e}'.format(rowData['padj']))
        
        # Color
        if rowData['padj'] < 0.05:
            if rowData['log2FC'] < -cutoff:
                color.append('#2b83ba')
            elif rowData['log2FC'] > cutoff:
                color.append('#d7191c')
            else:
                color.append('black')
        else:
            if rowData['log2FC'] < -cutoff:
                color.append("#abdda4")
            elif rowData['log2FC'] > cutoff:
                color.append('#fdae61')
            else:
                color.append('black')
                
    # Return 
    volcano_plot_results = {'x': signature['log2FC'], 'y': signature['-Log pvalue'], 'text':text, 'color': color}
    return volcano_plot_results
