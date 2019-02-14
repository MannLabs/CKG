import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.stats import multitest, anova as aov
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.special import factorial, betainc
import umap.umap_ as umap
from sklearn import preprocessing, ensemble, cluster
from scipy import stats
import numpy as np
import networkx as nx
import community
import math
from random import shuffle
from fancyimpute import KNN
import kmapper as km
from report_manager import utils

def transform_into_long_format(data, index, columns, values, extra=[], use_index=False):
    df = data.copy()
    if not df.empty:
        cols = [columns, values]
        cols.extend(index)
        if len(extra) > 0:
            extra.extend(index)
            extra_cols = df[extra].set_index(index)
        df = df[cols]
        df = df.pivot_table(index=index, columns=columns, values=values)
        df = df.join(extra_cols)
        df = df.drop_duplicates()
        if not use_index:
            df = df.reset_index(drop=True)

    return df

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
        groups = data.loc[:, data.isnull().mean() <= missing_max].columns
    else:
        groups = data.copy()
        groups = groups.drop(["sample"], axis = 1)
        groups = data.set_index("group")
        groups = groups.isnull().groupby(level=0).mean()
        groups = groups[groups<=missing_max]
        groups = groups.dropna(how='all', axis=1).columns
      
    return list(groups)

def imputation_KNN(data, drop_cols=['group', 'sample', 'subject'], group='group', cutoff=0.5, alone = True):
    df = data.copy()
    value_cols = [c for c in df.columns if c not in drop_cols]
    for g in df[group].unique():
        missDf = df.loc[df[group]==g, value_cols]
        missDf = missDf.loc[:, missDf.notnull().mean() >= cutoff]
        if missDf.isnull().values.any():
            X = np.array(missDf.values, dtype=np.float64)
            X_trans = KNN(k=3).fit_transform(X)
            missingdata_df = missDf.columns.tolist()
            dfm = pd.DataFrame(X_trans, index =list(missDf.index), columns = missingdata_df)
            df.update(dfm)
    if alone:
        df = df.dropna()
    
    return df

def imputation_mixed_norm_KNN(data):
    df = imputation_KNN(data, cutoff=0.6, alone = False)
    df = imputation_normal_distribution(df, shift = 1.8, nstd = 0.3)
    
    return df

def imputation_normal_distribution(data, index=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3):
    np.random.seed(112736)
    df = data.copy()
    if index is not None:
        df = df.set_index(index)
    data_imputed = df.T
    for i in data_imputed.loc[:, data_imputed.isnull().any()]:
        missing = data_imputed[i].isnull()
        std = data_imputed[i].std()
        mean = data_imputed[i].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        value = 0.0
        if not math.isnan(std) and not math.isnan(mean) and not math.isnan(sigma) and not math.isnan(mu):
            value = np.random.normal(mu, sigma, size=len(data_imputed[missing]))
            value[value<0] = 0.0
        data_imputed.loc[missing, i] = value

    return data_imputed.T


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

def get_proteomics_measurements_ready(data, index=['group', 'sample', 'subject'], imputation = True, method = 'distribution', missing_method = 'percentage', missing_max = 0.3, value_col='LFQ_intensity'):
    df = data.copy()
    conditions = df['group'].unique()
    df = df.set_index(index)
    df['identifier'] = df['name'].map(str) + "-" + df['identifier'].map(str)
    df = df.pivot_table(values=value_col, index=df.index, columns='identifier', aggfunc='first')
    df = df.reset_index()
    df[index] = df["index"].apply(pd.Series)
    df = df.drop(["index"], axis=1)
    aux = index
    
    if missing_method == 'at_least_x_per_group':
        aux.extend(extract_number_missing(df, conditions, missing_max))
    elif missing_method == 'percentage':
        aux.extend(extract_percentage_missing(df, conditions, missing_max))  
    
    df = df[list(set(aux))]
    if imputation:
        if method == "KNN":
            df = imputation_KNN(df)
        elif method == "distribution":
            df = imputation_normal_distribution(df, shift = 1.8, nstd = 0.3)
        elif method == 'group_median':
            df = imputation_median_by_group(df)
        elif method == 'mixed':
            df = imputation_mixed_norm_KNN(df)
        else:
            sys.exit()
    
    df = df.reset_index()
    return df

def run_pca(data, drop_cols=['sample', 'subject'], group='group', components=2):
    np.random.seed(112736)
    result = {}
    df = data.copy()
    df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    X = df._get_numeric_data()
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
        if components>3:
            cols = resultDf.columns[4:]
        resultDf.columns = ["name", "x", "y", "z"] + cols
    result['pca'] = resultDf
    return result, args

def run_tsne(data, drop_cols=['sample', 'subject'], group='group', components=2, perplexity=40, n_iter=1000, init='pca'):
    result = {}
    df = data.copy()
    df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    X = df._get_numeric_data()
    y = df.index
    
    tsne = TSNE(n_components=components, verbose=0, perplexity=perplexity, n_iter=n_iter, init=init)
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
    
def run_umap(data, drop_cols=['sample', 'subject'], group='group', n_neighbors=10, min_dist=0.3, metric='cosine'):
    result = {}
    df = data.copy()
    df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    X = df._get_numeric_data()
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

def calculate_correlations(x, y, method='pearson'):
    if method == "pearson":
        coefficient, pvalue = stats.pearsonr(x, y)
    elif method == "spearman":
        coefficient, pvalue = stats.spearmanr(x, y)

    return (coefficient, pvalue)

def apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep'):
    rejected, padj = multitest.fdrcorrection(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh'):
    rejected, padj, num_hyp, alpha_stages = multitest.fdrcorrection_twostage(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=0.05, permutations=250):
    i = permutations
    df_index = list(df.index)
    columns = ['identifier']
    rand_pvalues = None
    while i>0:
        shuffle(df_index)
        df_random = df.reset_index(drop=True)
        df_random.index = df_index
        df_random.index.name = 'group'
        columns = ['identifier', 't-statistics', 'pvalue_'+str(i), '-log10 pvalue']
        if list(df_random.index) != list(df.index):
            rand_scores = df_random.apply(func=calculate_annova, axis=0, result_type='expand').T
            rand_scores.columns = columns
            rand_scores = rand_scores.set_index("identifier")
            rand_scores = rand_scores.dropna(how="all")
            rand_scores = rand_scores['pvalue_'+str(i)]
            if rand_pvalues is None:
                rand_pvalues = rand_scores.to_frame()
            else:
                rand_pvalues = rand_pvalues.join(rand_scores)
            i -= 1
    count = observed_pvalues.to_frame().apply(func=get_counts_permutation_fdr, result_type='expand', axis=1, args=(rand_pvalues, observed_pvalues, permutations, alpha))
    count.columns = ['padj', 'rejected']
    #count = count.set_index('identifier')
    
    return count

def get_counts_permutation_fdr(value, random, observed, n, alpha):
    a = (random <= value.values[0]).sum().sum()
    b = (observed <= value.values[0]).sum()
    qvalue = 1
    if b != 0:
        qvalue = (a/b * 1/n)
    return (qvalue, qvalue <= alpha)

def convertToEdgeList(data, cols):
    data.index.name = None
    edge_list = data.stack().reset_index()
    edge_list.columns = cols

    return edge_list

def runCorrelation(data, alpha=0.05, method='pearson', correction=('fdr', 'indep')):
    calculated = set()
    df = data.copy()
    df = df.dropna()._get_numeric_data()
    
    r, p = runEfficientCorrelation(df, method=method)
    rdf = pd.DataFrame(r, index=df.columns, columns=df.columns)
    pdf = pd.DataFrame(p, index=df.columns, columns=df.columns)
    rdf.values[[np.arange(len(rdf))]*2] = np.nan
    pdf.values[[np.arange(len(pdf))]*2] = np.nan
    correlation = convertToEdgeList(rdf, ["node1", "node2", "weight"])
    pvalues = convertToEdgeList(pdf, ["node1", "node2", "pvalue"])
     
    if correction[0] == 'fdr':
        rejected, padj = apply_pvalue_fdrcorrection(pvalues["pvalue"].tolist(), alpha=alpha, method=correction[1])
    elif correction[0] == '2fdr':
        rejected, padj = apply_pvalue_twostage_fdrcorrection(pvalues["pvalue"].tolist(), alpha=alpha, method=correction[1])
    
    correlation["padj"] = padj
    correlation["rejected"] = rejected
    correlation = correlation[correlation.rejected]
    
    return correlation

def runEfficientCorrelation(data, method='pearson'):
    matrix = data.values
    if method == 'pearson':
        r = np.corrcoef(matrix, rowvar=False)
    elif method == 'spearman':
        r, p = spearmanr(matrix, axis=0)
    
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = pf
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

    
    return r, p
    
def calculate_paired_ttest(df, condition1, condition2):
    group1 = df[condition1]
    group2 = df[condition2]
    print(group1)
    print(group2)
    if isinstance(group1, np.float):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float):
        group2 = np.array(group2)
    else:
        group2 = group2.values
    
    t, pvalue = stats.ttest_rel(group1, group2, nan_policy='omit')
    log = -math.log(pvalue, 10)
        
    return (df.name, t, pvalue, log)

def calculate_ttest(df, condition1, condition2):
    group1 = df[condition1]
    group2 = df[condition2]
    if isinstance(group1, np.float):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float):
        group2 = np.array(group2)
    else:
        group2 = group2.values
  
    t, pvalue = stats.ttest_ind(group1, group2, nan_policy='omit')
    log = -math.log(pvalue, 10)
        
    return (df.name, t, pvalue, log)

def calculate_THSD(df):
    col = df.name
    result = pairwise_tukeyhsd(df, list(df.index))
    df_results = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
    df_results.columns = ['group1', 'group2', 'log2FC', 'lower', 'upper', 'rejected'] 
    df_results['identifier'] = col
    df_results = df_results.set_index('identifier')
    df_results['FC'] = df_results['log2FC'].apply(lambda x: np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)))
    
    return df_results

def calculate_annova(df, group='group'):
    col = df.name
    group_values = df.groupby(group).apply(list).tolist()
    t, pvalue = stats.f_oneway(*group_values)
    log = -math.log(pvalue,10)

    return (col, t, pvalue, log)

def calculate_repeated_measures_annova(df, sample='sample', group='group'):
    col = df.name
    #group_values = df.groupby('group').apply(list).tolist()
    num_df, den_df, f_stat, pvalue = aov.AnovaRM(df.reset_index(), col, sample, within=[group], aggregate_func='mean').fit().anova_table.values.tolist()[0]
    log = -math.log(pvalue,10)

    return (col, f_stat, pvalue, log)

def get_max_permutations(df, group='group'):
    num_groups = len(list(df.index))
    num_per_group = df.groupby(group).size().tolist() 
    max_perm = factorial(num_groups)/np.prod(factorial(np.array(num_per_group)))
    
    return max_perm

def anova(data, alpha=0.05, drop_cols=["sample", 'subject'], group='group', permutations=50):
    columns = ['identifier', 't-statistics', 'pvalue', '-log10 pvalue']
    df = data.copy()
    df = df.set_index(group)
    df = df.drop(drop_cols, axis=1)
    scores = df.apply(func = calculate_annova, axis=0,result_type='expand', group=group).T
    scores.columns = columns
    scores = scores.set_index("identifier")
    scores = scores.dropna(how="all")
    
    
    max_perm = get_max_permutations(df)
    #FDR correction
    if permutations > 0 and max_perm>=10:
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
        aux = scores.loc[scores.rejected,:]
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
    
    #sigdf = df[list(scores[scores.rejected].index)]
    res = None
    for col in df.columns:
        pairthsd = calculate_THSD(df[col])
        if res is None:
            res = pairthsd
        else:
            res = pd.concat([res,pairthsd], axis=0)
    if res is not None:
        res = res.join(scores[['t-statistics', 'pvalue', '-log10 pvalue', 'padj']].astype('float'))
        res['correction'] = scores['correction']
    else:
        res = scores
        res["log2FC"] = np.nan
    
    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha
    
    return res

def repeated_measurements_anova(data, alpha=0.5, drop_cols=['sample'], subject='subject', sample='sample', group='group', permutations=50):
    columns = ['identifier', 'F-statistics', 'pvalue', '-log10 pvalue']
    df = data.copy()
    group_df = df.groupby(group)
    list_items = []
    for n, g in group_df:
        list_items.append(g[sample].tolist())
    max_intersection = []
    for l1,l2 in itertools.combinations(list_items,2):
            print(len(l1),len(set(l1)))
            print(len(l2),len(set(l2)))
            print(l2)
            intersect = set(l1).intersection(l2)
            if len(intersect) > len(max_intersection):
                max_intersection = intersect
    
    df = df[df[sample].isin(max_intersection)]
    df = df.set_index([sample,group])
    df = df.drop(drop_cols, axis=1)
    scores = df.apply(func = calculate_repeated_measures_annova, axis=0,result_type='expand', sample=sample, group=group).T
    scores.columns = columns
    scores = scores.set_index("identifier")
    scores = scores.dropna(how="all")
    df = df.drop([sample], axis=1)

    #FDR correction
    if permutations > 0:
        #max_perm = get_max_permutations(df)
        #if max_perm < permutations:
        #    permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        aux = scores.loc[scores.rejected,:]
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['padj'] = padj
        scores['rejected'] = rejected
    
    #sigdf = df[list(scores[scores.rejected].index)]
    res = None
    for col in df.columns:
        pairthsd = calculate_THSD(df[col])
        if res is None:
            res = pairthsd
        else:
            res = pd.concat([res,pairthsd], axis=0)
    if res is not None:
        res = res.join(scores[['t-statistics', 'pvalue', '-log10 pvalue', 'padj']].astype('float'))
    else:
        res = scores
        res["log2FC"] = np.nan
    
    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha
    
    return res

def ttest(data, condition1, condition2, alpha = 0.05, drop_cols=["sample", 'subject'], group='group', paired=False, permutations=50):
    df = data.copy()
    df = df.set_index(group)
    df = df.drop(drop_cols, axis = 1)
    tdf = df.loc[[condition1, condition2],:].T
    columns = ['identifier', 't-statistics', 'pvalue', '-log10 pvalue']
    if paired:
        scores = tdf.apply(func = calculate_paired_ttest, axis = 1, result_type='expand', args =(condition1, condition2))
    else:
        scores = tdf.apply(func = calculate_ttest, axis = 1, result_type='expand', args =(condition1, condition2))
    scores.columns = columns
    scores = scores.set_index("identifier")
    scores = scores.dropna(how="all")
    
    #Fold change
    scores["log2FC"] = tdf.apply(func = calculate_fold_change, axis = 1, args =(condition1, condition2))
    #Cohen's d
    scores["cohen_d"] = tdf.apply(func = cohen_d, axis = 1, args =(condition1, condition2, 1))
    
    #Hedge's g
    scores["hedges_g"] = tdf.apply(func = hedges_g, axis = 1, args =(condition1, condition2, 1))
    #FDR correction
    if permutations > 0:
        max_perm = get_max_permutations(df)
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['padj'] = padj
        scores['rejected'] = rejected
    scores = scores.reset_index()
    aux = scores.loc[scores.rejected,:]
    return scores

def runFisher(group1, group2, alternative='two-sided'):
    '''         annotated   not-annotated
        group1      a               b   
        group2      c               d
        ------------------------------------

        group1 = [a, b]
        group2 = [c, d]

        odds, pvalue = stats.fisher_exact([[a, b], [c, d]])
    '''
    odds, pvalue = stats.fisher_exact([group1, group2], alternative)

    return (odds, pvalue)

def runEnrichment(data, foreground, background, method='fisher'):
    result = pd.DataFrame()
    df = data.copy()
    grouping = df['group'].value_counts().to_dict()
    terms = []
    ids = []
    pvalues = []
    if foreground in grouping and background in grouping:
        foreground_pop = grouping[foreground]
        background_pop = grouping[background]
        countsdf = df.groupby(['annotation','group']).agg(['count'])[('identifier','count')].reset_index().set_index('annotation')
        countsdf.columns = ['group', 'count']
        for annotation in countsdf.index:
            counts = countsdf[annotation]
            num_foreground = counts.loc[counts['group'] == foreground,'count'].values
            num_background = counts.loc[counts['group'] == background,'count'].values
            if len(num_foreground) == 1:
                num_foreground = num_foreground[0]
            else:
                num_foreground = 0
            if len(num_background) == 1:
                num_background = num_background[0]
            else:
                num_background = 0
            if method == 'fisher':
                odds, pvalue = runFisher([num_foreground, foreground_pop-num_foreground],[num_background, background_pop-num_background])
            terms.append(annotation)
            pvalues.append(pvalue)
            ids.append(df.loc[(df['annotation']==annotation) & (df['group'] == foregorund), "identifier"].tolist())

    if len(pvalues) > 1:
        rejected,padj = apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep')
        result = pd.DataFrame({'terms':terms, 'identifiers':ids, 'padj':padj, 'rejected':rejected})
        result = result[result.rejected]
    return result
    
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

def runMapper(data, lenses=["l2norm"], n_cubes = 15, overlap=0.5, n_clusters=3, linkage="complete", affinity="correlation"):
    X = data._get_numeric_data()
    labels ={i:data.index[i] for i in range(len(data.index))} 

    model = ensemble.IsolationForest(random_state=1729)
    model.fit(X)
    lens1 = model.decision_function(X).reshape((X.shape[0], 1))
    
    # Create another 1-D lens with L2-norm
    mapper = km.KeplerMapper(verbose=0)
    lens2 = mapper.fit_transform(X, projection=lenses[0])
    
    # Combine both lenses to get a 2-D [Isolation Forest, L^2-Norm] lens
    lens = np.c_[lens1, lens2]
    
    # Define the simplicial complex
    simplicial_complex = mapper.map(lens,
                      X,
                      nr_cubes=n_cubes,
                      overlap_perc=overlap,
                      clusterer=cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity))
    


    return simplicial_complex, {"labels":labels}

def most_central_edge(G):
    centrality = nx.eigenvector_centrality_numpy(G, weight='width')
    
    return max(centrality, key=centrality.get)

def get_louvain_partitions(G, weight):
    partition = community.best_partition(G)
    
    return partition

def get_network_communities(graph, args):
    if 'communities_algorithm' not in args:
        args['communities_algorithm'] = 'louvain'
    
    if args['communities_algorithm'] == 'louvain':
        communities = get_louvain_partitions(graph, args['values'])
    elif args['communities_algorithm'] == 'greedy_modularity':
        gcommunities = nx.algorithms.community.greedy_modularity_communities(graph, weight=args['values'])
        communities = utils.generator_to_dict(gcommunities)
    elif args['communities_algorithm'] == 'asyn_label_propagation':
        gcommunities = nx.algorithms.community.label_propagation.asyn_lpa_communities(graph, args['values'])
        communities = utils.generator_to_dict(gcommunities)
    elif args['communities_algorithm'] == 'girvan_newman':
        gcommunities = nx.algorithms.community.girvan_newman(graph, most_valuable_edge=most_central_edge)
        communities = utils.generator_to_dict(gcommunities)
    
    return communities

