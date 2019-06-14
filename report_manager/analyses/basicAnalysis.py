import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.utils import shuffle
from statsmodels.stats import multitest, anova as aov
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats
from scipy.special import factorial, betainc
import umap.umap_ as umap
from sklearn import preprocessing, ensemble, cluster
from scipy import stats
import pingouin as pg
import numpy as np
import networkx as nx
import community
import math
from fancyimpute import KNN
import kmapper as km
from report_manager import utils
from report_manager.analyses import wgcnaAnalysis as wgcna
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time
from joblib import Parallel, delayed
from numba import jit

def transform_into_wide_format(data, index, columns, values, extra=[]):
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
        df = df.reset_index()

    return df

def transform_into_long_format(data, drop_columns, group, columns=['mame','y']):
    data = data.drop(drop_columns, axis=1)
    
    long_data = pd.melt(data, id_vars=group, var_name=columns[0], value_name=columns[1])
    long_data = long_data.set_index(group)
    long_data.columns = columns
    
    return long_data

def get_ranking_with_markers(data, drop_columns, group, columns, list_markers, annotation={}):
    long_data = transform_into_long_format(data, drop_columns, group, columns)
    if len(set(long_data['name'].values.tolist()).intersection(list_markers)) > 0:
        long_data = long_data.drop_duplicates()
        long_data['symbol'] = [ 17 if p in list_markers else 0 for p in long_data['name'].tolist()]
        long_data['size'] = [25 if p in list_markers else 7 for p in long_data['name'].tolist()]
        long_data['name'] = [p+' marker in '+annotation[p] if p in annotation else p for p in long_data['name'].tolist()]
        
    return long_data


def extract_number_missing(df, missing_max, drop_cols=['sample'], group='group'):
    if group is None:
        groups = data.loc[:, data.notnull().sum(axis = 1) >= missing_max]
    else:
        groups = data.copy()
        groups = groups.drop(["sample"], axis = 1)
        groups = data.set_index("group").notnull().groupby(level=0).sum(axis = 1)
        groups = groups[groups>=missing_max]

    groups = groups.dropna(how='all', axis=1)
    return list(groups.columns)

def extract_percentage_missing(data, missing_max, drop_cols=['sample'], group='group'):
    if group is None:
        groups = data.loc[:, data.isnull().mean() <= missing_max].columns
    else:
        groups = data.copy()
        groups = groups.drop(drop_cols, axis = 1)
        groups = data.set_index(group)
        groups = groups.isnull().groupby(level=0).mean()
        groups = groups[groups<=missing_max]
        groups = groups.dropna(how='all', axis=1).columns

    return list(groups)

def imputation_KNN(data, drop_cols=['group', 'sample', 'subject'], group='group', cutoff=0.6, alone = True):
    df = data.copy()
    value_cols = [c for c in df.columns if c not in drop_cols]
    for g in df[group].unique():
        missDf = df.loc[df[group]==g, value_cols]
        missDf = missDf.loc[:, missDf.notnull().mean() >= cutoff]
        if missDf.isnull().values.any():
            X = np.array(missDf.values, dtype=np.float64)
            X_trans = KNN(k=3,verbose=False).fit_transform(X)
            missingdata_df = missDf.columns.tolist()
            dfm = pd.DataFrame(X_trans, index =list(missDf.index), columns = missingdata_df)
            df.update(dfm)
    if alone:
        df = df.dropna()

    return df

def imputation_mixed_norm_KNN(data, index_cols=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3, group='group', cutoff=0.6):
    df = imputation_KNN(data, drop_cols=index_cols, group=group, cutoff=cutoff, alone = False)
    df = imputation_normal_distribution(df, index=index_cols, shift=shift, nstd=nstd)

    return df

def imputation_normal_distribution(data, index=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3):
    print("IN")
    np.random.seed(112736)
    df = data.copy()
    if index is not None:
        df = df.set_index(index)
    data_imputed = df.T.sort_index()
    null_columns = data_imputed.columns[data_imputed.isnull().any()]
    for c in null_columns:
        missing = data_imputed[data_imputed[c].isnull()].index.tolist()
        std = data_imputed[c].std()
        mean = data_imputed[c].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        value = 0.0
        if not math.isnan(std) and not math.isnan(mean) and not math.isnan(sigma) and not math.isnan(mu):
            value = np.random.normal(mu, sigma, size=len(missing))
            value[value<0] = 0.0
        data_imputed.loc[missing, c] = value

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

def calculate_coefficient_variation(values):
    cv = scipy.stats.variation(values.apply(lambda x: np.power(2,x)).values) *100
    
    return cv

def get_coefficient_variation(data, drop_columns, group, columns):
    df = data.copy()
    formated_df = df.drop(drop_columns, axis=1)
    cvs = formated_df.groupby(group).apply(func=calculate_coefficient_variation)
    cols = formated_df.set_index(group).columns.tolist()
    cvs_df = pd.DataFrame()
    for i in cvs.index:
        gcvs = cvs[i].tolist()
        ints = formated_df.set_index('group').mean().values.tolist()
        tdf = pd.DataFrame(data= {'name':cols, 'x':gcvs, 'y':ints})
        tdf['group'] = i
        
        if cvs_df.empty:
            cvs_df = tdf.copy()
        else:
            cvs_df = cvs_df.append(tdf)

    return cvs_df


def get_proteomics_measurements_ready(df, index=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', imputation = True, method = 'distribution', missing_method = 'percentage', missing_max = 0.3, value_col='LFQ_intensity'):
    df = df.set_index(index)
    if extra_identifier is not None and extra_identifier in df.columns:
        df[identifier] = df[extra_identifier].map(str) + "-" + df[identifier].map(str)
    df = df.pivot_table(values=value_col, index=df.index, columns=identifier, aggfunc='first')
    df = df.reset_index()
    df[index] = df["index"].apply(pd.Series)
    df = df.drop(["index"], axis=1)
    aux = index
    if missing_method == 'at_least_x_per_group':
        aux.extend(extract_number_missing(df, missing_max, drop_cols, group=None))
    elif missing_method == 'percentage':
        aux.extend(extract_percentage_missing(df,  missing_max, drop_cols, group=None))

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
            
    df = df.reset_index()

    return df

def get_clinical_measurements_ready(df, subject_id='subject', sample_id='biological_sample', group_id='group', columns=['clinical_variable'], values='values', extra=['group'], imputation=True, imputation_method='KNN'):
    index = [subject_id, sample_id]
    drop_cols = [subject_id, sample_id]
    drop_cols.append(group_id)
    
    processed_df = transform_into_wide_format(df, index=index, columns=columns, values=values, extra=extra)
    if imputation:
        if imputation_method.lower() == "knn":
            df = imputation_KNN(processed_df, drop_cols=drop_cols, group=group_id)
        elif imputation_method.lower() == "distribution":
            df = imputation_normal_distribution(processed_df, index=index)
        elif imputation_method.lower() == 'group_median':
            df = imputation_median_by_group(processed_df)
        elif imputation_method.lower() == 'mixed':
            df = imputation_mixed_norm_KNN(processed_df,index_cols=index, group=group_id)
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
    X = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.transpose() * np.sqrt(pca.explained_variance_))
    loadings.index = df.columns
    loadings.columns = ['x', 'y']
    loadings['value'] = np.sqrt(np.power(loadings['x'],2) + np.power(loadings['y'],2))
    loadings = loadings.sort_values(by='value', ascending=False)
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

    result['pca'] = (resultDf, loadings)
    
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

def apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='fdr_i'):
    #rejected, padj = multitest.fdrcorrection(pvalues, alpha, method)
    rejected, padj = multitest.multipletests(pvalues, alpha, method)[:2]

    return (rejected, padj)

def apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh'):
    rejected, padj, num_hyp, alpha_stages = multitest.fdrcorrection_twostage(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=0.05, permutations=250):
    initial_seed = 176782
    i = permutations
    df_index = df.index.values
    columns = ['identifier']
    rand_pvalues = None
    while i>0:
        df_index = shuffle(df_index, random_state=int(initial_seed + i))
        df_random = df.reset_index(drop=True)
        df_random.index = df_index
        df_random.index.name = 'group'
        columns = ['identifier', 'F-statistics', 'pvalue_'+str(i)]
        if list(df_random.index) != list(df.index):
            rand_scores = df_random.apply(func=calculate_anova, axis=0, result_type='expand').T
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
    if a!=0 and b != 0:
        qvalue = (a/b * 1/n)
    return (qvalue, qvalue <= alpha)

def convertToEdgeList(data, cols):
    data.index.name = None
    edge_list = data.stack().reset_index()
    edge_list.columns = cols

    return edge_list

@jit(nopython=False, parallel=True)
def run_correlation(df, alpha=0.05, subject='subject', group='group', method='pearson', correction=('fdr', 'fdr_i')):
    calculated = set()
    correlation = pd.DataFrame()
    if check_is_paired(df, subject, group):
        if len(df[subject].unique()) > 2:
            correlation = run_rm_correlation(df, alpha=alpha, subject=subject, correction=correction)
    else:
        df = df.dropna()._get_numeric_data()
        if not df.empty:
            r, p = run_efficient_correlation(df, method=method)
            print("Hola")
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

def calculate_rm_correlation(df, x, y, subject):
    r, dof, pvalue, ci, power = pg.rm_corr(data=df, x=x, y=y, subject=subject)
    
    return (x, y, r, pvalue, dof, ci, power)

@jit(nopython=False, parallel=True)
def run_rm_correlation(df, alpha=0.05, subject='subject', correction=('fdr', 'fdr_i')):
    print("STARTING CORRELATION")
    calculated = set()
    rows = []
    #df = df.dropna()._get_numeric_data()
    if not df.empty:
        df = df.set_index(subject)._get_numeric_data()
        start = time.time()
        combinations = itertools.combinations(df.columns, 2)
        df = df.reset_index()
        for x, y in combinations:
            row = calculate_rm_correlation(df, x, y, subject)
            rows.append(row)
        end = time.time()
        correlation = pd.DataFrame(rows, columns=["node1", "node2", "weight", "pvalue", "dof", "CI95%", "power"])
        
        if correction[0] == 'fdr':
            rejected, padj = apply_pvalue_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])
        elif correction[0] == '2fdr':
            rejected, padj = apply_pvalue_twostage_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])

        correlation["padj"] = padj
        correlation["rejected"] = rejected
        correlation = correlation[correlation.rejected]
    print("DONE")
    return correlation

@jit(nopython=False, parallel=True)
def run_efficient_correlation(data, method='pearson'):
    matrix = data.values
    if method == 'pearson':
        r = np.corrcoef(matrix, rowvar=False)
    elif method == 'spearman':
        r, p = stats.spearmanr(matrix, axis=0)

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
    group1 = df[[condition1]].values
    group2 = df[[condition2]].values
    
    mean1 = group1.mean() 
    mean2 = group2.mean()
    log2fc = mean1 - mean2
    t, pvalue = stats.ttest_rel(group1, group2, nan_policy='omit')

    return (t, pvalue, mean1, mean2, log2fc)

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
    
    mean1 = group1.mean() 
    mean2 = group2.mean()
    log2fc = mean1 - mean2
    t, pvalue = stats.ttest_ind(group1, group2, nan_policy='omit')

    return (t, pvalue, mean1, mean2, log2fc)

def calculate_THSD(df):
    col = df.name
    result = pairwise_tukeyhsd(df.values, list(df.index))
    df_results = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
    df_results.columns = ['group1', 'group2', 'log2FC', 'lower', 'upper', 'rejected']
    df_results['identifier'] = col
    df_results = df_results.set_index('identifier')
    df_results['FC'] = df_results['log2FC'].apply(lambda x: np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)))
    df_results['rejected'] = df_results['rejected']

    return df_results

def calculate_pairwise_ttest(df, column, subject='subject', group='group', correction='none'):
    posthoc_columns = ['Contrast', 'group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired', 'Parametric', 'T', 'dof', 'tail', 'padj', 'BF10', 'efsize', 'eftype']
    if correction == "none":
        valid_cols = ['group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired','Parametric', 'T', 'dof', 'BF10', 'efsize', 'eftype']
    else:
        valid_cols = posthoc_columns
    posthoc = pg.pairwise_ttests(data=df, dv=column, between=group, subject=subject, effsize='hedges', return_desc=True, padjust=correction)
    posthoc.columns =  posthoc_columns
    posthoc = posthoc[valid_cols]
    posthoc = complement_posthoc(posthoc, column)
    posthoc = posthoc.set_index('identifier')

    return posthoc

def complement_posthoc(posthoc, identifier):
    posthoc['identifier'] = identifier
    posthoc['log2FC'] = posthoc['mean(group1)'] -posthoc['mean(group2)']
    posthoc['FC'] = posthoc['log2FC'].apply(lambda x: np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)))

    return posthoc

def calculate_anova(df, group='group'):
    col = df.name
    group_values = df.groupby(group).apply(np.array).values
    t, pvalue = stats.f_oneway(*group_values)
    return (col, t, pvalue)

def calculate_repeated_measures_anova(df, column, subject='subject', group='group', alpha=0.05):
    aov_result = pg.rm_anova(data=df, dv=column, within=group,subject=subject, detailed=True, correction=True)
    aov_result.columns = ['Source', 'SS', 'DF', 'MS', 'F', 'pvalue', 'padj', 'np2', 'eps', 'sphericity', 'Mauchlys sphericity', 'p-spher']
    t, pvalue, padj = aov_result.loc[0, ['F', 'pvalue', 'padj']].values 

    return (column, t, pvalue, padj)

def get_max_permutations(df, group='group'):
    num_groups = len(list(df.index))
    num_per_group = df.groupby(group).size().tolist()
    max_perm = factorial(num_groups)/np.prod(factorial(np.array(num_per_group)))

    return max_perm

def check_is_paired(df, subject, group):
    is_pair = False
    if subject is not None:
        count_subject_groups = df.groupby(subject)[group].count()
        is_pair = (count_subject_groups > 1).all()

    return is_pair

def run_anova(df, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=250):
    columns = ['identifier', 'F-statistics', 'pvalue']
    if subject is not None and check_is_paired(df, subject, group):
        groups = df[group].unique()
        drop_cols = [d for d in drop_cols if d != subject]
        if len(groups) == 2:
            res = run_ttest(df, groups[0], groups[1], alpha = alpha, drop_cols=drop_cols, subject=subject, group=group, paired=True, correction='fdr_bh')
        else:
            
            res = run_repeated_measurements_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, group=group, permutations=0)
    else:
        df = df.set_index([group])
        df = df.drop(drop_cols, axis=1)
        scores = df.apply(func = calculate_anova, axis=0, result_type='expand', group=group).T
        scores.columns = columns
        scores = scores.set_index("identifier")
        scores = scores.dropna(how="all")

        max_perm = get_max_permutations(df, group=group)
        #FDR correction
        if permutations > 0 and max_perm>=10:
            if max_perm < permutations:
                permutations = max_perm
            observed_pvalues = scores.pvalue
            count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
            scores= scores.join(count)
            scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
        else:
            rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'fdr_i')
            scores['correction'] = 'FDR correction BH'
            scores['padj'] = padj
            scores['rejected'] = rejected
        res = None
        for col in df.columns:
            pairwise = calculate_THSD(df[col])
            if res is None:
                res = pairwise
            else:
                res = pd.concat([res,pairwise], axis=0)
        if res is not None:
            res = res.join(scores[['F-statistics', 'pvalue', 'padj']].astype('float'))
            res['correction'] = scores['correction']
        else:
            res = scores
            res["log2FC"] = np.nan

        res = res.reset_index()
        res['rejected'] = res['padj'] < alpha
        res['-log10 pvalue'] = res['padj'].apply(lambda x: -np.log10(x))
    
    return res

def run_repeated_measurements_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', group='group', permutations=150):
    df = df.set_index([subject,group])
    df = df.drop(drop_cols, axis=1)
    aov_result = []
    for col in df.columns:
        aov = calculate_repeated_measures_anova(df.reset_index(), column=col, subject=subject, group=group, alpha=alpha)
        aov_result.append(aov)

    scores = pd.DataFrame(aov_result, columns = ['identifier','F-statistics', 'pvalue', 'padj'])
    scores = scores.set_index('identifier')

    scores['correction'] = 'Greenhouse-Geisser correction'
    
    res = None
    for col in df.columns:
        pairwise = calculate_pairwise_ttest(df[col].reset_index(), column=col, subject=subject, group=group)
        if res is None:
            res = pairwise
        else:
            res = pd.concat([res,pairwise], axis=0)
    if res is not None:
        res = res.join(scores[['F-statistics', 'pvalue', 'padj']].astype('float'))
        res['correction'] = 'Greenhouse-Geisser correction' 
    else:
        res = scores
        res["log2FC"] = np.nan

    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha
    res['-log10 pvalue'] = res['padj'].apply(lambda x: - np.log10(x))
    
    return res

def run_ttest(df, condition1, condition2, alpha = 0.05, drop_cols=["sample"], subject='subject', group='group', paired=False, correction='fdr_i', permutations=150):
    columns = ['T-statistics', 'pvalue', 'mean_group1', 'mean_group2', 'log2FC']
    df = df.set_index([group, subject])
    df = df.drop(drop_cols, axis = 1)
    cols = df.columns
    if paired:
        scores = df.T.apply(func = calculate_paired_ttest, axis=1, result_type='expand', args =(condition1, condition2))
    else:
        scores = df.T.apply(func = calculate_ttest, axis=1, result_type='expand', args =(condition1, condition2))
    scores.columns = columns
    scores = scores.dropna(how="all")

    max_perm = get_max_permutations(df, group=group)
    #FDR correction
    if permutations > 0:
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'fdr_i')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
    #scores['rejected'] = scores['padj'] <= alpha
    scores['group1'] = condition1
    scores['group2'] = condition2
    scores['FC'] = [np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)) for x in scores['log2FC'].values]
    scores['-log10 pvalue'] = [- np.log10(x) for x in scores['padj'].values]
    scores = scores.reset_index() 
    print(scores.shape)

    return scores

def run_fisher(group1, group2, alternative='two-sided'):
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

def run_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher'):
    foreground_list = regulation_data[regulation_data[reject_col]][identifier].unique().tolist()
    background_list = regulation_data[~regulation_data[reject_col]][identifier].unique().tolist()
    grouping = []
    for i in annotation[identifier]:
        if i in foreground_list:
            grouping.append('foreground')
        elif i in background_list:
            grouping.append('background')
        else:
            grouping.append(np.nan)
    annotation['group'] = grouping
    annotation = annotation.dropna(subset=['group'])

    result = run_enrichment(annotation, foreground='foreground', background='background', foreground_pop=len(foreground_list), background_pop=len(background_list), annotation_col=annotation_col, group_col=group_col, identifier_col=identifier, method=method)
    
    
    return result

def run_enrichment(data, foreground, background, foreground_pop, background_pop, annotation_col='annotation', group_col='group', identifier_col='identifier', method='fisher'):
    result = pd.DataFrame()
    df = data.copy()
    terms = []
    ids = []
    pvalues = []
    fnum = []
    bnum = []
    countsdf = df.groupby([annotation_col,group_col]).agg(['count'])[(identifier_col,'count')].reset_index()
    countsdf.columns = [annotation_col, group_col, 'count']
    for annotation in countsdf[countsdf[group_col] == foreground][annotation_col].unique().tolist():
        counts = countsdf[countsdf[annotation_col] == annotation]
        num_foreground = counts.loc[counts[group_col] == foreground,'count'].values
        num_background = counts.loc[counts[group_col] == background,'count'].values
        
        if len(num_foreground) == 1:
            num_foreground = num_foreground[0]
        if len(num_background) == 1:
            num_background = num_background[0]
        else:
            num_background=0
        if method == 'fisher':
            odds, pvalue = run_fisher([num_foreground, foreground_pop-num_foreground],[num_background, background_pop-num_background])
        fnum.append(num_foreground)
        bnum.append(num_background)
        terms.append(annotation)
        pvalues.append(pvalue)
        ids.append(",".join(df.loc[(df[annotation_col]==annotation) & (df[group_col] == foreground), identifier_col].tolist()))
    if len(pvalues) > 1:
        rejected, padj = apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='fdr_i')
        result = pd.DataFrame({'terms':terms, 'identifiers':ids, 'foreground':fnum, 'background':bnum, 'pvalue':pvalues, 'padj':padj, 'rejected':rejected})
        #result = result[result.rejected]
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

def run_mapper(data, lenses=["l2norm"], n_cubes = 15, overlap=0.5, n_clusters=3, linkage="complete", affinity="correlation"):
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

def run_WGCNA(data, drop_cols_exp, drop_cols_cli, RsquaredCut=0.8, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0):
    """ 
    Runs an automated weighted gene co-expression network analysis (WGCNA), using input proteomics/transcriptomics/genomics and clinical variables data.

    Args:
        data: 
        drop_cols_exp: list of strings for columns in experimental dataset to be removed.
        drop_cols_cli: list of strings for columns in clinical dataset to be removed.
        RsquaredCut: desired minimum scale free topology fitting index R^2.
        networkType: network type ('unsigned', 'signed', 'signed hybrid', 'distance').
        minModuleSize: minimum module size.
        deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
        pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.
        merge_modules: if True, very similar modules are merged.
        MEDissThres: maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging.
        verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    
    Returns:
        Tuple with multiple pandas dataframes.
    """
    result = {}
    dfs = wgcna.get_data(data, drop_cols_exp=drop_cols_exp, drop_cols_cli=drop_cols_cli)
    if 'clinical' in dfs:
        data_cli = dfs['clinical']   #Extract clinical data
        data_exp, = [i for i in dfs.keys() if i != 'clinical']   #Get dictionary key for experimental data
        data_exp = dfs[data_exp]   #Extract experimental data

        softPower = wgcna.pick_softThreshold(data_exp, RsquaredCut=RsquaredCut, networkType=networkType, verbose=verbose)
        dissTOM, moduleColors = wgcna.build_network(data_exp, softPower=softPower, networkType=networkType, minModuleSize=minModuleSize, deepSplit=deepSplit,
                                              pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules, MEDissThres=MEDissThres, verbose=verbose)

        Features_per_Module = wgcna.get_FeaturesPerModule(data_exp, moduleColors, mode='dataframe')
        MEs = wgcna.calculate_module_eigengenes(data_exp, moduleColors, softPower=softPower, dissimilarity=False)
        moduleTraitCor, textMatrix = wgcna.calculate_ModuleTrait_correlation(data_exp, data_cli, MEs)
        MM, MMPvalue = wgcna.calculate_ModuleMembership(data_exp, MEs)
        FS, FSPvalue = wgcna.calculate_FeatureTraitSignificance(data_exp, data_cli)
        METDiss, METcor = wgcna.get_EigengenesTrait_correlation(MEs, data_cli)

        result['dissTOM'] = dissTOM
        result['module_colors'] = moduleColors
        result['features_per_module'] = Features_per_Module
        result['MEs'] = MEs
        result['module_trait_cor'] = moduleTraitCor
        result['text_matrix'] = textMatrix
        result['module_membership'] = MM
        result['module_membership_pval'] = MMPvalue
        result['feature_significance'] = FS
        result['feature_significance_pval'] = FSPvalue
        result['ME_trait_diss'] = METDiss
        result['ME_trait_cor'] = METcor

        return result

    else:
        return None

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
    elif args['communities_algorithm'] == 'affinity_propagation':
        adjacency = nx.to_pandas_adjacency(graph, weight='width')
        nodes = list(adjacency.columns)
        communities = AffinityPropagation().fit(adjacency.values).labels_
        communities = {nodes[i]:communities[i] for i in range(len(communities))}


    return communities

def get_publications_abstracts(data, publication_col="publication", join_by=['publication','Proteins','Diseases'], index="PMID"):
    abstracts = pd.DataFrame()
    if not data.empty:
        abstracts = utils.getMedlineAbstracts(list(data.reset_index()[publication_col].unique()))
        abstracts = abstracts.set_index(index)
        abstracts = abstracts.join(data.reset_index()[join_by].set_index(publication_col)).reset_index()

    return abstracts 
