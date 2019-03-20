import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from statsmodels.stats import multitest, anova as aov
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.special import factorial, betainc
import umap.umap_ as umap
from sklearn import preprocessing, ensemble, cluster
from scipy import stats
import pingouin as pg
import numpy as np
import networkx as nx
import community
import math
from random import shuffle
from fancyimpute import KNN
import kmapper as km
from report_manager import utils
from report_manager.analyses import wgcnaAnalysis as wgcna
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time
from joblib import Parallel, delayed
import numba

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
            X_trans = KNN(k=3,verbose=False).fit_transform(X)
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

def apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep'):
    rejected, padj = multitest.fdrcorrection(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh'):
    rejected, padj, num_hyp, alpha_stages = multitest.fdrcorrection_twostage(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=0.05, permutations=250):
    np.random.seed(176782)
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
    if b != 0:
        qvalue = (a/b * 1/n)
    return (qvalue, qvalue <= alpha)

def convertToEdgeList(data, cols):
    data.index.name = None
    edge_list = data.stack().reset_index()
    edge_list.columns = cols

    return edge_list

def runCorrelation(df, alpha=0.05, method='pearson', correction=('fdr', 'indep')):
    calculated = set()
    correlation = pd.DataFrame()
    df = df.dropna()._get_numeric_data()
    if not df.empty:
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

def rmcorr(data, x, y, subject):
    """Repeated measures correlation (Bakdash and Marusich 2017).
    
    https://www.frontiersin.org/articles/10.3389/fpsyg.2017.00456/full
    
    Tested against the `rmcorr` R package.
    
    Parameters
    ----------
    data : pd.DataFrame 
        Dataframe containing the variables
    x, y : string
        Name of columns in data containing the two dependent variables
    subject : string
        Name of column in data containing the subject indicator
    
    Returns
    -------
    r : float
        Repeated measures correlation coefficient  
    p : float
        P-value
    dof : int
        Degrees of freedom
        
    Notes
    -----
    Repeated measures correlation (rmcorr) is a statistical technique 
    for determining the common within-individual association for paired measures 
    assessed on two or more occasions for multiple individuals.
    
    Please note that NaN are automatically removed from the dataframe.
    
    Examples
    --------
    
        >>> import numpy as np
        >>> import pandas as pd
        >>> # Generate random correlated data
        >>> np.random.seed(123)
        >>> mean, cov = [4, 6], [[1, 0.6], [0.6, 1]]
        >>> x, y = np.round(np.random.multivariate_normal(mean, cov, 30), 1).T
        >>> data = pd.DataFrame({'X': x, 'Y': y, 'Ss': np.repeat(np.arange(10), 3)})
        >>> # Compute the repeated measure correlation
        >>> rmcorr(data, x='X', y='Y', subject='Ss')
            (0.647, 0.001, 19)
    """ 
    # Remove Nans
    data = data[[x, y, subject]].dropna(axis=0)
    # ANCOVA model
    formula = y + ' ~ ' + 'C(' + subject + ') + ' + x
    model = ols(formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=3)
    # Extract the sign of the correlation
    sign = np.sign(model.params[x])
    # Extract degrees of freedom
    dof = int(table.loc['Residual', 'df'])
    # Extract correlation coefficient from sum of squares
    ssfactor = table.loc[x, 'sum_sq']
    sserror = table.loc['Residual', 'sum_sq']
    rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    # Extract p-value
    pval = table.loc[x, 'PR(>F)']
    return np.round(rm, 3), pval, dof

def calculate_rm_correlation(df, x, y, subject):
    cordata = df[[x,y]]
    cols = cordata.columns
    cordata.columns = ['x', 'y']
    r, p, dof = rmcorr(data=cordata.reset_index(), x='x', y='y', subject=subject)
    
    return (cols[0],cols[1],r,p)

def run_rm_correlation(df, alpha=0.05, subject='subject', correction=('fdr', 'indep')):
    calculated = set()
    rows = []
    #df = df.dropna()._get_numeric_data()
    if not df.empty:
        df = df.set_index(subject)._get_numeric_data()
        start = time.time()
        for x, y in itertools.combinations(df.columns, 2):
            row = calculate_rm_correlation(df, x, y, subject)
            rows.append(row)
        end = time.time()
        print(end - start)

        print(rows)
        correlation = pd.DataFrame(rows, columns=["node1", "node2", "weight", "pvalue"])
        
        if correction[0] == 'fdr':
            rejected, padj = apply_pvalue_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])
        elif correction[0] == '2fdr':
            rejected, padj = apply_pvalue_twostage_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])

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
    t, pvalue = stats.ttest_rel(group1, group2, nan_policy='omit')
    log = -math.log(pvalue, 10)

    return (df.name, t, pvalue, log, mean1, mean2, log2fc)

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
    log = -math.log(pvalue, 10)

    return (df.name, t, pvalue, log, mean1, mean2, log2fc)

def calculate_THSD(df):
    col = df.name
    result = pairwise_tukeyhsd(df, list(df.index))
    df_results = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
    df_results.columns = ['group1', 'group2', 'log2FC', 'lower', 'upper', 'rejected']
    df_results['identifier'] = col
    df_results = df_results.set_index('identifier')
    df_results['FC'] = df_results['log2FC'].apply(lambda x: np.power(np.abs(x),2) * -1 if x < 0 else np.power(np.abs(x),2))
    df_results['rejected'] = df_results['rejected']

    return df_results

def calculate_pairwise_ttest(df, column, subject='subject', group='group', correction='none'):
    posthoc_columns = ['Contrast', 'group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired', 'T', 'tail', 'pvalue', 'BF10', 'efsize', 'eftype']
    if correction == "none":
        valid_cols = ['group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired', 'T', 'BF10', 'efsize', 'eftype']
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
    posthoc['FC'] = posthoc['log2FC'].apply(lambda x: np.power(np.abs(x),2) * -1 if x < 0 else np.power(np.abs(x),2))

    return posthoc

def calculate_anova(df, group='group'):
    col = df.name
    group_values = df.groupby(group).apply(list).tolist()
    t, pvalue = stats.f_oneway(*group_values)
    log = -math.log(pvalue,10)
    return (col, t, pvalue, log)

def calculate_repeated_measures_anova(df, column, subject='subject', group='group', alpha=0.05):
    aov_result = pg.rm_anova(dv=column, within=group,subject=subject, data=df, detailed=True, remove_na=True, correction=True)
    aov_result.columns = ['Source', 'SS', 'DF', 'MS', 'F', 'pvalue', 'padj (GG)', 'np2', 'eps', 'sphericity', 'Mauchlys sphericity', 'p-spher']
    t, pvalue = aov_result.loc[0, ['F', 'pvalue']].values 
    log = -math.log(pvalue,10)

    return (column, t, pvalue, log)

def get_max_permutations(df, group='group'):
    num_groups = len(list(df.index))
    num_per_group = df.groupby(group).size().tolist()
    max_perm = factorial(num_groups)/np.prod(factorial(np.array(num_per_group)))

    return max_perm

def check_is_paired(df, subject, group):
    count_subject_groups = df.groupby(subject)[group].count()

    return (count_subject_groups > 1).all()

def anova(df, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=50):
    columns = ['identifier', 't-statistics', 'pvalue', '-log10 pvalue']
    if check_is_paired(df, subject, group):
        groups = df[group].unique()
        drop_cols = [d for d in drop_cols if d != subject]
        if len(groups) == 2:
            res = ttest(df, groups[0], groups[1], alpha = alpha, drop_cols=drop_cols, subject=subject, group=group, paired=True, permutations=permutations)
        else:
            res = repeated_measurements_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, group=group, permutations=0)
    else:
        df = df.set_index([group])
        df = df.drop(drop_cols, axis=1)
        scores = df.apply(func = calculate_anova, axis=0, result_type='expand', group=group).T
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
        else:
            rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
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
            res = res.join(scores[['t-statistics', 'pvalue', '-log10 pvalue', 'padj']].astype('float'))
            res['correction'] = scores['correction']
        else:
            res = scores
            res["log2FC"] = np.nan

        res = res.reset_index()
        res['rejected'] = res['padj'] < alpha
        res['rejected'] = res['rejected'] 
    
    return res

def repeated_measurements_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', group='group', permutations=50):
    df = df.set_index([subject,group])
    df = df.drop(drop_cols, axis=1)
    aov_result = []
    for col in df.columns:
        aov = calculate_repeated_measures_anova(df[col].reset_index(), column=col, subject=subject, group=group, alpha=alpha)
        aov_result.append(aov)

    scores = pd.DataFrame(aov_result, columns = ['identifier','t-statistics', 'pvalue', '-log10 pvalue'])
    scores = scores.set_index('identifier')

    max_perm = get_max_permutations(df)
    #FDR correction
    if permutations > 0 and max_perm>=10:
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
        scores['rejected'] = scores['rejected'] 
    
    #sigdf = df[list(scores[scores.rejected].index)]
    res = None
    for col in df.columns:
        pairwise = calculate_pairwise_ttest(df[col].reset_index(), column=col, subject=subject, group=group)
        if res is None:
            res = pairwise
        else:
            res = pd.concat([res,pairwise], axis=0)
    if res is not None:
        res = res.join(scores[['t-statistics', 'pvalue', '-log10 pvalue', 'padj']].astype('float'))
        res['correction'] = scores['correction']
    else:
        res = scores
        res["log2FC"] = np.nan

    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha
    res['rejected'] = res['rejected'] 
    
    
    return res

def ttest(df, condition1, condition2, alpha = 0.05, drop_cols=["sample"], subject='subject', group='group', paired=False, permutations=50):
    df = df.set_index([subject, group])
    df = df.drop(drop_cols, axis = 1)
    #tdf = df.loc[[condition1, condition2],:].T
    #columns = ['identifier', 't-statistics', 'pvalue', '-log10 pvalue', 'mean(group1)', 'mean(group2)', 'log2FC']
    scores = None
    for col in df.columns:
        ttest = calculate_pairwise_ttest(df[col].reset_index(), col, subject=subject, group=group, correction='fdr_bh')
        if scores is None:
            scores = ttest
        else:
            scores = pd.concat([scores, ttest], axis=0)
    #if paired:
    #    scores = tdf.apply(func = calculate_paired_ttest, axis = 1, result_type='expand', args =(condition1, condition2))
    #else:
    #    scores = tdf.apply(func = calculate_ttest, axis = 1, result_type='expand', args =(condition1, condition2))
    #scores.columns = columns
    #scores = scores.dropna(how="all")

    #Cohen's d
    #scores["cohen_d"] = tdf.apply(func = cohen_d, axis = 1, args =(condition1, condition2, 1))

    #Hedge's g
    #scores["hedges_g"] = tdf.apply(func = hedges_g, axis = 1, args =(condition1, condition2, 1))
    max_perm = get_max_permutations(df)
    #FDR correction
    if permutations > 0 and max_perm>=10:
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
        scores['rejected'] = scores['rejected']
    scores['group1'] = condition1
    scores['group2'] = condition2
    scores['FC'] = scores['log2FC'].apply(lambda x: np.power(np.abs(x),2) * -1 if x < 0 else np.power(np.abs(x),2))
    scores['-log10 pvalue'] = scores['pvalue'].apply(lambda x: - math.log(x,10))
    scores = scores.reset_index() 

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

def runWGCNA(data, drop_cols_exp, drop_cols_cli, RsquaredCut=0.8, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0):
    dfs = wgcna.get_data(data, drop_cols_exp=drop_cols_exp, drop_cols_cli=drop_cols_cli)
    if 'clinical' in dfs:
        data_cli = dfs['clinical']
        data_exp, = [i for i in dfs.keys() if i != 'clinical']
        data_exp = dfs[data_exp]

        softPower = wgcna.pick_softThreshold(data_exp, RsquaredCut=RsquaredCut, networkType=networkType, verbose=verbose)
        dissTOM, moduleColors = wgcna.build_network(data_exp, softPower=softPower, networkType=networkType, minModuleSize=minModuleSize, deepSplit=deepSplit,
                                              pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules, MEDissThres=MEDissThres, verbose=verbose)

        Features_per_Module = wgcna.get_FeaturesPerModule(data_exp, moduleColors, mode='dataframe')
        MEs = wgcna.calculate_module_eigengenes(data_exp, moduleColors, softPower=softPower, dissimilarity=False)
        moduleTraitCor, textMatrix = wgcna.calculate_ModuleTrait_correlation(data_exp, data_cli, MEs)
        MM, MMPvalue = wgcna.calculate_ModuleMembership(data_exp, MEs)
        FS, FSPvalue = wgcna.calculate_FeatureTraitSignificance(data_exp, data_cli)
        METDiss, METcor = wgcna.get_EigengenesTrait_correlation(MEs, data_cli)

        return data_exp, data_cli, dissTOM, moduleColors, Features_per_Module, MEs, moduleTraitCor, textMatrix, MM, MMPvalue, FS, FSPvalue, METDiss, METcor

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
