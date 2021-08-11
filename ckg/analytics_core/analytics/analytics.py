import warnings
import os
import uuid
import time
import pandas as pd
import collections
import re
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.utils import shuffle
from statsmodels.stats import multitest
from statsmodels.stats.power import FTestAnovaPower
import scipy.stats
from scipy.special import factorial, betainc
import umap
from sklearn import preprocessing, ensemble, cluster, metrics
from scipy import stats
import gseapy as gp
import pingouin as pg
import numpy as np
import networkx as nx
import community
import snf
import math
from sklearn.impute import KNNImputer
import kmapper as km
from ckg import ckg_utils
from ckg.analytics_core import utils
from ckg.analytics_core.analytics import wgcnaAnalysis as wgcna
from ckg.analytics_core.analytics import kaplan_meierAnalysis
import statsmodels.api as sm
from statsmodels.formula.api import ols
from combat.pycombat import pycombat

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

try:
    from rpy2 import robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    # pandas2ri.activate()
    R = ro.r
    base = importr('base')
    stats_r = importr('stats')
    samr = importr('samr')
    r_installation = True
except ImportError:
    r_installation = False
    print("R functions will not work. Module Rpy2 not installed.")
except Exception as e:
    r_installation = False
    print("Error loading R package.", e)


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    :param tuple vector: vector
    :return tuple unit_vector: unit vector
    """
    return vector / np.linalg.norm(vector)


def flatten(t, my_list=[]):
    """
    Code from: https://gist.github.com/shaxbee/0ada767debf9eefbdb6e
    Acknowledgements: Zbigniew Mandziejewicz (shaxbee)
    Generator flattening the structure

    >>> list(flatten([2, [2, (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
    [2, 2, 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
    """

    for x in t:
        if not isinstance(x, collections.Iterable) or isinstance(x, str):
            my_list.append(x)
        else:
            flatten(x, my_list)

    return my_list


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    :param tuple v1: vector 1
    :param tuple v2: vector 2
    :return float angle: angle between two vectors in radians

    Example::
        angle = angle_between((1, 0, 0), (0, 1, 0))
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def transform_into_wide_format(data, index, columns, values, extra=[]):
    """
    This function converts a Pandas DataFrame from long to wide format using
    pandas pivot_table() function.

    :param data: long-format Pandas DataFrame
    :param list index: columns that will be converted into the index
    :param str columns: column name whose unique values will become the new column names
    :param str values: column to aggregate
    :param list extra: additional columns to be kept as columns
    :return: Wide-format pandas DataFrame

    Example::

        result = transform_into_wide_format(df, index='index', columns='x', values='y', extra='group')

    """
    df = pd.DataFrame()
    extra_cols = None
    if data is not None:
        df = data.copy()
        if not df.empty:
            if index is None:
                df = df.reset_index()
                index = 'index'
            cols = []
            utils.append_to_list(cols, columns)
            utils.append_to_list(cols, values)
            utils.append_to_list(cols, index)
            if len(extra) > 0:
                utils.append_to_list(extra, index)
                extra_cols = df[extra].set_index(index)
            df = df[cols]
            df = df.drop_duplicates()
            if isinstance(index, list):
                df = df.pivot_table(index=index, columns=columns, values=values, aggfunc='first')
            else:
                df = df.pivot(index=index, columns=columns, values=values)
            if extra_cols is not None:
                df = df.join(extra_cols)

            df = df.drop_duplicates()
            df = df.reset_index()

    return df


def transform_into_long_format(data, drop_columns, group, columns=['name', 'y']):
    """
    Converts a Pandas DataDrame from wide to long format using pd.melt()
    function.

    :param data: wide-format Pandas DataFrame
    :param list drop_columns: columns to be deleted
    :param group: column(s) to use as identifier variables
    :type group: str or list
    :param list columns: names to use for the 1)variable column, and for the 2)value column
    :return: Long-format Pandas DataFrame.

    Example::

        result = transform_into_long_format(df, drop_columns=['sample', 'subject'], group='group', columns=['name','y'])
    """
    long_data = pd.DataFrame()
    if data is not None:
        data = data.drop(drop_columns, axis=1)

        long_data = pd.melt(data, id_vars=group, var_name=columns[0], value_name=columns[1])
        long_data = long_data.set_index(group)
        long_data.columns = columns

    return long_data


def get_ranking_with_markers(data, drop_columns, group, columns, list_markers, annotation={}):
    """
    This function creates a long-format dataframe with features and values to be plotted together with disease biomarker annotations.

    :param data: wide-format Pandas DataFrame with samples as rows and features as columns
    :param list drop_columns: columns to be deleted
    :param str group: column to use as identifier variables
    :param list columns: names to use for the 1)variable column, and for the 2)value column
    :param list list_markers: list of features from data, known to be markers associated to disease.
    :param dict annotation: markers, from list_markers, and associated diseases.
    :return: Long-format pandas DataFrame with group identifiers as rows and columns: 'name' (identifier), 'y' (LFQ intensity), 'symbol' and 'size'.

    Example::

        result = get_ranking_with_markers(data, drop_columns=['sample', 'subject'], group='group', columns=['name', 'y'], list_markers, annotation={})
    """
    long_data = pd.DataFrame()
    if data is not None:
        long_data = transform_into_long_format(data, drop_columns, group, columns)
        if len(set(long_data['name'].values.tolist()).intersection(list_markers)) > 0:
            long_data = long_data.drop_duplicates()
            long_data['symbol'] = [17 if p in list_markers else 0 for p in long_data['name'].tolist()]
            long_data['size'] = [25 if p in list_markers else 7 for p in long_data['name'].tolist()]
            long_data['name'] = [p+' marker in '+annotation[p] if p in annotation else p for p in long_data['name'].tolist()]

    return long_data


def extract_number_missing(data, min_valid, drop_cols=['sample'], group='group'):
    """
    Counts how many valid values exist in each column and filters column labels with more valid values than the minimum threshold defined.

    :param data: pandas DataFrame with group as rows and protein identifier as column.
    :param str group: column label containing group identifiers. If None, number of valid values is counted across all samples, otherwise is counted per unique group identifier.
    :param int min_valid: minimum number of valid values to be filtered.
    :param list drop_columns: column labels to be dropped.
    :return: List of column labels above the threshold.

    Example::

        result = extract_number_missing(data, min_valid=3, drop_cols=['sample'], group='group')
    """
    if group is None:
        groups = data.loc[:, data.notnull().sum(axis=0) >= min_valid]
    else:
        groups = data.copy()
        groups = groups.drop(drop_cols, axis=1)
        groups = groups.set_index(group).notnull().groupby(level=0).sum(axis=1)
        groups = groups[groups >= min_valid]

    groups = groups.dropna(how='all', axis=1)
    return groups.columns.unique().tolist()


def extract_percentage_missing(data, missing_max, drop_cols=['sample'], group='group', how='all'):
    """
    Extracts ratio of missing/valid values in each column and filters column labels with lower ratio than the minimum threshold defined.

    :param data: pandas dataframe with group as rows and protein identifier as column.
    :param str group: column label containing group identifiers. If None, ratio is calculated across all samples, otherwise is calculated per unique group identifier.
    :param float missing_max: maximum ratio of missing/valid values to be filtered.
    :param str how: define if labels with a higher percentage of missing values than the threshold in any group ('any') or in all groups ('all') should be filtered
    :return: List of column labels below the threshold.

    Example::
        result = extract_percentage_missing(data, missing_max=0.3, drop_cols=['sample'], group='group')
    """
    if group is None:
        groups = data.loc[:, data.isnull().mean() <= missing_max].columns
    else:
        groups = data.copy()
        groups = groups.drop(drop_cols, axis=1)
        groups = groups.set_index(group)
        groups = groups.isnull().groupby(level=0).mean()
        groups = groups[groups <= missing_max]
        if how == 'all':
            groups = groups.dropna(how='all', axis=1).columns.unique().tolist()
        elif how == 'any':
            groups = groups.dropna(how='any', axis=1).columns.unique().tolist()
        else:
            if how in groups.index:
                groups = groups.loc[how, :].dropna().index.unique().tolist()

    return groups


def imputation_KNN(data, drop_cols=['group', 'sample', 'subject'], group='group', cutoff=0.6, alone=True):
    """
    k-Nearest Neighbors imputation for pandas dataframes with missing data. For more information visit https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/knn.py.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str group: column label containing group identifiers.
    :param list drop_cols: column labels to be dropped. Final dataframe should only have gene/protein/etc identifiers as columns.
    :param float cutoff: minimum ratio of missing/valid values required to impute in each column.
    :param boolean alone: if True removes all columns with any missing values.
    :return: Pandas dataframe with samples as rows and protein identifiers as columns.

    Example::

        result = imputation_KNN(data, drop_cols=['group', 'sample', 'subject'], group='group', cutoff=0.6, alone=True)
    """
    np.random.seed(112736)
    df = data.copy()
    cols = df.columns
    df = df._get_numeric_data()
    if group in data.columns:
        df[group] = data[group]
        cols = list(set(cols).difference(df.columns))
        value_cols = [c for c in df.columns if c not in drop_cols]
        for g in df[group].unique():
            missDf = df.loc[df[group] == g, value_cols]
            missDf = missDf.loc[:, missDf.notnull().mean() >= cutoff].dropna(axis=1, how='all')
            if missDf.isnull().values.any():
                X = np.array(missDf.values, dtype=np.float64)
                X_trans = KNNImputer(n_neighbors=3).fit_transform(X)
                missingdata_df = missDf.columns.tolist()
                dfm = pd.DataFrame(X_trans, index=list(missDf.index), columns=missingdata_df)
                df.update(dfm)
        if alone:
            df = df.dropna(axis=1)

        df = df.join(data[cols])
    
    return df


def imputation_mixed_norm_KNN(data, index_cols=['group', 'sample', 'subject'], shift=1.8, nstd=0.3, group='group', cutoff=0.6):
    """
    Missing values are replaced in two steps: 1) using k-Nearest Neighbors we impute protein columns with a higher ratio of missing/valid values than the defined cutoff, \
    2) the remaining missing values are replaced by random numbers that are drawn from a normal distribution.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str group: column label containing group identifiers.
    :param list index_cols: list of column labels to be set as dataframe index.
    :param float shift: specifies the amount by which the distribution used for the random numbers is shifted downwards. This is in units of the \
                        standard deviation of the valid data.
    :param float nstd: defines the width of the Gaussian distribution relative to the standard deviation of measured values. \
                        A value of 0.5 would mean that the width of the distribution used for drawing random numbers is half of the standard deviation of the data.
    :param float cutoff: minimum ratio of missing/valid values required to impute in each column.
    :return: Pandas dataframe with samples as rows and protein identifiers as columns.

    Example::

        result = imputation_mixed_norm_KNN(data, index_cols=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3, group='group', cutoff=0.6)
    """
    df = imputation_KNN(data, drop_cols=index_cols, group=group, cutoff=cutoff, alone=False)
    df = imputation_normal_distribution(df, index_cols=index_cols, shift=shift, nstd=nstd)

    return df


def imputation_normal_distribution(data, index_cols=['group', 'sample', 'subject'], shift=1.8, nstd=0.3):
    """
    Missing values will be replaced by random numbers that are drawn from a normal distribution. The imputation is done for each sample (across all proteins) separately.
    For more information visit http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list index_cols: list of column labels to be set as dataframe index.
    :param float shift: specifies the amount by which the distribution used for the random numbers is shifted downwards. This is in units of the standard deviation of the valid data.
    :param float nstd: defines the width of the Gaussian distribution relative to the standard deviation of measured values. A value of 0.5 would mean that the width of the distribution used for drawing random numbers is half of the standard deviation of the data.
    :return: Pandas dataframe with samples as rows and protein identifiers as columns.

    Example::

        result = imputation_normal_distribution(data, index_cols=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3)
    """
    np.random.seed(112736)
    df = data.copy()
    if index_cols is not None:
        df = df.set_index(index_cols)

    data_imputed = df.T.sort_index()
    null_columns = data_imputed.isnull().any().index.tolist()
    for c in null_columns:
        missing = data_imputed[data_imputed[c].isnull()].index.tolist()
        std = data_imputed[c].std()
        mean = data_imputed[c].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        value = 0.0
        if not math.isnan(std) and not math.isnan(mean) and not math.isnan(sigma) and not math.isnan(mu):
            value = np.random.normal(mu, sigma, size=len(missing))
        i = 0
        for m in missing:
            if not isinstance(value, np.ndarray):
                data_imputed.loc[m, c] = value
            else:
                data_imputed.loc[m, c] = value[i]
                i += 1

    return data_imputed.T

def combat_batch_correction(data, batch_col, index_cols):
    """
    This function corrects processed data for batch effects. For more information visit: https://pypi.org/project/pycombat/
    :param data: pandas dataframe with samples as rows and protein identifiers as columns.
    :param batch_col: column with the batch identifiers
    :param index_cols: list of columns that don't need to be corrected (i.e group)
    :return: pandas dataframe with samples as rows and protein identifiers as columns.
    Example::
        result = combat_batch_correction(data, batch_col='batch', index_cols=['subject', 'sample', 'group'])
    """
    df_corrected = pd.DataFrame()
    index_cols = [c for c in index_cols if c != batch_col]
    data = data.set_index(index_cols)
    df = data.drop(batch_col, axis=1)
    df_numeric = df._get_numeric_data()
    num_batches = len(data[batch_col].unique())
    if not df_numeric.empty and num_batches > 1:
        info_cols = list(set(df.columns.tolist()).difference(df_numeric.columns))
        df_corrected = pd.DataFrame(pycombat(df_numeric.T, data[batch_col].values).T, index = df.index)
        df_corrected = df_corrected.join(df[info_cols])
        df_corrected = df_corrected.reset_index()

    return df_corrected


def normalize_data_per_group(data, group, method='median', normalize=None):
    """
    This function normalizes the data by group using the selected method

    :param data: DataFrame with the data to be normalized (samples x features)
    :param group_col: Column containing the groups
    :param str method: normalization method to choose among: median_polish, median,
                        quantile, linear
    :param str normalize: whether the normalization should be done by 'features' (columns) or 'samples' (rows) (default None)
    :return: Pandas dataframe.

    Example::

        result = normalize_data_per_group(data, group='group' method='median')
    """
    ndf = pd.DataFrame(columns=data.columns)
    for n, gdf in data.groupby(group):
        norm_group = normalize_data(gdf, method=method, normalize=normalize)
        ndf = ndf.append(norm_group)

    return ndf


def normalize_data(data, method='median', normalize=None):
    """
    This function normalizes the data using the selected method

    :param data: DataFrame with the data to be normalized (samples x features)
    :param string method: normalization method to choose among: median_polish, median,
                        quantile, linear
    :param str normalize: whether the normalization should be done by 'features' (columns) or 'samples' (rows) (default None)
    :return: Pandas dataframe.

    Example::

        result = normalize_data(data, method='median_polish')
    """
    normData = None
    numeric_cols = data.select_dtypes(include=['int64', 'float64'])
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64'])
    if not numeric_cols.empty:
        if method == 'median_polish':
            normData = median_polish_normalization(numeric_cols, max_iter=250)
        elif method == 'median_zero':
            normData = median_zero_normalization(numeric_cols, normalize)
        elif method == 'median':
            normData = median_normalization(numeric_cols, normalize)
        elif method == 'quantile':
            normData = quantile_normalization(numeric_cols)
        elif method == 'linear':
            normData = linear_normalization(numeric_cols, method="l1", normalize=normalize)
        elif method == 'zscore':
            normData = zscore_normalization(numeric_cols, normalize)

    if non_numeric_cols is not None and not non_numeric_cols.empty:
        normData = normData.join(non_numeric_cols)

    return normData


def median_zero_normalization(data, normalize='samples'):
    """
    This function normalizes each sample by using its median.

    :param data:
    :param str normalize: whether the normalization should be done by 'features' (columns) or 'samples' (rows)
    :return: Pandas dataframe.

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = median_normalization(data, normalize='samples')
        result
                a         b         c
            0 -1.333333  0.666667  0.666667
            1 -2.666667 -3.666667  6.333333
            2 -2.000000  0.000000  2.000000
            3 -2.333333 -0.333333  2.666667
            4 -2.000000 -2.000000  4.000000
    """
    if normalize is None or normalize == 'samples':
        normData = data.sub(data.median(axis=1), axis=0)
    else:
        normData = data.sub(data.median(axis=0), axis=1)

    return normData


def median_normalization(data, normalize='samples'):
    """
    This function normalizes each sample by using its median.

    :param data:
    :param str normalize: whether the normalization should be done by 'features' (columns) or 'samples' (rows)
    :return: Pandas dataframe.

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = median_normalization(data, normalize='samples')
        result
                a         b         c
            0 -1.333333  0.666667  0.666667
            1 -2.666667 -3.666667  6.333333
            2 -2.000000  0.000000  2.000000
            3 -2.333333 -0.333333  2.666667
            4 -2.000000 -2.000000  4.000000
    """
    if normalize is None or normalize == 'samples':
        normData = data.sub(data.median(axis=1) - data.median(axis=1).median(), axis=0)
    else:
        normData = data.sub(data.median(axis=0) - data.median(axis=0).median(), axis=1)

    return normData


def zscore_normalization(data, normalize='samples'):
    """
    This function normalizes each sample by using its mean and standard deviation (mean=0, std=1).

    :param data:
    :param str normalize: whether the normalization should be done by 'features' (columns) or 'samples' (rows)
    :return: Pandas dataframe.

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = zscore_normalization(data, normalize='samples')
        result
                  a         b         c
                0 -1.154701  0.577350  0.577350
                1 -0.484182 -0.665750  1.149932
                2 -1.000000  0.000000  1.000000
                3 -0.927173 -0.132453  1.059626
                4 -0.577350 -0.577350  1.154701
    """
    if normalize is None or normalize == 'samples':
        normData = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)

    else:
        normData = data.sub(data.mean(axis=0), axis=1).div(data.std(axis=0), axis=1)

    return normData


def median_polish_normalization(data, max_iter=250):
    """
    This function iteratively normalizes each sample and each feature to its
    median until medians converge.

    :param data:
    :param int max_iter: number of maximum iterations to prevent infinite loop.
    :return: Pandas dataframe.

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = median_polish_normalization(data, max_iter = 10)
        result
                a    b     c
            0  2.0  4.0   7.0
            1  5.0  7.0  10.0
            2  4.0  6.0   9.0
            3  3.0  5.0   8.0
            4  3.0  5.0   8.0
    """
    mediandf = data.copy()
    for i in range(max_iter):
        row_median = mediandf.median(axis=1)
        mediandf = mediandf.sub(row_median, axis=0)
        col_median = mediandf.median(axis=0)
        mediandf = mediandf.sub(col_median, axis=1)
        if (mediandf.median(axis=0) == 0).all() and (mediandf.median(axis=1) == 0).all():
            break

    normData = data - mediandf

    return normData


def quantile_normalization(data):
    """
    Applies quantile normalization to each column in pandas dataframe.

    :param data: pandas dataframe with features as columns and samples as rows.
    :return: Pandas dataframe

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = quantile_normalization(data)
        result
                a    b    c
            0  3.2  4.6  4.6
            1  4.6  3.2  8.6
            2  3.2  4.6  8.6
            3  3.2  4.6  8.6
            4  3.2  3.2  8.6
    """
    rank_mean = data.T.stack().groupby(data.T.rank(method='first').stack().astype(int)).mean()
    normdf = data.T.rank(method='min').stack().astype(int).map(rank_mean).unstack().T

    return normdf


def linear_normalization(data, method="l1", normalize='samples'):
    """
    This function scales input data to a unit norm. For more information visit https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html.

    :param data: pandas dataframe with samples as rows and features as columns.
    :param str method: norm to use to normalize each non-zero sample or non-zero feature (depends on axis).
    :param str normalize: axis used to normalize the data along. If 'samples', independently normalize each sample, if 'features' normalize each feature.
    :return: Pandas dataframe

    Example::
        data = pd.DataFrame({'a': [2,5,4,3,3], 'b':[4,4,6,5,3], 'c':[4,14,8,8,9]})
        result = linear_normalization(data, method = "l1", by = 'feature')
        result
                a         b         c
            0  0.117647  0.181818  0.093023
            1  0.294118  0.181818  0.325581
            2  0.235294  0.272727  0.186047
            3  0.176471  0.227273  0.186047
            4  0.176471  0.136364  0.209302
    """
    if normalize is None or normalize == 'samples':
        normvalues = preprocessing.normalize(data.fillna(0).values, norm=method, axis=0, copy=True, return_norm=False)
    else:
        normvalues = preprocessing.normalize(data.fillna(0).values, norm=method, axis=1, copy=True, return_norm=False)

    normdf = pd.DataFrame(normvalues, index=data.index, columns=data.columns)

    return normdf


def remove_group(data):
    """
    Removes column with label 'group'.

    :param data: pandas dataframe with one column labelled 'group'
    :return: Pandas dataframe

    Example::

        result = remove_group(data)
    """
    data.drop(['group'], axis=1)
    return data


def calculate_coefficient_variation(values):
    """
    Compute the coefficient of variation, the ratio of the biased standard
    deviation to the mean, in percentage. For more information
    visit https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.variation.html.

    :param ndarray values: numpy array
    :return: The calculated variation along rows.
    :rtype: ndarray

    Example::

        result = calculate_coefficient_variation()
    """
    cv = scipy.stats.variation(values.apply(lambda x: np.power(2, x)).values) * 100

    return cv


def get_coefficient_variation(data, drop_columns, group, columns=['name', 'y']):
    """
    Extracts the coefficients of variation in each group.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list drop_columns: column labels to be dropped from the dataframe
    :param str group: column label containing group identifiers.
    :param list columns: names to use for the variable column(s), and for the value column(s)
    :return: Pandas dataframe with columns 'name' (protein identifier), 'x' (coefficient of variation), 'y' (mean) and 'group'.

    Exmaple::

        result = get_coefficient_variation(data, drop_columns=['sample', 'subject'], group='group')
    """
    df = data.copy()
    formated_df = df.drop(drop_columns, axis=1)
    cvs = formated_df.groupby(group).apply(func=calculate_coefficient_variation)
    cols = formated_df.set_index(group).columns.tolist()
    cvs_df = pd.DataFrame()
    for i in cvs.index:
        gcvs = cvs[i].tolist()
        ints = formated_df.set_index('group').mean().values.tolist()
        tdf = pd.DataFrame(data={'name': cols, 'x': gcvs, 'y': ints})
        tdf['group'] = i

        if cvs_df.empty:
            cvs_df = tdf.copy()
        else:
            cvs_df = cvs_df.append(tdf)

    return cvs_df


def transform_proteomics_edgelist(df, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', value_col='LFQ_intensity'):
    """
    Transforms a long format proteomics matrix into a wide format

    :param df: long-format pandas dataframe with columns 'group', 'sample', 'subject', 'identifier' (protein), 'name' (gene) and 'LFQ_intensity'.
    :param list index_cols: column labels to be be kept as index identifiers.
    :param list drop_cols: column labels to be dropped from the dataframe.
    :param str group: column label containing group identifiers.
    :param str identifier: column label containing feature identifiers.
    :param str extra_identifier: column label containing additional protein identifiers (e.g. gene names).
    :param str value_col: column label containing expression values.

    :return: Pandas dataframe with samples as rows and protein identifiers (UniprotID~GeneName) as columns (with additional columns 'group', 'sample' and 'subject').

    Example:
        df = transform_proteomics_edgelist(original, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', value_col='LFQ_intensity')
    """
    wdf = None
    if df.columns.isin(index_cols).sum() == len(index_cols):
        wdf = df[df[group].notna()]
        wdf[index_cols] = wdf[index_cols].astype(str)
        wdf = wdf.set_index(index_cols)
        if extra_identifier is not None and extra_identifier in wdf.columns:
            wdf[identifier] = wdf[extra_identifier].map(str) + "~" + wdf[identifier].map(str)
        wdf = wdf.pivot_table(values=value_col, index=wdf.index, columns=identifier, aggfunc='first')
        wdf = wdf.reset_index()
        wdf[index_cols] = wdf["index"].apply(pd.Series)
        wdf = wdf.drop(["index"], axis=1)

    return wdf


def get_proteomics_measurements_ready(df, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', filter_samples=False, filter_samples_percent=0.5, imputation=True, imputation_method='distribution', missing_method='percentage', missing_per_group=True, missing_max=0.3, min_valid=1, value_col='LFQ_intensity', shift=1.8, nstd=0.3, knn_cutoff=0.6, normalize=False, normalization_method='median', normalize_group=False, normalize_by=None):
    """
    Processes proteomics data extracted from the database: 1) filter proteins with high number of missing values (> missing_max or min_valid), 2) impute missing values.

    :param df: long-format pandas dataframe with columns 'group', 'sample', 'subject', 'identifier' (protein), 'name' (gene) and 'LFQ_intensity'.
    :param list index_cols: column labels to be be kept as index identifiers.
    :param list drop_cols: column labels to be dropped from the dataframe.
    :param str group: column label containing group identifiers.
    :param str identifier: column label containing feature identifiers.
    :param str extra_identifier: column label containing additional protein identifiers (e.g. gene names).
    :param bool filter_samples: if True filter samples with valid values below percentage (filter_samples_percent).
    :param float filter_samples_percent: defines the maximum percentage of missing values allowed in a sample.
    :param bool imputation: if True performs imputation of missing values.
    :param str imputation_method:  method for missing values imputation ('KNN', 'distribuition', or 'mixed')
    :param str missing_method: defines which expression rows are counted to determine if a column has enough valid values to survive the filtering process.
    :param bool missing_per_group: if True filter proteins based on valid values per group; if False filter across all samples.
    :param float missing_max: maximum ratio of missing/valid values to be filtered.
    :param int min_valid: minimum number of valid values to be filtered.
    :param str value_col: column label containing expression values.
    :param float shift: when using distribution imputation, the down-shift
    :param float nstd: when using distribution imputation, the width of the distribution
    :param float knn_cutoff: when using KNN imputation, the minimum percentage of valid values for which to use KNN imputation (i.e. 0.6 -> if 60% valid values use KNN, otherwise MinProb)
    :param bool normalize: whether or not to normalize the data
    :param str normalization_method: method to be used to normalize the data ('median', 'quantile', 'linear', 'zscore', 'median_polish') (only with normalize=True)
    :param bool normalize_group: normalize per group or not (only with normalize=True)
    :param str normalize_by: whether the normalization should be done by 'features' (columns) or 'samples' (rows) (only with normalize=True)
    :return: Pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').

    Example 1::

        result = get_proteomics_measurements_ready(df, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', imputation=True, method = 'distribution', missing_method = 'percentage', missing_per_group=True, missing_max = 0.3, value_col='LFQ_intensity')

    Example 2::

        result = get_proteomics_measurements_ready(df, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', imputation = True, method = 'mixed', missing_method = 'at_least_x', missing_per_group=False, min_valid=5, value_col='LFQ_intensity')
    """
    df = transform_proteomics_edgelist(df, index_cols=index_cols, drop_cols=drop_cols, group=group, identifier=identifier, extra_identifier=extra_identifier, value_col=value_col)
    if df is not None:
        aux = []
        aux.extend(index_cols)
        g = group
        if not missing_per_group:
            g = None
        if missing_method == 'at_least_x':
            aux.extend(extract_number_missing(df, min_valid, drop_cols, group=g))
        elif missing_method == 'percentage':
            aux.extend(extract_percentage_missing(df,  missing_max, drop_cols, group=g))

        df = df[list(set(aux))]
        if filter_samples:
            df = df.loc[~(df.T.isna().mean() > filter_samples_percent)]

        if normalize:
            if not normalize_group:
                df = normalize_data(df, method=normalization_method, normalize=normalize_by)
            else:
                df = normalize_data_per_group(df, group=group, method=normalization_method, normalize=normalize_by)

        if imputation:
            if imputation_method.lower() == "knn":
                df = imputation_KNN(df, drop_cols=index_cols, group=group, cutoff=knn_cutoff, alone=True)
            elif imputation_method == "distribution":
                df = imputation_normal_distribution(df, index_cols=index_cols, shift=shift, nstd=nstd)
                df = df.reset_index()
            elif imputation_method == 'mixed':
                df = imputation_mixed_norm_KNN(df, index_cols=index_cols, shift=shift, nstd=nstd, group=group, cutoff=knn_cutoff)
                df = df.reset_index()
        else:
            df = df.set_index(index_cols).dropna(axis=1).reset_index()
    return df


def get_clinical_measurements_ready(df, subject_id='subject', sample_id='biological_sample', group_id='group', columns=['clinical_variable'], values='values', extra=['group'], imputation=True, imputation_method='KNN', missing_method='percentage', missing_max=0.3, min_valid=1):
    """
    Processes clinical data extracted from the database by converting dataframe to wide-format and imputing missing values.

    :param df: long-format pandas dataframe with columns 'group', 'biological_sample', 'subject', 'clinical_variable', 'value'.
    :param str subject_id: column label containing subject identifiers.
    :param str sample_id: column label containing biological sample identifiers.
    :param str group_id: column label containing group identifiers.
    :param list columns: column name whose unique values will become the new column names
    :param str values: column label containing clinical variable values.
    :param list extra: additional column labels to be kept as columns
    :param bool imputation: if True performs imputation of missing values.
    :param str imputation_method: method for missing values imputation ('KNN', 'distribuition', or 'mixed').
    :param str missing_method: defines which expression rows are counted to determine if a column has enough valid values to survive the filtering process.
    :param float missing_max: maximum ratio of missing/valid values to be filtered.
    :param int min_valid: minimum number of valid values to be filtered.
    :return: Pandas dataframe with samples as rows and clinical variables as columns (with additional columns 'group', 'subject' and 'biological_sample').

    Example::

        result = get_clinical_measurements_ready(df, subject_id='subject', sample_id='biological_sample', group_id='group', columns=['clinical_variable'], values='values', extra=['group'], imputation=True, imputation_method='KNN')
    """
    index = [subject_id, sample_id]
    aux = [group_id]
    aux.extend(index)
    drop_cols = index

    processed_df = df[df['rel_type'] == 'HAS_QUANTIFIED_CLINICAL']
    processed_df[values] = processed_df[values].astype('float')
    processed_df = transform_into_wide_format(processed_df, index=index, columns=columns, values=values, extra=extra)
    if missing_method == 'at_least_x':
        aux.extend(extract_number_missing(processed_df, min_valid, drop_cols, group=group_id))
    elif missing_method == 'percentage':
        aux.extend(extract_percentage_missing(processed_df,  missing_max, drop_cols, group=group_id))

    processed_df = processed_df[list(set(aux))]
    drop_cols.append(group_id)
    if imputation:
        if imputation_method.lower() == "knn":
            df = imputation_KNN(processed_df, drop_cols=drop_cols, group=group_id, cutoff=0.0)
        elif imputation_method.lower() == "distribution":
            df = imputation_normal_distribution(processed_df, index_cols=index)
        elif imputation_method.lower() == 'mixed':
            df = imputation_mixed_norm_KNN(processed_df,index_cols=index, group=group_id)

    return df


def get_summary_data_matrix(data):
    """
    Returns some statistics on the data matrix provided.

    :param data: pandas dataframe.
    :return: dictionary with the type of statistics as key and the statistic as value in the shape of a pandas data frame

    Example::

        result = get_summary_data_matrix(data)
    """
    summary = {}
    summary["Data Matrix Shape"] = pd.DataFrame(data=[data.shape], columns=["Rows", "Columns"])
    summary["Stats"] = data.describe().transpose().reset_index()

    return summary


def check_equal_variances(data, drop_cols=['group','sample', 'subject'], group_col='group', alpha=0.05):
    levene_results = []
    for c in data.columns.drop(drop_cols):
        values = []
        for group, g in data.groupby(group_col):
            values.append(g[c].values)
        levene_results.append(stats.levene(*values)+(c,))
    levene_results = pd.DataFrame(levene_results, columns=['test', 'pvalue','identifier'])
    levene_results['pass'] = levene_results['pvalue'] > alpha

    return levene_results


def check_normality(data, drop_cols=['group','sample', 'subject'], group_col='group', alpha=0.05):
    shapiro_wilk_results = []
    for group in data[group_col].unique().tolist():
        for c in data.columns.drop(drop_cols):
            shapiro_wilk_results.append(stats.shapiro(data.loc[data[group_col] == group, c].values)+(group, c))
    shapiro_wilk_results = pd.DataFrame(shapiro_wilk_results, columns=['test', 'pvalue', 'group', 'identifier'])
    shapiro_wilk_results['pass'] = shapiro_wilk_results['pvalue'] > alpha

    return shapiro_wilk_results


def run_pca(data, drop_cols=['sample', 'subject'], group='group', annotation_cols=['sample'], components=2, dropna=True):
    """
    Performs principal component analysis and returns the values of each component for each sample and each protein, and the loadings for each protein. \
    For information visit https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list drop_cols: column labels to be dropped from the dataframe.
    :param str group: column label containing group identifiers.
    :param list annotation_cols: list of columns to be added in the scatter plot annotation
    :param int components: number of components to keep.
    :param bool dropna: if True removes all columns with any missing values.
    :return: tuple: 1) three pandas dataframes: components, loadings and variance; 2) xaxis and yaxis titles with components loadings for plotly.

    Example::

        result = run_pca(data, drop_cols=['sample', 'subject'], group='group', components=2, dropna=True)
    """

    np.random.seed(112736)
    resultDf = pd.DataFrame()
    loadings = pd.DataFrame()
    var_exp = []
    args = {}
    if not data.empty:
        df = data.copy()
        annotations = pd.DataFrame()
        if annotation_cols is not None: 
            if len(list(set(annotation_cols).intersection(data.columns))) > 0:
                annotations = data.set_index(group)[annotation_cols]
        drop_cols_int = list(set(drop_cols).intersection(df.columns))
        if len(drop_cols_int) > 0:
            df = df.drop(drop_cols_int, axis=1)

        y = df[group].tolist()
        df = df.set_index(group)
        df = df.select_dtypes(['number'])
        if dropna:
            df = df.dropna(axis=1)
        X = df.values
        
        
        if X.size > 0 and X.shape[1] > components:
            pca = PCA(n_components=components)
            X = pca.fit_transform(X)
            var_exp = pca.explained_variance_ratio_
            loadings = pd.DataFrame(pca.components_.transpose())
            loadings.index = df.columns
            values = {index:np.sqrt(np.power(row, 2).sum()) for index, row in loadings.iterrows()}
            loadings['value'] = loadings.index.map(values.get)
            loadings = loadings.sort_values(by='value', ascending=False)
            args = {"x_title":"PC1"+" ({0:.2f})".format(var_exp[0]),"y_title":"PC2"+" ({0:.2f})".format(var_exp[1]), 'group':'group'}
            if components == 2:
                resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
                resultDf = resultDf.assign(**annotations)
                resultDf = resultDf.reset_index()
                resultDf.columns = ["group", "x", "y"] + annotation_cols

                loadings.columns = ['x', 'y', 'value']
            if components > 2:
                args.update({"z_title":"PC3"+" ({0:.2f})".format(var_exp[2])})
                resultDf = pd.DataFrame(X, index = y)
                resultDf = resultDf.assign(**annotations)
                resultDf = resultDf.reset_index()
                pca_cols = []
                loading_cols = []
                if components>3:
                    pca_cols = [str(i) for i in resultDf.columns[4:]]
                    loading_cols = [str(i) for i in loadings.columns[3:]]

                resultDf.columns = ["group", "x", "y", "z"]  + pca_cols
                loadings.columns = ['x', 'y', 'z'] + loading_cols

    return (resultDf, loadings, var_exp), args


def run_tsne(data, drop_cols=['sample', 'subject'], group='group', annotation_cols=['sample'], components=2, perplexity=40, n_iter=1000, init='pca', dropna=True):
    """
    Performs t-distributed Stochastic Neighbor Embedding analysis. For more information visit https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list drop_cols: column labels to be dropped from the dataframe.
    :param str group: column label containing group identifiers.
    :param int components: dimension of the embedded space.
    :param list annotation_cols: list of columns to be added in the scatter plot annotation
    :param int perplexity: related to the number of nearest neighbors that is used in other manifold learning algorithms. Consider selecting a value between 5 and 50.
    :param int n_iter: maximum number of iterations for the optimization (at least 250).
    :param str init: initialization of embedding ('random', 'pca' or numpy array of shape n_samples x n_components).
    :param bool dropna: if True removes all columns with any missing values.
    :return: Two dictionaries: 1) pandas dataframe with embedding vectors, 2) xaxis and yaxis titles for plotly.

    Example::

        result = run_tsne(data, drop_cols=['sample', 'subject'], group='group', components=2, perplexity=40, n_iter=1000, init='pca', dropna=True)
    """
    result = {}
    args = {}
    df = data.copy()
    if len(set(drop_cols).intersection(df.columns)) == len(drop_cols):
        df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    if dropna:
        df = df.dropna(axis=1)
    df = df.select_dtypes(['number'])
    X = df.values
    y = df.index
    annotations = pd.DataFrame()
    if annotation_cols is not None:
        if len(list(set(annotation_cols).intersection(data.columns))) > 0:
            annotations = data[annotation_cols]
    if X.size > 0:
        tsne = TSNE(n_components=components, verbose=0, perplexity=perplexity, n_iter=n_iter, init=init)
        X = tsne.fit_transform(X)
        args = {"x_title":"C1","y_title":"C2"}
        if components == 2:
            resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
            resultDf = resultDf.reset_index()
            resultDf.columns = ["group", "x", "y"]
        if components > 2:
            args.update({"z_title":"C3"})
            resultDf = pd.DataFrame(X, index = y)
            resultDf = resultDf.reset_index()
            cols = []
            if len(components)>4:
                cols = resultDf.columns[4:]
            resultDf.columns = ["group", "x", "y", "z"] + cols
        resultDf = resultDf.join(annotations)
        result['tsne'] = resultDf
    return result, args


def run_umap(data, drop_cols=['sample', 'subject'], group='group', annotation_cols=['sample'], n_neighbors=10, min_dist=0.3, metric='cosine', dropna=True):
    """
    Performs Uniform Manifold Approximation and Projection. For more information vist https://umap-learn.readthedocs.io.

    :param data: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list drop_cols: column labels to be dropped from the dataframe.
    :param str group: column label containing group identifiers.
    :param list annotation_cols: list of columns to be added in the scatter plot annotation
    :param int n_neighbors: number of neighboring points used in local approximations of manifold structure.
    :param float min_dist: controls how tightly the embedding is allowed compress points together.
    :param str metric: metric used to measure distance in the input space.
    :param bool dropna: if True removes all columns with any missing values.
    :return: Two dictionaries: 1) pandas dataframe with embedding of the training data in low-dimensional space, 2) xaxis and yaxis titles for plotly.

    Example::

        result = run_umap(data, drop_cols=['sample', 'subject'], group='group', n_neighbors=10, min_dist=0.3, metric='cosine', dropna=True)
    """
    np.random.seed(1145536)
    result = {}
    args = {}
    df = data.copy()
    if len(set(drop_cols).intersection(df.columns)) == len(drop_cols):
        df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    if dropna:
        df = df.dropna(axis=1)
    df = df.select_dtypes(['number'])
    X = df.values
    y = df.index

    annotations = pd.DataFrame()
    if annotation_cols is not None:
        if len(list(set(annotation_cols).intersection(data.columns))) > 0:
            annotations = data[annotation_cols]

    if X.size:
        X = umap.UMAP(n_neighbors=10, min_dist=0.3, metric= metric).fit_transform(X)
        args = {"x_title":"C1","y_title":"C2"}
        resultDf = pd.DataFrame(X, index = y)
        resultDf = resultDf.reset_index()
        cols = []
        if len(resultDf.columns)>3:
                cols = resultDf.columns[3:]
        resultDf.columns = ["group", "x", "y"] + cols
        resultDf = resultDf.join(annotations)
        result['umap'] = resultDf

    return result, args


def calculate_correlations(x, y, method='pearson'):
    """
    Calculates a Spearman (nonparametric) or a Pearson (parametric) correlation coefficient and p-value to test for non-correlation.

    :param ndarray x: array 1
    :param ndarray y: array 2
    :param str method: chooses which kind of correlation method to run
    :return: Tuple with two floats, correlation coefficient and two-tailed p-value.

    Example::

        result = calculate_correlations(x, y, method='pearson')
    """
    if method == "pearson":
        coefficient, pvalue = stats.pearsonr(x, y)
    elif method == "spearman":
        coefficient, pvalue = stats.spearmanr(x, y)

    return (coefficient, pvalue)

def apply_pvalue_correction(pvalues, alpha=0.05, method='bonferroni'):
    """
    Performs p-value correction using the specified method. For more information visit https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html.

    :param ndarray pvalues: et of p-values of the individual tests.
    :param float alpha: error rate.
    :param str method: method of p-value correction:
        - bonferroni : one-step correction
        - sidak : one-step correction
        - holm-sidak : step down method using Sidak adjustments
        - holm : step-down method using Bonferroni adjustments
        - simes-hochberg : step-up method (independent)
        - hommel : closed method based on Simes tests (non-negative)
        - fdr_bh : Benjamini/Hochberg (non-negative)
        - fdr_by : Benjamini/Yekutieli (negative)
        - fdr_tsbh : two stage fdr correction (non-negative)
        - fdr_tsbky : two stage fdr correction (non-negative)
    :return: Tuple with two arrays, boolen for rejecting H0 hypothesis and float for adjusted p-value.

    Exmaple::

        result = apply_pvalue_correction(pvalues, alpha=0.05, method='bonferroni')
    """
    p = np.array(pvalues)
    mask = np.isfinite(p)
    pval_corrected = np.full(p.shape, np.nan)
    pval_corrected[mask] = multitest.multipletests(p[mask], alpha, method)[1]
    rejected = [p <= alpha for p in pval_corrected]

    return (rejected, pval_corrected.tolist())


def apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep'):
    """
    Performs p-value correction for false discovery rate. For more information visit https://www.statsmodels.org/devel/generated/statsmodels.stats.multitest.fdrcorrection.html.

    :param ndarray pvalues: et of p-values of the individual tests.
    :param float alpha: error rate.
    :param str method: method of p-value correction ('indep', 'negcorr').
    :return: Tuple with two arrays, boolen for rejecting H0 hypothesis and float for adjusted p-value.

    Exmaple::

        result = apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep')
    """
    rejected, padj = multitest.fdrcorrection(pvalues, alpha, method)

    return (rejected, padj)


def apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh'):
    """
    Iterated two stage linear step-up procedure with estimation of number of true hypotheses. For more information visit https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection_twostage.html.

    :param ndarray pvalues: et of p-values of the individual tests.
    :param float alpha: error rate.
    :param str method: method of p-value correction ('bky', 'bh').
    :return: Tuple with two arrays, boolen for rejecting H0 hypothesis and float for adjusted p-value.

    Exmaple::

        result = apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh')
    """
    rejected, padj, num_hyp, alpha_stages = multitest.fdrcorrection_twostage(pvalues, alpha, method)

    return (rejected, padj)


def apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group, alpha=0.05, permutations=50):
    """
    This function applies multiple hypothesis testing correction using a permutation-based false discovery rate approach.

    :param df: pandas dataframe with samples as rows and features as columns.
    :param oberved_pvalues: pandas Series with p-values calculated on the originally measured data.
    :param str group: name of the column containing group identifiers.
    :param float alpha: error rate. Values velow alpha are considered significant.
    :param int permutations: number of permutations to be applied.
    :return: Pandas dataframe with adjusted p-values and rejected columns.

    Example::

        result = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group='group', alpha=0.05, permutations=50)
    """
    i = permutations
    df_index = df.index.values
    df_columns = df.columns.values
    seen = [str([df_index]+df_columns.tolist())]
    rand_pvalues = []
    while i>0:
        df_index = shuffle(df_index)
        df_columns = shuffle(df_columns)
        df_random = df.reset_index(drop=True)
        df_random.index = df_index
        df_random.index.name = group
        df_random.columns = df_columns
        rand_index = str([df_random.index]+df_columns.tolist())
        if rand_index not in seen:
            seen.append(rand_index)
            df_random = df_random.reset_index()
            for col in df_random.columns.drop(group):
                rand_pvalues.append(calculate_anova(df_random, column=col, group=group)[-1])
            i -= 1
    rand_pvalues = np.array(rand_pvalues)

    qvalues = []
    for i, row in observed_pvalues.to_frame():
        qvalues.append(get_counts_permutation_fdr(row['pvalue'], rand_pvalues, samr_df['pvalue'], permutations, alpha)+(i,))

    qvalues = pd.DataFrame(qvalues, columns=['padj', 'rejected', 'identifier']).set_index('identifier')

    return qvalues


def get_counts_permutation_fdr(value, random, observed, n, alpha):
    """
    Calculates local FDR values (q-values) by computing the fraction of accepted hits from the permuted data over accepted hits from the measured data normalized by the total number of permutations.

    :param float value: computed p-value on measured data for a feature.
    :param ndarray random: p-values computed on the permuted data.
    :param observed: pandas Series with p-values calculated on the originally measured data.
    :param int n: number of permutations to be applied.
    :param float alpha: error rate. Values velow alpha are considered significant.
    :return: Tuple with q-value and boolean for H0 rejected.

    Example::

        result = get_counts_permutation_fdr(value, random, observed, n=250, alpha=0.05)
    """
    a = random[random <= value].shape[0] + 0.0000000000001 #Offset in case of a = 0.0
    b = (observed <= value).sum()
    qvalue = (a/b/float(n))

    return (qvalue, qvalue <= alpha)


def convertToEdgeList(data, cols):
    """
    This function converts a pandas dataframe to an edge list where index becomes the source nodes and columns the target nodes.

    :param data: pandas dataframe.
    :param list cols: names for dataframe columns.
    :return: Pandas dataframe with columns cols.
    """
    data.index.name = None
    edge_list = data.stack().reset_index()
    edge_list.columns = cols

    return edge_list


def run_correlation(df, alpha=0.05, subject='subject', group='group', method='pearson', correction='fdr_bh'):
    """
    This function calculates pairwise correlations for columns in dataframe, and returns it in the shape of a edge list with 'weight' as correlation score, and the ajusted p-values.

    :param df: pandas dataframe with samples as rows and features as columns.
    :param str subject: name of column containing subject identifiers.
    :param str group: name of column containing group identifiers.
    :param str method: method to use for correlation calculation ('pearson', 'spearman').
    :param floar alpha: error rate. Values velow alpha are considered significant.
    :param string correction: type of correction see apply_pvalue_correction for methods
    :return: Pandas dataframe with columns: 'node1', 'node2', 'weight', 'padj' and 'rejected'.

    Example::

        result = run_correlation(df, alpha=0.05, subject='subject', group='group', method='pearson', correction='fdr_bh')
    """
    correlation = pd.DataFrame()
    #ToDo
    #The Repeated measurements correlation calculation is too time consuming so it only runs if
    #the number of features is less than 200
    if check_is_paired(df, subject, group):
        if len(df[subject].unique()) > 2:
            if len(df.columns) < 200:
                correlation = run_rm_correlation(df, alpha=alpha, subject=subject, correction=correction)
    else:
        df = df.dropna(axis=1)._get_numeric_data()
        if not df.empty:
            r, p = run_efficient_correlation(df, method=method)
            rdf = pd.DataFrame(r, index=df.columns, columns=df.columns)
            pdf = pd.DataFrame(p, index=df.columns, columns=df.columns)
            correlation = convertToEdgeList(rdf, ["node1", "node2", "weight"])
            pvalues = convertToEdgeList(pdf, ["node1", "node2", "pvalue"])
            correlation = pd.merge(correlation, pvalues, on=['node1','node2'])

            rejected, padj = apply_pvalue_correction(correlation["pvalue"].tolist(), alpha=alpha, method=correction)
            correlation["padj"] = padj
            correlation["rejected"] = rejected
            correlation = correlation[correlation.rejected]
            correlation["pvalue"] = correlation["pvalue"].apply(lambda x: str(round(x, 5)))
            correlation["padj"] = correlation["padj"].apply(lambda x: str(round(x, 5)))
    
    return correlation


def run_multi_correlation(df_dict, alpha=0.05, subject='subject', on=['subject', 'biological_sample'] , group='group', method='pearson', correction='fdr_bh'):
    """
    This function merges all input dataframes and calculates pairwise correlations for all columns.

    :param dict df_dict: dictionary of pandas dataframes with samples as rows and features as columns.
    :param str subject: name of the column containing subject identifiers.
    :param str group: name of the column containing group identifiers.
    :param list on: column names to join dataframes on (must be found in all dataframes).
    :param str method: method to use for correlation calculation ('pearson', 'spearman').
    :param float alpha: error rate. Values velow alpha are considered significant.
    :param string correction: type of correction see apply_pvalue_correction for methods
    :return: Pandas dataframe with columns: 'node1', 'node2', 'weight', 'padj' and 'rejected'.

    Example::

        result = run_multi_correlation(df_dict, alpha=0.05, subject='subject', on=['subject', 'biological_sample'] , group='group', method='pearson', correction='fdr_bh')
    """
    multidf = pd.DataFrame()
    correlation = None
    for dtype in df_dict:
        if multidf.empty:
            if isinstance(df_dict[dtype], pd.DataFrame):
                multidf = df_dict[dtype]
        else:
            if isinstance(df_dict[dtype], pd.DataFrame):
                multidf = pd.merge(multidf, df_dict[dtype], how='inner', on=on)
    if not multidf.empty:
        correlation = run_correlation(multidf, alpha=alpha, subject=subject, group=group, method=method, correction=correction)

    return correlation


def calculate_rm_correlation(df, x, y, subject):
    """
    Computes correlation and p-values between two columns a and b in df.

    :param df: pandas dataframe with subjects as rows and two features and columns.
    :param str x: feature a name.
    :param str y: feature b name.
    :param subject: column name containing the covariate variable.
    :return: Tuple with values for: feature a, feature b, correlation, p-value and degrees of freedom.

    Example::

        result = calculate_rm_correlation(df, x='feature a', y='feature b', subject='subject')
    """
    # ANCOVA model
    cols = ["col0","col1", subject]
    a = "col0"
    b = "col1"
    df.columns = cols

    formula = b + ' ~ ' + 'C(' + subject + ') + ' + a
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=3)
    # Extract the sign of the correlation and dof
    sign = np.sign(model.params[a])
    dof = int(table.loc['Residual', 'df'])
    # Extract correlation coefficient from sum of squares
    ssfactor = table.loc[a, 'sum_sq']
    sserror = table.loc['Residual', 'sum_sq']
    rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    # Extract p-value
    pvalue = table.loc[a, 'PR(>F)']
    pvalue *= 0.5

    #r, dof, pvalue, ci, power = pg.rm_corr(data=df, x=x, y=y, subject=subject)

    return (x, y, rm, pvalue, dof)


def run_rm_correlation(df, alpha=0.05, subject='subject', correction='fdr_bh'):
    """
    Computes pairwise repeated measurements correlations for all columns in dataframe, and returns results as an edge list with 'weight' as correlation score, p-values, degrees of freedom and ajusted p-values.

    :param df: pandas dataframe with samples as rows and features as columns.
    :param str subject: name of column containing subject identifiers.
    :param float alpha: error rate. Values velow alpha are considered significant.
    :param string correction: type of correction type see apply_pvalue_correction for methods
    :return: Pandas dataframe with columns: 'node1', 'node2', 'weight', 'pvalue', 'dof', 'padj' and 'rejected'.

    Example::

        result = run_rm_correlation(df, alpha=0.05, subject='subject', correction='fdr_bh')
    """
    rows = []
    if not df.empty:
        df = df.set_index(subject)._get_numeric_data().dropna(axis=1)
        df.columns = df.columns.astype(str)
        combinations = itertools.combinations(df.columns, 2)
        df = df.reset_index()
        for x, y in combinations:
            row = [x, y]
            subset = df[[x,y, subject]]
            row.extend(pg.rm_corr(subset, x, y, subject).values.tolist()[0])
            rows.append(row)

        correlation = pd.DataFrame(rows, columns=["node1", "node2", "weight", "dof", "pvalue", "CI95%", "power"])
        rejected, padj = apply_pvalue_correction(correlation["pvalue"].tolist(), alpha=alpha, method=correction)
        correlation["padj"] = padj
        correlation["rejected"] = rejected
        correlation = correlation[correlation.rejected]
        correlation["padj"] = correlation["padj"].apply(lambda x: str(round(x, 5)))

    return correlation


def run_efficient_correlation(data, method='pearson'):
    """
    Calculates pairwise correlations and returns lower triangle of the matrix with correlation values and p-values.

    :param data: pandas dataframe with samples as index and features as columns (numeric data only).
    :param str method: method to use for correlation calculation ('pearson', 'spearman').
    :return: Two numpy arrays: correlation and p-values.

    Example::

        result = run_efficient_correlation(data, method='pearson')
    """
    matrix = data.values
    if method == 'pearson':
        r = np.corrcoef(matrix, rowvar=False)
    elif method == 'spearman':
        r, p = stats.spearmanr(matrix, axis=0)

    diagonal = np.triu_indices(r.shape[0], 1)
    rf = r[diagonal]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = pf
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

    r[diagonal] = np.nan
    p[diagonal] = np.nan

    return r, p


def calculate_ttest_samr(df, labels, n=2, s0=0, paired=False):
    """
    Calculates modified T-test using 'samr' R package.

    :param df: pandas dataframe with group as columns and protein identifier as rows
    :param list abels: integers reflecting the group each sample belongs to (e.g. group1 = 1, group2 = 2)
    :param int n: number of samples
    :param float s0: exchangeability factor for denominator of test statistic
    :param bool paired: True if samples are paired
    :return: Pandas dataframe with columns 'identifier', 'group1', 'group2', 'mean(group1)', 'mean(group1)', 'log2FC', 'FC', 't-statistics', 'p-value'.

    Example::

        result = calculate_ttest_samr(df, labels, n=2, s0=0.1, paired=False)
    """
    if not r_installation:
        raise Exception("No valid installation of R")

    result = pd.DataFrame(columns=['identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'FC', 't-statistics', 'pvalue'])
    conditions = df.columns.unique()
    mean1 = df[conditions[0]].mean(axis=1)
    mean2 = df[conditions[1]].mean(axis=1)

    if paired:
        counts = {}
        labels = []
        for col in df.columns:
            cur_count = counts.get(col, 0)
            labels.append([(cur_count+1)*-1 if col == conditions[0] else (cur_count+1)][0])
            counts[col] = cur_count + 1

        ttest_res = samr.paired_ttest_func(df.values, base.unlist(labels), s0=s0)
    else:
        ttest_res = samr.ttest_func(df.values, base.unlist(labels), s0=s0)

    pvalues = [2 * stats_r.pt(-base.abs(i), df=n-1)[0] for i in ttest_res[0]]

    result = pd.DataFrame([df.index, mean1, mean2, ttest_res[1], ttest_res[0], pvalues]).T
    result.columns = ['identifier', 'mean(group1)', 'mean(group2)', 'log2FC', 't-statistics', 'pvalue']
    result['group1'] = conditions[0]
    result['group2'] = conditions[1]
    cols = ['identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'FC', 't-statistics', 'pvalue']
    result['FC'] = result['log2FC'].apply(lambda x: np.power(2,x))

    result = result[cols]

    return result


def calculate_ttest(df, condition1, condition2, paired=False, is_logged=True, non_par=False, tail='two-sided',  correction='auto', r=0.707):
    """
    Calculates the t-test for the means of independent samples belonging to two different groups. For more information visit https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html.

    :param df: pandas dataframe with groups and subjects as rows and protein identifier as column.
    :param str condition1: identifier of first group.
    :param str condition2: ientifier of second group.
    :param bool is_logged: data is logged transformed
    :param bool non_par: if True, normality and variance equality assumptions are checked and non-parametric test Mann Whitney U test if not passed
    :return: Tuple with t-statistics, two-tailed p-value, mean of first group, mean of second group and logfc.

    Example::

        result = calculate_ttest(df, 'group1', 'group2')
    """
    t = None
    pvalue = np.nan
    group1 = df[[condition1]].values
    group2 = df[[condition2]].values

    mean1 = group1.mean()
    std1 = group1.std()
    mean2 = group2.mean()
    std2 = group2.std()
    if is_logged:
        fc = mean1 - mean2
    else:
        fc = mean1/mean2

    test = 't-Test'
    if not non_par:
        result = pg.ttest(group1, group2, paired, tail, correction, r)
    else:
        test = 'Mann Whitney'
        result = pg.mwu(group1, group2, tail=tail)

    if 'T' in result.columns:
        t = result['T'].values[0]
    elif 'U-val' in result.columns:
        t = result['U-val'].values[0]
    if 'p-val' in result.columns:
        pvalue = result['p-val'].values[0]

    return (t, pvalue, mean1, mean2, std1, std2, fc, test)


def calculate_THSD(df, column, group='group', alpha=0.05, is_logged=True):
    """
    Pairwise Tukey-HSD posthoc test using pingouin stats. For more information visit https://pingouin-stats.org/generated/pingouin.pairwise_tukey.html

    :param df: pandas dataframe with group and protein identifier as columns
    :param str column: column containing the protein identifier
    :param str group: column label containing the between factor
    :param float alpha: significance level
    :return: Pandas dataframe.

    Example::

        result = calculate_THSD(df, column='HBG2~P69892', group='group', alpha=0.05)
    """
    posthoc = None
    posthoc = pg.pairwise_tukey(data=df, dv=column, between=group, alpha=alpha, tail='two-sided')
    posthoc.columns = ['group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'std_error', 'tail', 't-statistics', 'posthoc pvalue', 'effsize']
    posthoc['efftype'] = 'hedges'
    posthoc = complement_posthoc(posthoc, identifier=column, is_logged=is_logged)

    return posthoc


def calculate_pairwise_ttest(df, column, subject='subject', group='group', correction='none', is_logged=True):
    """
    Performs pairwise t-test using pingouin, as a posthoc test, and calculates fold-changes. For more information visit https://pingouin-stats.org/generated/pingouin.pairwise_ttests.html.

    :param df: pandas dataframe with subject and group as rows and protein identifier as column.
    :param str column: column label containing the dependant variable
    :param str subject: column label containing subject identifiers
    :param str group: column label containing the between factor
    :param str correction: method used for testing and adjustment of p-values.
    :return: Pandas dataframe with means, standard deviations, test-statistics, degrees of freedom and effect size columns.

    Example::

        result = calculate_pairwise_ttest(df, 'protein a', subject='subject', group='group', correction='none')
    """

    posthoc_columns = ['Contrast', 'group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'posthoc Paired', 'posthoc Parametric', 'posthoc T-Statistics', 'posthoc dof', 'posthoc tail', 'posthoc pvalue', 'posthoc BF10', 'posthoc effsize']
    valid_cols = ['group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'posthoc Paired', 'posthoc Parametric', 'posthoc T-Statistics', 'posthoc dof', 'posthoc tail', 'posthoc pvalue', 'posthoc BF10', 'posthoc effsize']
    posthoc = df.pairwise_ttests(dv=column, between=group, subject=subject, effsize='hedges', return_desc=True, padjust=correction)
    posthoc.columns =  posthoc_columns
    posthoc = posthoc[valid_cols]
    posthoc = complement_posthoc(posthoc, column, is_logged)
    posthoc['efftype'] = 'hedges'

    return posthoc


def complement_posthoc(posthoc, identifier, is_logged):
    """
    Calculates fold-changes after posthoc test.

    :param posthoc: pandas dataframe from posthoc test. Should have at least columns 'mean(group1)' and 'mean(group2)'.
    :param str identifier: feature identifier.
    :return: Pandas dataframe with additional columns 'identifier', 'log2FC' and 'FC'.
    """
    posthoc['identifier'] = identifier
    if is_logged:
        posthoc['log2FC'] = posthoc['mean(group1)'] - posthoc['mean(group2)']
        posthoc['FC'] = posthoc['log2FC'].apply(lambda x: np.power(2,x))
    else:
        posthoc['FC'] = posthoc['mean(group1)']/posthoc['mean(group2)']

    return posthoc


def calculate_anova_samr(df, labels, s0=0):
    """
    Calculates modified one-way ANOVA using 'samr' R package.

    :param df: pandas dataframe with group as columns and protein identifier as rows
    :param list labels: integers reflecting the group each sample belongs to (e.g. group1 = 1, group2 = 2, group3 = 3)
    :param float s0: exchangeability factor for denominator of test statistic
    :return: Pandas dataframe with protein identifiers and F-statistics.

    Example::

        result = calculate_anova_samr(df, labels, s0=0.1)
    """
    if not r_installation:
        raise Exception("No valid installation of R")

    aov_res = wgcna.multiclass_func(df.values, base.unlist(labels), s0=s0)

    result = pd.DataFrame([df.index, aov_res[0]]).T
    result.columns = ['identifier', 'F-statistics']

    return result


def calculate_anova(df, column, group='group'):
    """
    Calculates one-way ANOVA using pingouin.

    :param df: pandas dataframe with group as rows and protein identifier as column
    :param str column: name of the column in df to run ANOVA on
    :param str group: column with group identifiers
    :return: Tuple with t-statistics and p-value.
    """
    aov_result = pg.anova(data=df, dv=column, between=group)
    df1, df2, t, pvalue = aov_result[['ddof1', 'ddof2', 'F', 'p-unc']].values.tolist()[0]

    return (column, df1, df2, t, pvalue)

def calculate_ancova(data, column, group='group', covariates=[]):
    """
    Calculates one-way ANCOVA using pingouin.

    :param df: pandas dataframe with group as rows and protein identifier as column
    :param str column: name of the column in df to run ANOVA on
    :param str group: column with group identifiers
    :param list covariates: list of covariates (columns in df)
    :return: Tuple with column, F-statistics and p-value.
    """
    ancova_result = pg.ancova(data=data, dv=column, between=group, covar=covariates)
    t, df, pvalue = ancova_result.loc[ancova_result['Source'] == group, ['F', 'DF', 'p-unc']].values.tolist().pop()

    return (column, df, df, t, pvalue)

def calculate_repeated_measures_anova(df, column, subject='subject', within='group'):
    """
    One-way and two-way repeated measures ANOVA using pingouin stats.

    :param df: pandas dataframe with samples as rows and protein identifier as column. Data must be in long-format for two-way repeated measures.
    :param str column: column label containing the dependant variable
    :param str subject: column label containing subject identifiers
    :param str within: column label containing the within factor
    :return: Tuple with protein identifier, t-statistics and p-value.

    Example::

        result = calculate_repeated_measures_anova(df, 'protein a', subject='subject', within='group')
    """
    df1 = np.nan
    df2 = np.nan
    t = np.nan
    pvalue = np.nan
    try:
        aov_result = pg.rm_anova(data=df, dv=column, within=within, subject=subject, detailed=True, correction=True)
        t, pvalue = aov_result.loc[0, ['F', 'p-unc']].values.tolist()
        df1, df2 = aov_result['DF']
    except Exception as e:
        print("Repeated measurements Anova for column: {} could not be calculated. Error {}".format(column, e))

    return (column, df1, df2, t, pvalue)

def calculate_mixed_anova(df, column, subject='subject', within='group', between='group2'):
    """
    One-way and two-way repeated measures ANOVA using pingouin stats.

    :param df: pandas dataframe with samples as rows and protein identifier as column. Data must be in long-format for two-way repeated measures.
    :param str column: column label containing the dependant variable
    :param str subject: column label containing subject identifiers
    :param str within: column label containing the within factor
    :param str within: column label containing the between factor
    :return: Tuple with protein identifier, t-statistics and p-value.

    Example::

        result = calculate_mixed_anova(df, 'protein a', subject='subject', within='group', between='group2')
    """
    df1 = np.nan
    df2 = np.nan
    t = np.nan
    pvalue = np.nan
    try:
        aov_result = pg.mixed_anova(data=df, dv=column, within=within, between=between, subject=subject, correction=True)
        aov_result['identifier'] = column
    except Exception as e:
        print("Mixed Anova for column: {} could not be calculated. Error {}".format(column, e))

    return aov_result[['identifier', 'DF1', 'DF2', 'F', 'p-unc', 'Source']]


def get_max_permutations(df, group='group'):
    """
    Get maximum number of permutations according to number of samples.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns
    :param str group: column with group identifiers
    :return: Maximum number of permutations.
    :rtype: int
    """
    num_groups = len(list(df.index))
    num_per_group = df.groupby(group).size().tolist()
    max_perm = factorial(num_groups)/np.prod(factorial(np.array(num_per_group)))

    return max_perm


def check_is_paired(df, subject, group):
    """
    Check if samples are paired.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers
    :return: True if paired samples.
    :rtype: bool
    """
    is_pair = False
    if subject is not None:
        count_subject_groups = df.groupby(subject)[group].count()
        is_pair = (count_subject_groups > 1).all()

    return is_pair


def run_anova(df, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=0, correction='fdr_bh', is_logged=True, non_par=False):
    """
    Performs statistical test for each protein in a dataset.
    Checks what type of data is the input (paired, unpaired or repeated measurements) and performs posthoc tests for multiclass data.
    Multiple hypothesis correction uses permutation-based if permutations>0 and Benjamini/Hochberg if permutations=0.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers
    :param list drop_cols: column labels to be dropped from the dataframe
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates.
    :param bool non_par: if True, normality and variance equality assumptions are checked and non-parametric test Mann Whitney U test if not passed
    :return: Pandas dataframe with columns 'identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'Log2FC', 'std_error', 'tail', 't-statistics', 'posthoc pvalue', 'effsize', 'efftype', 'FC', 'rejected', 'F-statistics', 'p-value', 'correction', '-log10 p-value', and 'method'.

    Example::

        result = run_anova(df, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=50)
    """
    res = pd.DataFrame()
    if subject is not None and check_is_paired(df, subject, group):
        groups = df[group].unique()
        drop_cols = [d for d in drop_cols if d != subject]
        if len(groups) == 2:
            res = run_ttest(df, groups[0], groups[1], alpha = alpha, drop_cols=drop_cols, subject=subject, group=group, paired=True, correction=correction, permutations=permutations, is_logged=is_logged, non_par=non_par)
        elif len(groups) > 2:
            res = run_repeated_measurements_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, within=group, permutations=0, is_logged=is_logged)
    elif len(df[group].unique()) == 2:
        groups = df[group].unique()
        drop_cols = [d for d in drop_cols if d != subject]
        res = run_ttest(df, groups[0], groups[1], alpha = alpha, drop_cols=drop_cols, subject=subject, group=group, paired=False, correction=correction, permutations=permutations, is_logged=is_logged, non_par=non_par)
    elif len(df[group].unique()) > 2:
        df = df.drop(drop_cols, axis=1)
        aov_results = []
        pairwise_results = []
        for col in df.columns.drop(group).tolist():
            aov = calculate_anova(df[[group, col]], column=col, group=group)
            aov_results.append(aov)
            pairwise_result = calculate_pairwise_ttest(df[[group, col]], column=col, subject=subject, group=group, is_logged=is_logged)
            pairwise_cols = pairwise_result.columns
            pairwise_results.extend(pairwise_result.values.tolist())
        df = df.set_index([group])
        res = format_anova_table(df, aov_results, pairwise_results, pairwise_cols, group, permutations, alpha, correction)
        res['Method'] = 'One-way anova'
        res = correct_pairwise_ttest(res, alpha, correction)

    return res

def run_ancova(df, covariates, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=0, correction='fdr_bh', is_logged=True, non_par=False):
    """
    Performs statistical test for each protein in a dataset.
    Checks what type of data is the input (paired, unpaired or repeated measurements) and performs posthoc tests for multiclass data.
    Multiple hypothesis correction uses permutation-based if permutations>0 and Benjamini/Hochberg if permutations=0.

    :param df: pandas dataframe with samples as rows and protein identifiers and covariates as columns (with additional columns 'group', 'sample' and 'subject').
    :param list covariates: list of covariates to include in the model (column in df)
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers
    :param list drop_cols: column labels to be dropped from the dataframe
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates.
    :param bool non_par: if True, normality and variance equality assumptions are checked and non-parametric test Mann Whitney U test if not passed
    :return: Pandas dataframe with columns 'identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'Log2FC', 'std_error', 'tail', 't-statistics', 'posthoc pvalue', 'effsize', 'efftype', 'FC', 'rejected', 'F-statistics', 'p-value', 'correction', '-log10 p-value', and 'method'.

    Example::

        result = run_ancova(df, covariates=['age'], alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=50)
    """
    df = df.drop(drop_cols, axis=1)
    for cova in covariates:
        if df[cova].dtype != np.number:
            df[cova] = pd.Categorical(df[cova])
            df[cova] = df[cova].cat.codes

    pairwise_results = []
    ancova_result = []
    for col in df.columns.tolist():
        if col not in covariates and col != group:
            ancova = calculate_ancova(df[[group, col]+covariates], col, group=group, covariates=covariates)
            ancova_result.append(ancova)
            pairwise_result = pairwise_ttest_with_covariates(df, column=col, group=group, covariates=covariates, is_logged=is_logged)
            pairwise_cols = pairwise_result.columns
            pairwise_results.extend(pairwise_result.values.tolist())
    df = df.set_index([group])
    res = format_anova_table(df, ancova_result, pairwise_results, pairwise_cols, group, permutations, alpha, correction)
    res['Method'] = 'One-way ancova'
    res = correct_pairwise_ttest(res, alpha, correction)

    return res


def pairwise_ttest_with_covariates(df, column, group, covariates, is_logged):
    formula = "Q('%s') ~ C(Q('%s'))" % (column, group)
    for c in covariates:
        formula += " + Q('%s')" % (c)
    model = ols(formula, data=df).fit()
    pw = model.t_test_pairwise("C(Q('%s'))" % (group)).result_frame
    pw = pw.reset_index()
    groups = "|".join([re.escape(s) for s in df[group].unique().tolist()])
    regex = r"({})\-({})".format(groups, groups)
    pw['group1'] = pw['index'].apply(lambda x: re.search(regex, x).group(2))
    pw['group2'] = pw['index'].apply(lambda x: re.search(regex, x).group(1))

    means = df.groupby(group)[column].mean().to_dict()
    stds = df.groupby(group)[column].std().to_dict()
    pw['mean(group1)'] = [means[g] for g in pw['group1'].tolist()]
    pw['mean(group2)'] = [means[g] for g in pw['group2'].tolist()]
    pw['std(group1)'] = [stds[g] for g in pw['group1'].tolist()]
    pw['std(group2)'] = [stds[g] for g in pw['group2'].tolist()]
    pw = pw.drop(['pvalue-hs', 'reject-hs'], axis=1)
    pw = pw.rename(columns={'t': 'posthoc T-Statistics', 'P>|t|': 'posthoc pvalue'} )

    pw = pw[['group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'posthoc T-Statistics', 'posthoc pvalue', 'coef', 'std err', 'Conf. Int. Low', 'Conf. Int. Upp.']]
    pw = complement_posthoc(pw, column, is_logged)

    return pw


def correct_pairwise_ttest(df, alpha, correction='fdr_bh'):
    posthoc_df = pd.DataFrame()
    if 'group1' in df and 'group2' in df and "posthoc pvalue" in df:
        for comparison in df.groupby(["group1","group2"]).groups:
            index = df.groupby(["group1","group2"]).groups.get(comparison)
            posthoc_pvalues = df.loc[index, "posthoc pvalue"].tolist()
            rejected, posthoc_padj = apply_pvalue_correction(posthoc_pvalues, alpha=alpha, method=correction)
            if posthoc_df.empty:
                posthoc_df = pd.DataFrame({"index":index, "posthoc padj":posthoc_padj})
            else:
                posthoc_df = posthoc_df.append(pd.DataFrame({"index":index, "posthoc padj":posthoc_padj}))
        posthoc_df = posthoc_df.set_index("index")
        df = df.join(posthoc_df)

    return df

def run_repeated_measurements_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', within='group', permutations=50, correction='fdr_bh', is_logged=True):
    """
    Performs repeated measurements anova and pairwise posthoc tests for each protein in dataframe.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param srt within: column with within factor identifiers
    :param list drop_cols: column labels to be dropped from the dataframe
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates
    :return: Pandas dataframe

    Example::

        result = run_repeated_measurements_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', within='group', permutations=50)
    """
    df = df.drop(drop_cols, axis=1).dropna(axis=1)
    aov_results = []
    pairwise_results = []
    index = [within, subject]
    for col in df.columns.drop(index).tolist():
        cols = index + [col]
        aov = calculate_repeated_measures_anova(df[cols], column=col, subject=subject, within=within)
        aov_results.append(aov)
        pairwise_result = calculate_pairwise_ttest(df[[within, subject, col]], subject=subject, column=col, group=within, is_logged=is_logged)
        pairwise_cols = pairwise_result.columns
        pairwise_results.extend(pairwise_result.values.tolist())

    df = df.set_index([subject, within])
    res = format_anova_table(df, aov_results, pairwise_results, pairwise_cols, within, permutations, alpha, correction)
    res['Method'] = 'Repeated measurements anova'
    res = correct_pairwise_ttest(res, alpha, correction=correction)

    return res


def run_mixed_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', within='group', between='group2', permutations=50, correction='fdr_bh', is_logged=True):
    """
    In statistics, a mixed-design analysis of variance model, also known as a split-plot ANOVA, is used to test 
    for differences between two or more independent groups whilst subjecting participants to repeated measures.
    Thus, in a mixed-design ANOVA model, one factor (a fixed effects factor) is a between-subjects variable and the other
    (a random effects factor) is a within-subjects variable. Thus, overall, the model is a type of mixed-effects model.
    [source:https://en.wikipedia.org/wiki/Mixed-design_analysis_of_variance]

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param srt within: column with within factor identifiers
    :param srt between: column with between factor identifiers
    :param list drop_cols: column labels to be dropped from the dataframe
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates
    :return: Pandas dataframe

    Example::

        result = run_mixed_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', within='group', between='group2', permutations=50)
    """
    df = df.drop(drop_cols, axis=1).dropna(axis=1)
    aov_results = []
    pairwise_results = []
    index = [within, subject, between]
    for col in df.columns.drop(index).tolist():
        cols = index + [col]
        aov = calculate_mixed_anova(df[cols], column=col, subject=subject, within=within, between=between)
        aov_results.append(aov)
        
    res = pd.concat(aov_results)
    res = res[res['Source'] == 'Interaction']
    res = res[['identifier', 'DF1', 'DF2', 'F', 'p-unc']]
    res.columns = ['identifier', 'dfk', 'dfn', 'F-statistics', 'pvalue']
    rejected, padj = apply_pvalue_correction(res['pvalue'].tolist(), alpha=alpha, method=correction)
    res['correction'] = 'FDR correction BH'
    res['padj'] = padj
    res['rejected'] = res['padj'] < alpha
    res['testing'] = 'Interaction'
    res['within'] = ','.join(df[within].unique().tolist())
    res['between'] = ','.join(df[between].unique().tolist())
    
    return res


def format_anova_table(df, aov_results, pairwise_results, pairwise_cols, group, permutations, alpha, correction):
    """
    Performs p-value correction (permutation-based and FDR) and converts pandas dataframe into final format.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param list[tuple] aov_results: list of tuples with anova results (one tuple per feature).
    :param list[dataframes] pairwise_results: list of pandas dataframes with posthoc tests results
    :param str group: column with group identifiers
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates
    :return: Pandas dataframe
    """
    columns = ['identifier', 'dfk', 'dfn', 'F-statistics', 'pvalue']
    scores = pd.DataFrame(aov_results, columns = columns)
    scores = scores.set_index('identifier')
    corrected = False
    #FDR correction
    if permutations > 0:
        max_perm = get_max_permutations(df, group=group)
        if max_perm>=10:
            if max_perm < permutations:
                permutations = max_perm
            observed_pvalues = scores.pvalue
            count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group=group, alpha=alpha, permutations=permutations)
            scores= scores.join(count)
            scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
            corrected = True

    if not corrected:
        rejected, padj = apply_pvalue_correction(scores['pvalue'].tolist(), alpha=alpha, method=correction)
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        corrected = True

    res = pd.DataFrame(pairwise_results, columns=pairwise_cols).set_index("identifier")
    if not res.empty:
        res = res.join(scores[['F-statistics', 'pvalue', 'padj']].astype('float'))
        res['correction'] = scores['correction']
    else:
        res = scores
        res["log2FC"] = np.nan

    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha

    if 'posthoc pvalue' in res.columns:
        res['-log10 pvalue'] = [- np.log10(x) for x in res['posthoc pvalue'].values]
    else:
        res['-log10 pvalue'] = [- np.log10(x) for x in res['pvalue'].values]

    return res


def run_ttest(df, condition1, condition2, alpha = 0.05, drop_cols=["sample"], subject='subject', group='group', paired=False, correction='fdr_bh', permutations=0, is_logged=True, non_par=False):
    """
    Runs t-test (paired/unpaired) for each protein in dataset and performs permutation-based (if permutations>0) or Benjamini/Hochberg (if permutations=0) multiple hypothesis correction.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str condition1: first of two conditions of the independent variable
    :param str condition2: second of two conditions of the independent variable
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers (independent variable)
    :param list drop_cols: column labels to be dropped from the dataframe
    :param bool paired: paired or unpaired samples
    :param str correction: method of pvalue correction see apply_pvalue_correction for methods
    :param float alpha: error rate for multiple hypothesis correction
    :param int permutations: number of permutations used to estimate false discovery rates.
    :param bool is_logged: data is log-transformed
    :param bool non_par: if True, normality and variance equality assumptions are checked and non-parametric test Mann Whitney U test if not passed
    :return: Pandas dataframe with columns 'identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'std(group1)', 'std(group2)', 'Log2FC', 'FC', 'rejected', 'T-statistics', 'p-value', 'correction', '-log10 p-value', and 'method'.

    Example::

        result = run_ttest(df, condition1='group1', condition2='group2', alpha = 0.05, drop_cols=['sample'], subject='subject', group='group', paired=False, correction='fdr_bh', permutations=50)
    """
    columns = ['T-statistics', 'pvalue', 'mean_group1', 'mean_group2', 'std(group1)', 'std(group2)', 'log2FC', 'test']
    df = df.set_index(group)
    df = df.drop(drop_cols, axis = 1)
    method = 'Unpaired t-test'
    if non_par:
        method = 'Unpaired t-Test and Mann-Whitney U test'

    if paired:
        df = df.reset_index().set_index([group, subject])
        method = 'Paired t-test'
    else:
        if subject is not None:
            df = df.drop([subject], axis = 1)

    scores = df.T.apply(func = calculate_ttest, axis=1, result_type='expand', args =(condition1, condition2, paired, is_logged, non_par))
    scores.columns = columns
    scores = scores.dropna(how="all")

    corrected = False
    #FDR correction
    if permutations > 0:
        max_perm = get_max_permutations(df, group=group)
        if max_perm>=10:
            if max_perm < permutations:
                permutations = max_perm
            observed_pvalues = scores.pvalue
            count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group=group, alpha=alpha, permutations=permutations)
            scores= scores.join(count)
            scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
            corrected = True

    if not corrected:
        rejected, padj = apply_pvalue_correction(scores["pvalue"].tolist(), alpha=alpha, method=correction)
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
        corrected = True

    scores['group1'] = condition1
    scores['group2'] = condition2
    if is_logged:
        scores['FC'] = scores['log2FC'].apply(lambda x: np.power(2,x))
    else:
        scores = scores.rename(columns={'log2FC':'FC'})

    scores['-log10 pvalue'] = [-np.log10(x) if x != 0 else -np.log10(alpha) for x in scores['pvalue'].values]
    scores['Method'] = method
    scores.index.name = 'identifier'
    scores = scores.reset_index()

    return scores

def define_samr_method(df, subject, group, drop_cols):
    """
    Method to identify the correct problem type to run with SAMR

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers
    :param str droop_cols: columns to be dropped
    :return: tuple with the method to be used (One Class, Two class paired, Two class unpaired or Multiclass) and the labels (conditions)

    Example::

        method, labels = define_samr_method(df, subject, group)
    """
    paired = check_is_paired(df, subject, group)
    groups = df[group].unique()
    df = df.set_index(group).drop(drop_cols, axis=1).T
    conditions = set(df.columns)
    d = {v:k+1 for k, v in enumerate(conditions)}
    labels = [d.get(item,item)  for item in df.columns]

    if len(groups) == 1:
        method = 'One class'
    elif len(groups) == 2:
        method = 'Two class unpaired'
    else:
        method = 'Multiclass'
    if subject is not None:
        if len(groups) == 2:
            if paired:
                method = 'Two class paired'
                labels = []
                counts = {}
                conditions = df.columns.unique()
                for col in df.columns:
                    cur_count = counts.get(col, 0)
                    labels.append([(cur_count+1)*-1 if col == conditions[0] else (cur_count+1)][0])
                    counts[col] = cur_count + 1

    return method, labels


def calculate_pvalue_from_tstats(tstat, dfn, dfk):
    """
    Calculate two-tailed p-values from T- or F-statistics.

    tstat: T/F distribution
    dfn: degrees of freedrom *n* (values) per protein (keys), i.e. number of obervations - number of groups (dict)
    dfk: degrees of freedrom *n* (values) per protein (keys), i.e. number of groups - 1 (dict)
    """
    pval = scipy.stats.t.sf(np.abs(tstat), dfn)*2

    return pval


def run_samr(df, subject='subject', group='group', drop_cols=['subject', 'sample'], alpha=0.05, s0='null', permutations=250, fc=0, is_logged=True, localfdr=False):
    """
    Python adaptation of the 'samr' R package for statistical tests with permutation-based correction and s0 parameter.
    For more information visit https://cran.r-project.org/web/packages/samr/samr.pdf.
    The method only runs if R is installed and permutations is higher than 0, otherwise ANOVA.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns (with additional columns 'group', 'sample' and 'subject').
    :param str subject: column with subject identifiers
    :param str group: column with group identifiers
    :param list drop_cols: columnlabels to be dropped from the dataframe
    :param float alpha: error rate for multiple hypothesis correction
    :param float s0: exchangeability factor for denominator of test statistic
    :param int permutations: number of permutations used to estimate false discovery rates. If number of permutations is equal to zero, the function will run anova with FDR Benjamini/Hochberg correction.
    :param float fc: minimum fold change to define practical significance (needed when computing delta table)
    :return: Pandas dataframe with columns 'identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'Log2FC', 'FC', 'T-statistics', 'p-value', 'padj', 'correction', '-log10 p-value', 'rejected' and 'method'

    Example::

        result = run_samr(df, subject='subject', group='group', drop_cols=['subject', 'sample'], alpha=0.05, s0=1, permutations=250, fc=0)
    """
    R_dataprep_function = R('''result <- function(x, y, genenames) {
                                list(x=as.matrix(x), y=y, genenames=genenames,logged2=TRUE)
                                }''')

    R_samr_function = R('''result <- function(data, res_type, s0, nperms) {
                                samr(data=data, resp.type=res_type, assay.type="array", s0=s0, nperms=nperms, random.seed = 12345, s0.perc=NULL)
                                }''')

    if permutations > 0 and r_installation:
        method, labels = define_samr_method(df, subject, group, drop_cols)
        groups = df[group].unique()
        df_py = df.drop(drop_cols, axis=1)
        df_r = df.drop(drop_cols+[group], axis=1).T.astype(float)
        data = R_dataprep_function(pandas2ri.py2rpy(df_r), np.array(labels), np.array(df_r.index))

        aov_results =[]
        pairwise_results = []
        for col in df_py.columns.drop(group).tolist():
            aov = calculate_anova(df_py[[group, col]], column=col, group=group)
            aov_results.append(aov)
            pairwise_result = calculate_pairwise_ttest(df_py[[group, col]], column=col, subject=subject, group=group, is_logged=is_logged)
            pairwise_cols = pairwise_result.columns
            pairwise_results.extend(pairwise_result.values.tolist())

        pairwise_results = pd.DataFrame(pairwise_results, columns=pairwise_cols)
        columns = ['identifier', 'dfk', 'dfn', 'F-statistics', 'pvalue']
        scores = pd.DataFrame(aov_results, columns = columns)
        scores = scores.set_index('identifier')
        dfk = scores['dfk'].values[0]
        dfn = scores['dfn'].values[0]

        if s0 is None or s0 == "null":
            s0 = ro.r("NULL")

        samr_res = R_samr_function(data=data, res_type=method, s0=s0, nperms=permutations)
        perm_pvalues = pd.DataFrame(samr_res.rx2('ttstar'), index=df_py.columns.drop(group))
        perm_pvalues = flatten(perm_pvalues.values.tolist(), my_list=[])
        perm_pvalues = np.array(calculate_pvalue_from_tstats(perm_pvalues, dfn, dfk))
        nperms_run = samr_res.rx2('nperms.act')[0]
        s0_used = samr_res.rx2('s0')[0]

        samr_df = pd.DataFrame(samr_res.rx2('tt'), index=df_py.columns.drop(group)).reset_index()
        samr_df.columns = ['identifier', 'F-statistics']
        samr_df = samr_df.set_index('identifier')
        samr_df['pvalue'] = list(calculate_pvalue_from_tstats(samr_df['F-statistics'].values, dfn, dfk))
        qvalues = []
        for i, row in samr_df.iterrows():
            qvalues.append(get_counts_permutation_fdr(row['pvalue'], perm_pvalues, samr_df['pvalue'], nperms_run, alpha)+(i,))

        qvalues = pd.DataFrame(qvalues, columns=['padj', 'rejected', 'identifier'])
        result = scores.join(qvalues.set_index('identifier'))
        result['F-statistics'] = result.index.map(samr_df['F-statistics'].to_dict().get)

        if not pairwise_results.empty:
            result = pairwise_results.set_index("identifier").join(result)
            if method != 'Multiclass':
                result = result.drop(['posthoc Paired', 'posthoc Parametric', 'posthoc T-Statistics',
                                    'posthoc dof', 'posthoc tail', 'posthoc pvalue', 'posthoc BF10'], axis=1)
                result = result.rename(columns={'F-statistics': 'T-statistics', 'posthoc effsize': 'effsize'})
            result = correct_pairwise_ttest(result.reset_index(), alpha)
            result = result.set_index('identifier')

        if 'posthoc pvalue' in result.columns:
            result['-log10 pvalue'] = [- np.log10(x) for x in result['posthoc pvalue'].values]
        else:
            result['-log10 pvalue'] = [- np.log10(x) for x in result['pvalue'].values]

        if nperms_run < permutations:
            rejected, padj = apply_pvalue_correction(result["pvalue"].tolist(), alpha=alpha, method='fdr_bh')
            result['padj'] = padj
            result['correction'] = 'FDR correction BH'
            result['Note'] = 'Maximum number of permutations: {}. Corrected instead using FDR correction BH'.format(nperms_run)
        else:
            result['correction'] = 'permutation FDR ({} perm)'.format(nperms_run)

        contrasts = ['diff_mean_group{}'.format(str(i+1)) for i in np.arange(len(set(labels)))]
        result['rejected'] = result['padj'] < alpha
        result['Method'] = 'SAMR {}'.format(method)
        result['s0'] = s0_used
        result = result.reset_index()
    else:
        result = run_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, group=group, permutations=permutations)

    return result

def calculate_discriminant_lines(result):
    for row in result.iterrows():
        lfc = row['log2fc']
        lpval = row['-log10 pvalue']

        v1 = (lfc, lpval)
        v2 = (0.0, lpval)

        angle = angle_between(v1, v2)


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


def run_kolmogorov_smirnov(dist1, dist2, alternative='two-sided'):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

    :param list dist1: sequence of 1-D ndarray (first distribution to compare) drawn from a continuous distribution
    :param list dist2: sequence of 1-D ndarray (second distribution to compare) drawn from a continuous distribution
    :param str alternative: defines the alternative hypothesis (default is ‘two-sided’):
        * **'two-sided'**
        * **'less'**
        * **'greater'**
    :return: statistic float and KS statistic pvalue float Two-tailed p-value.
    Example::

        result = run_kolmogorov_smirnov(dist1, dist2, alternative='two-sided')

    """

    result = stats.ks_2samp(dist1, dist2, alternative=alternative, mode='auto')

    return result


def run_site_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher', regex=r"(\w+~.+)_\w\d+\-\w+", correction='fdr_bh'):
    """
    This function runs a simple enrichment analysis for significantly regulated protein sites in a dataset.

    :param regulation_data: pandas dataframe resulting from differential regulation analysis.
    :param annotation: pandas dataframe with annotations for features (columns: 'annotation', 'identifier' (feature identifiers), and 'source').
    :param str identifier: name of the column from annotation containing feature identifiers.
    :param list groups: column names from regulation_data containing group identifiers.
    :param str annotation_col: name of the column from annotation containing annotation terms.
    :param str reject_col: name of the column from regulatio_data containing boolean for rejected null hypothesis.
    :param str group_col: column name for new column in annotation dataframe determining if feature belongs to foreground or background.
    :param str method: method used to compute enrichment (only 'fisher' is supported currently).
    :param str regex: how to extract the annotated identifier from the site identifier
    :return: Pandas dataframe with columns: 'terms', 'identifiers', 'foreground', 'background', 'pvalue', 'padj' and 'rejected'.

    Example::

        result = run_site_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher', match="(\w+~.+)_\w\d+\-\w+")
    """
    result = pd.DataFrame()
    if regulation_data is not None:
        if not regulation_data.empty:
            new_ids = []
            for ident in regulation_data[identifier].tolist():
                match = re.search(regex, ident)
                if match is not None:
                    new_ids.append(match.group(1))
                else:
                    new_ids.append(ident)
            regulation_data[identifier] = new_ids
            regulation_data = regulation_data.drop_duplicates()
            result = run_regulation_enrichment(regulation_data, annotation, identifier, groups, annotation_col, reject_col, group_col, method, correction)

    return result

def run_up_down_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher', correction='fdr_bh', alpha=0.05, lfc_cutoff=1):
    """
    This function runs a simple enrichment analysis for significantly regulated proteins distinguishing between up- and down-regulated.

    :param regulation_data: pandas dataframe resulting from differential regulation analysis (CKG's regulation table).
    :param annotation: pandas dataframe with annotations for features (columns: 'annotation', 'identifier' (feature identifiers), and 'source').
    :param str identifier: name of the column from annotation containing feature identifiers.
    :param list groups: column names from regulation_data containing group identifiers.
    :param str annotation_col: name of the column from annotation containing annotation terms.
    :param str reject_col: name of the column from regulation_data containing boolean for rejected null hypothesis.
    :param str group_col: column name for new column in annotation dataframe determining if feature belongs to foreground or background.
    :param str method: method used to compute enrichment (only 'fisher' is supported currently).
    :param str correction: method to be used for multiple-testing correction
    :param float alpha: adjusted p-value cutoff to define significance
    :param float lfc_cutoff: log fold-change cutoff to define practical significance
    :return: Pandas dataframe with columns: 'terms', 'identifiers', 'foreground', 'background', 'pvalue', 'padj' and 'rejected'.

    Example::

        result = run_up_down_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher', correction='fdr_bh', alpha=0.05, lfc_cutoff=1)
    """
    enrichment_results = {}
    for g1, g2 in regulation_data.groupby(groups).groups:
        df = regulation_data.groupby(groups).get_group((g1,g2))
        if 'posthoc padj' in df:
            df['up_pairwise_regulation'] = (df['posthoc padj'] <= alpha) & (df['log2FC'] >= lfc_cutoff)
            df['down_pairwise_regulation'] = (df['posthoc padj'] <= alpha) & (df['log2FC'] <= -lfc_cutoff)
        else:
            df['up_pairwise_regulation'] = (df['padj'] <= alpha) & (df['log2FC'] >= lfc_cutoff)
            df['down_pairwise_regulation'] = (df['padj'] <= alpha) & (df['log2FC'] <= -lfc_cutoff)

        enrichment = run_regulation_enrichment(df, annotation, identifier=identifier, groups=groups, annotation_col=annotation_col, reject_col='up_pairwise_regulation', group_col=group_col, method=method, correction=correction)
        enrichment['direction'] = 'upregulated'
        enrichment_results[g1+'~'+g2] = enrichment
        enrichment = run_regulation_enrichment(df, annotation, identifier=identifier, groups=groups, annotation_col=annotation_col, reject_col='down_pairwise_regulation', group_col=group_col, method=method, correction=correction)
        enrichment['direction'] = 'downregulated'
        enrichment_results[g1+'~'+g2] = enrichment_results[g1+'~'+g2].append(enrichment)

    return enrichment_results

def run_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher', correction='fdr_bh'):
    """
    This function runs a simple enrichment analysis for significantly regulated features in a dataset.

    :param regulation_data: pandas dataframe resulting from differential regulation analysis.
    :param annotation: pandas dataframe with annotations for features (columns: 'annotation', 'identifier' (feature identifiers), and 'source').
    :param str identifier: name of the column from annotation containing feature identifiers.
    :param list groups: column names from regulation_data containing group identifiers.
    :param str annotation_col: name of the column from annotation containing annotation terms.
    :param str reject_col: name of the column from regulatio_data containing boolean for rejected null hypothesis.
    :param str group_col: column name for new column in annotation dataframe determining if feature belongs to foreground or background.
    :param str method: method used to compute enrichment (only 'fisher' is supported currently).
    :param str correction: method to be used for multiple-testing correction
    :return: Pandas dataframe with columns: 'terms', 'identifiers', 'foreground', 'background', 'pvalue', 'padj' and 'rejected'.

    Example::

        result = run_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher')
    """
    result = {}
    foreground_list = regulation_data[regulation_data[reject_col]][identifier].unique().tolist()
    background_list = regulation_data[~regulation_data[reject_col]][identifier].unique().tolist()
    foreground_pop = len(foreground_list)
    background_pop = len(regulation_data[identifier].unique().tolist())
    grouping = []
    for i in annotation[identifier]:
        if i in foreground_list:
            grouping.append('foreground')
        elif i in background_list:
            grouping.append('background')
        else:
            grouping.append(np.nan)
    annotation[group_col] = grouping
    annotation = annotation.dropna(subset=[group_col])

    result = run_enrichment(annotation, foreground_id='foreground', background_id='background', foreground_pop=foreground_pop, background_pop=background_pop, annotation_col=annotation_col, group_col=group_col, identifier_col=identifier, method=method, correction=correction)

    return result


def run_enrichment(data, foreground_id, background_id, foreground_pop, background_pop, annotation_col='annotation', group_col='group', identifier_col='identifier', method='fisher', correction='fdr_bh'):
    """
    Computes enrichment of the foreground relative to a given backgroung, using Fisher's exact test, and corrects for multiple hypothesis testing.

    :param data: pandas dataframe with annotations for dataset features (columns: 'annotation', 'identifier', 'source', 'group').
    :param str foreground_id: group identifier of features that belong to the foreground.
    :param str background_id: group identifier of features that belong to the background.
    :param str annotation_col: name of the column containing annotation terms.
    :param str group_col: name of column containing the group identifiers.
    :param str identifier_col: name of column containing dependent variables identifiers.
    :param str method: method used to compute enrichment (only 'fisher' is supported currently).
    :return: Pandas dataframe with annotation terms, features, number of foregroung/background features in each term, p-values and corrected p-values (columns: 'terms', 'identifiers', 'foreground', 'background', 'pvalue', 'padj' and 'rejected').

    Example::

        result = run_enrichment(data, foreground='foreground', background='background', foreground_pop=len(foreground_list), background_pop=len(background_list), annotation_col='annotation', group_col='group', identifier_col='identifier', method='fisher')
    """
    result = pd.DataFrame()
    df = data.copy()
    terms = []
    ids = []
    pvalues = []
    fnum = []
    bnum = []
    countsdf = df.groupby([annotation_col, group_col]).agg(['count'])[(identifier_col, 'count')].reset_index()
    countsdf.columns = [annotation_col, group_col, 'count']
    for annotation in countsdf[countsdf[group_col] == foreground_id][annotation_col].unique().tolist():
        counts = countsdf[countsdf[annotation_col] == annotation]
        num_foreground = counts.loc[counts[group_col] == foreground_id,'count'].values
        num_background = counts.loc[counts[group_col] == background_id,'count'].values

        if len(num_foreground) == 1:
            num_foreground = num_foreground[0]
        if len(num_background) == 1:
            num_background = num_background[0]
        else:
            num_background=0
        if method == 'fisher' and num_foreground > 1:
            odds, pvalue = run_fisher([num_foreground, foreground_pop-num_foreground],[num_background, background_pop-foreground_pop-num_background])
            fnum.append(num_foreground)
            bnum.append(num_background)
            terms.append(annotation)
            pvalues.append(pvalue)
            ids.append(",".join(df.loc[(df[annotation_col]==annotation) & (df[group_col] == foreground_id), identifier_col].tolist()))
    if len(pvalues) > 1:
        rejected, padj = apply_pvalue_correction(pvalues, alpha=0.05, method=correction)
        result = pd.DataFrame({'terms':terms, 'identifiers':ids, 'foreground':fnum, 'background':bnum, 'foreground_pop':foreground_pop, 'background_pop':background_pop,'pvalue':pvalues, 'padj':padj, 'rejected':rejected})
        result = result.sort_values(by='padj',ascending=True)

    return result


def run_ssgsea(data, annotation, annotation_col='an notation', identifier_col='identifier', set_index=[], outdir=None, min_size=15, max_size=500, scale=False, permutations=0):
    """
    Project each sample within a data set onto a space of gene set enrichment scores using the ssGSEA projection methodology described in Barbie et al., 2009.

    :param data: pandas dataframe with the quantified features (i.e. subject x proteins)
    :param annotation: pandas dataframe with the annotation to be used in the enrichment (i.e. CKG pathway annotation file)
    :param str annotation_col: name of the column containing annotation terms.
    :param str identifier_col: name of column containing dependent variables identifiers.
    :param list set_index: column/s to be used as index. Enrichment will be calculated for these values (i.e ["subject"] will return subjects x pathways matrix of enrichment scores)
    :param str out_dir: directory path where results will be stored (default None, tmp folder is used)
    :param int min_size: minimum number of features (i.e. proteins) in enriched terms (i.e. pathways)
    :param int max_size: maximum number of features (i.e. proteins) in enriched terms (i.e. pathways)
    :param bool scale: whether or not to scale the data
    :param int permutations: number of permutations used in the ssgsea analysis
    :return: dictionary with two dataframes: es - enrichment scores, and nes - normalized enrichment scores.

    Example::
        stproject = "P0000008"
        p = project.Project(stproject, datasets={}, knowledge=None, report={}, configuration_files=None)
        p.build_project(False)
        p.generate_report()

        proteomics_dataset = p.get_dataset("proteomics")
        annotations = proteomics_dataset.get_dataframe("pathway annotation")
        processed = proteomics_dataset.get_dataframe('processed')

        result = run_ssgsea(processed, annotations, annotation_col='annotation', identifier_col='identifier', set_index=['group', 'sample','subject'], outdir=None, min_size=10, scale=False, permutations=0)

    """
    result = {}
    df = data.copy()
    if outdir is None:
        outdir = ckg_utils.read_ckg_config(key='tmp_directory')
    if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    name = []
    index = data[set_index]
    for i, row in data[set_index].iterrows():
        name.append("_".join(row[set_index].tolist()))

    df['Name'] = name
    index.index = name
    df = df.drop(set_index, axis=1).set_index('Name').transpose()

    if annotation_col in annotation and identifier_col in annotation:
        grouped_annotations = annotation.groupby(annotation_col)[identifier_col].apply(list).reset_index()
        fid = uuid.uuid4()
        file_path = os.path.join(outdir, str(fid)+'.gmt')
        with open(file_path, 'w') as out:
            for i, row in grouped_annotations.iterrows():
                out.write(row[annotation_col]+"\t"+"\t".join(list(filter(None, row[identifier_col])))+"\n")
        try:
            enrichment = gp.ssgsea(data=df, 
                                gene_sets=str(file_path), 
                                outdir=outdir, 
                                min_size=min_size,
                                max_size=max_size,
                                scale=scale, 
                                permutation_num=permutations, 
                                no_plot=True, 
                                processes=1, 
                                seed=10, 
                                format='png')

            enrichment_es = pd.DataFrame(enrichment.resultsOnSamples).transpose()
            enrichment_es = enrichment_es.join(index)
            enrichment_nes = enrichment.res2d.transpose()
            enrichment_nes = enrichment_nes.join(index)
            
            result = {'es': enrichment_es, 'nes': enrichment_nes}
        except Exception as e:
            print("Error in ssGSEA.", e)
    df = None

    return result

def calculate_fold_change(df, condition1, condition2):
    """
    Calculates fold-changes between two groups for all proteins in a dataframe.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns.
    :param str condition1: identifier of first group.
    :param str condition2: identifier of second group.
    :return: Numpy array.

    Example::

        result = calculate_fold_change(data, 'group1', 'group2')
    """
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


def pooled_standard_deviation(sample1,sample2, ddof):
    """
    Calculates the pooled standard deviation.
    For more information visit https://www.hackdeploy.com/learn-what-is-statistical-power-with-python/.

    :param array sample1: numpy array with values for first group
    :param array sample2: numpy array with values for second group
    :param int ddof: degrees of freedom
    """
    #calculate the sample size
    n1, n2 = len(sample1), len(sample2)
    #calculate the variances
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=ddof)
    #calculate the pooled standard deviation
    numerator = ((n1-1) * var1) + ((n2-1) * var2)
    denominator = n1+n2-2
    return np.sqrt(numerator/denominator)


def cohens_d(sample1, sample2, ddof):
    """
    Calculates Cohen's d effect size based on the distance between two means, measured in standard deviations.
    For more information visit https://www.hackdeploy.com/learn-what-is-statistical-power-with-python/.

    :param array sample1: numpy array with values for first group
    :param array sample2: numpy array with values for second group
    :param int ddof: degrees of freedom
    """
    u1, u2 = np.mean(sample1), np.mean(sample2)
    s_pooled = pooled_standard_deviation(sample1, sample2, ddof)

    return ((u1 - u2) / s_pooled)


def hedges_g(df, condition1, condition2, ddof = 0):
    """
    Calculates Hedges’ g effect size (more accurate for sample sizes below 20 than Cohen's d).
    For more information visit https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanstd.html.

    :param df: pandas dataframe with samples as rows and protein identifiers as columns.
    :param str condition1: identifier of first group.
    :param str condition2: identifier of second group.
    :param int ddof: means Delta Degrees of Freedom.
    :return: Numpy array.

    Example::

        result = hedges_g(data, 'group1', 'group2', ddof=0)
    """
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

def power_analysis(data, group='group', groups=None, alpha=0.05, power=0.8, dep_var='nobs', figure=False):
    quantiles = ['25% qtl es','mean es', '50% qtl es', '75% qtl es']
    if groups is None:
        groups = data[group].unique().tolist()
    k_groups = len(groups)
    effect_sizes = set()
    for col in data.drop([group], axis=1).columns:
        for g1, g2 in itertools.combinations(groups, 2):
            sample1 = data.loc[data[group] == g1, col].values
            sample2 = data.loc[data[group] == g2, col].values
            eff_size = np.abs(cohens_d(sample1,sample2, ddof=1))
            effect_sizes.add(eff_size)

    summary_eff = []
    if len(effect_sizes):
        effect_sizes = list(effect_sizes)
        summary_eff = [np.percentile(effect_sizes, 25),
                    np.mean(effect_sizes),
                    np.percentile(effect_sizes, 50),
                    np.percentile(effect_sizes, 75)]
        
    analysis = FTestAnovaPower()
    sample_sizes = np.array(range(3, 150))
    power_list = []
    labels = []
    samples = []
    for ii, es in enumerate(summary_eff):
        p = analysis.power(es, sample_sizes, alpha, k_groups)
        labels.extend(["%s = %4.2F" % (quantiles[ii], es)] * len(p))
        power_list.extend(p)
        samples.extend(sample_sizes)

    power_df = pd.DataFrame(data=list(zip(power_list, samples, labels)), columns=['power', '#samples', 'labels'])
    sample_size = analysis.solve_power(summary_eff[1], power=power, alpha=alpha, k_groups=k_groups)

    return (sample_size, power_df)


def run_mapper(data, lenses=["l2norm"], n_cubes = 15, overlap=0.5, n_clusters=3, linkage="complete", affinity="correlation"):
    """


    :param data:
    :param lenses:
    :param n_cubes:
    :param overlap:
    :param n_clusters:
    :param linkage:
    :param affinity:
    :return:

    """
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


def run_WGCNA(data, drop_cols_exp, drop_cols_cli, RsquaredCut=0.8, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0, sd_cutoff=0):
    """
    Runs an automated weighted gene co-expression network analysis (WGCNA), using input proteomics/transcriptomics/genomics and clinical variables data.

    :param dict data: dictionary of pandas dataframes with processed clinical and experimental datasets
    :param list drop_cols_exp: column names to be removed from the experimental dataset.
    :param list drop_cols_cli: column names to be removed from the clinical dataset.
    :param float RsquaredCut: desired minimum scale free topology fitting index R^2.
    :param str networkType: network type ('unsigned', 'signed', 'signed hybrid', 'distance').
    :param int minModuleSize: minimum module size.
    :param int deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
    :param bool pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.
    :param bool merge_modules: if True, very similar modules are merged.
    :param float MEDissThres: maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging.
    :param int verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    :return: Tuple with multiple pandas dataframes.

    Example::

        result = run_WGCNA(data, drop_cols_exp=['subject', 'sample', 'group', 'index'], drop_cols_cli=['subject', 'biological_sample', 'group', 'index'], RsquaredCut=0.8, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0)
    """
    if not r_installation:
        raise Exception("No valid installation of R")

    result = {}
    dfs = wgcna.get_data(data, drop_cols_exp=drop_cols_exp, drop_cols_cli=drop_cols_cli, sd_cutoff=sd_cutoff)
    if 'clinical' in dfs:
        data_cli = dfs['clinical']   #Extract clinical data
        for dtype in dfs:
            if dtype in ['proteomics', 'rnaseq']:
                data_exp = dfs[dtype]   #Extract experimental data
                dtype = 'wgcna-'+dtype
                result[dtype] = {}

                softPower = wgcna.pick_softThreshold(data_exp, RsquaredCut=RsquaredCut, networkType=networkType, verbose=verbose)

                dissTOM, moduleColors = wgcna.build_network(data_exp, softPower=softPower, networkType=networkType, minModuleSize=minModuleSize, deepSplit=deepSplit,
                                                    pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules, MEDissThres=MEDissThres, verbose=verbose)
                if not dissTOM.empty and len(moduleColors) > 0:
                    Features_per_Module = wgcna.get_FeaturesPerModule(data_exp, moduleColors, mode='dataframe')

                    MEs, MEDiss = wgcna.calculate_module_eigengenes(data_exp, moduleColors, softPower=softPower, dissimilarity=False)

                    moduleTraitCor, textMatrix = wgcna.calculate_ModuleTrait_correlation(data_exp, data_cli, MEs)

                    #MM, MMPvalue = wgcna.calculate_ModuleMembership(data_exp, MEs)

                    #FS, FSPvalue = wgcna.calculate_FeatureTraitSignificance(data_exp, data_cli)

                    METDiss, METcor = wgcna.get_EigengenesTrait_correlation(MEs, data_cli)

                    result[dtype]['dissTOM'] = dissTOM
                    result[dtype]['module_colors'] = moduleColors
                    result[dtype]['features_per_module'] = Features_per_Module
                    result[dtype]['MEs'] = MEs
                    result[dtype]['module_trait_cor'] = moduleTraitCor
                    result[dtype]['text_matrix'] = textMatrix
                    #result[dtype]['module_membership'] = MM
                    #result[dtype]['module_membership_pval'] = MMPvalue
                    #result[dtype]['feature_significance'] = FS
                    #result[dtype]['feature_significance_pval'] = FSPvalue
                    result[dtype]['ME_trait_diss'] = METDiss
                    result[dtype]['ME_trait_cor'] = METcor

    return result


def most_central_edge(G):
    """
    Compute the eigenvector centrality for the graph G, and finds the highest value.

    :param graph G: networkx graph
    :return: Highest eigenvector centrality value.
    :rtype: float
    """
    centrality = nx.eigenvector_centrality_numpy(G, weight='width')

    return max(centrality, key=centrality.get)


def get_louvain_partitions(G, weight):
    """
    Computes the partition of the graph nodes which maximises the modularity (or try..) using the Louvain heuristices. For more information visit https://python-louvain.readthedocs.io/en/latest/api.html.

    :param graph G: networkx graph which is decomposed.
    :param str weight: the key in graph to use as weight.
    :return: The partition, with communities numbered from 0 to number of communities.
    :rtype: dict
    """
    partition = community.best_partition(G, weight=weight)

    return partition


def get_network_communities(graph, args):
    """
    Finds communities in a graph using different methods. For more information on the methods visit:

        - https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
        - https://networkx.github.io/documentation/networkx-2.0/reference/algorithms/generated/networkx.algorithms.community.asyn_lpa.asyn_lpa_communities.html
        - https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html
        - https://networkx.github.io/documentation/latest/reference/generated/networkx.convert_matrix.to_pandas_adjacency.html

    :param graph graph: networkx graph
    :param dict args: config file arguments
    :return: Dictionary of nodes and which community they belong to (from 0 to number of communities).
    """
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


def get_publications_abstracts(data, publication_col="publication", join_by=['publication', 'Proteins', 'Diseases'], index="PMID"):
    """
    Accesses NCBI PubMed over the WWW and retrieves the abstracts corresponding to a list of one or more PubMed IDs.

    :param data: pandas dataframe of diseases and publications linked to a list of proteins (columns: 'Diseases', 'Proteins', 'linkout' and 'publication').
    :param str publication_col: column label containing PubMed ids.
    :param list join_by: column labels to be kept from the input dataframe.
    :param str index: column label containing PubMed ids from the NCBI retrieved data.
    :return: Pandas dataframe with publication information and columns 'PMID', 'abstract', 'authors', 'date', 'journal', 'keywords', 'title', 'url', 'Proteins' and 'Diseases'.

    Example::

        result = get_publications_abstracts(data, publication_col='publication', join_by=['publication','Proteins','Diseases'], index='PMID')
    """
    abstracts = pd.DataFrame()
    if not data.empty:
        abstracts = utils.getMedlineAbstracts(list(data.reset_index()[publication_col].unique()))
        if not abstracts.empty:
            abstracts = abstracts.set_index(index)
            abstracts = abstracts.join(data.reset_index()[join_by].set_index(publication_col))
            abstracts.index.name = index
            abstracts = abstracts.reset_index()
    return abstracts


def eta_squared(aov):
    """
    Calculates the effect size using Eta-squared.

    :param aov: pandas dataframe with anova results from statsmodels.
    :return: Pandas dataframe with additional Eta-squared column.
    """
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    """
    Calculates the effect size using Omega-squared.

    :param aov: pandas dataframe with anova results from statsmodels.
    :return: Pandas dataframe with additional Omega-squared column.
    """
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov


def run_two_way_anova(df, drop_cols=['sample'], subject='subject', group=['group', 'secondary_group']):
    """
    Run a 2-way ANOVA when data['secondary_group'] is not empty

    :param df: processed pandas dataframe with samples as rows, and proteins and groups as columns.
    :param list drop_cols: column names to drop from dataframe
    :param str subject: column name containing subject identifiers.
    :param list group: column names corresponding to independent variable groups
    :return: Two dataframes, anova results and residuals.

    Example::

        result = run_two_way_anova(data, drop_cols=['sample'], subject='subject', group=['group', 'secondary_group'])
    """
    data = df.copy()
    factorA, factorB = group
    data = data.set_index([subject]+group)
    data = data.drop(drop_cols, axis=1)
    data.columns = data.columns.str.replace(r"-", "_")

    aov_result = []
    residuals = {}
    for col in data.columns:
        model = ols('{} ~ C({})*C({})'.format(col, factorA, factorB), data[col].reset_index().sort_values(group, ascending=[True, False])).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        eta_squared(aov_table)
        omega_squared(aov_table)
        for i in aov_table.index:
            if i != 'Residual':
                t, p, eta, omega = aov_table.loc[i, ['F', 'PR(>F)', 'eta_sq', 'omega_sq']]
                protein = col.replace('_', '-')
                aov_result.append((protein, i, t, p, eta, omega))
        residuals[col] = model.resid

    anova_df = pd.DataFrame(aov_result, columns = ['identifier','source', 'F-statistics', 'pvalue', 'eta_sq', 'omega_sq'])
    anova_df = anova_df.set_index('identifier')
    anova_df = anova_df.dropna(how="all")

    return anova_df, residuals

def merge_for_polar(regulation_data, regulators, identifier_col='identifier', group_col='group', theta_col='modifier', aggr_func='mean', normalize=True):
    aggr_df = pd.DataFrame()
    if normalize:
        regulation_data = normalize_data(regulation_data, method='zscore')

    df = regulation_data.groupby(group_col)
    list_cols = []
    for i, group in df:
        if aggr_func == 'mean':
            list_cols.append(group.mean())
        elif aggr_func == 'median':
            list_cols.append(group.median())
        elif aggr_func == 'sum':
            list_cols.append(group.sum())
        else:
            break

    if len(list_cols) > 0:
        aggr_df = pd.DataFrame(list_cols, index=regulation_data[group_col].unique(), columns=regulation_data.set_index(group_col).columns).stack().reset_index()
        aggr_df.columns = [group_col, identifier_col, 'value']
        aggr_df = pd.merge(aggr_df, regulators, on=identifier_col)

    if not aggr_df.empty:
        list_cols = []
        for i, group in aggr_df.groupby([group_col, theta_col]):
            if aggr_func == 'mean':
                value = group['value'].mean()
            elif aggr_func == 'median':
                value = group['value'].median()
            elif aggr_func == 'sum':
                value = group['value'].sum()
            else:
                break

            list_cols.append((*i, value))

        if len(list_cols) > 0:
            aggr_df = pd.DataFrame(list_cols, columns=[group_col, theta_col, 'value'])

    return aggr_df

def run_qc_markers_analysis(data, qc_markers, sample_col='sample', group_col='group', drop_cols=['subject'], identifier_col='identifier', qcidentifier_col='identifier', qcclass_col='class'):
    """
    """
    bdf = None
    if data is not None and qc_markers is not None:
        if not data.empty and not qc_markers.empty:
            cols = list(set(data.columns.tolist()).intersection(qc_markers[qcidentifier_col].tolist()))

            if len(cols) > 0:
                nd = data.set_index([sample_col]).drop(drop_cols + [group_col], axis=1)
                z = pd.DataFrame(normalize_data(nd, method='zscore'), index=nd.index, columns=nd.columns)
                nd = z.unstack()
                nd = nd.reset_index().set_index(sample_col).join(data[[sample_col, group_col]].set_index(sample_col)).reset_index()
                bdf = pd.DataFrame()
                for i, group in qc_markers.groupby(qcclass_col):
                    c = group[group[qcidentifier_col].isin(cols)][qcidentifier_col].tolist()
                    if bdf.empty:
                        bdf = nd.set_index(identifier_col).loc[c].reset_index()
                        bdf.columns = [identifier_col, sample_col, 'z-score', group_col]
                        bdf[qcclass_col] = i
                    else:
                        aux = nd.set_index(identifier_col).loc[c].reset_index()
                        aux.columns = [identifier_col, sample_col, 'z-score', group_col]
                        aux[qcclass_col] = i
                        bdf = bdf.append(aux)

    return bdf


def get_snf_clusters(data_tuples, num_clusters=None, metric='euclidean', k=5, mu=0.5):
    """
    Cluster samples based on Similarity Network Fusion (SNF) (ref: https://www.ncbi.nlm.nih.gov/pubmed/24464287)

    :param df_tuples: list of (dataset,metric) tuples
    :param index: how the datasets can be merged (common columns)
    :param num_clusters: number of clusters to be identified, if None, the algorithm finds the best number based on SNF algorithm (recommended)
    :param distance_metric: distance metric used to calculate the sample similarity network
    :param k: number of neighbors used to measure local affinity (KNN)
    :param mu: normalization factor to scale similarity kernel when constructing affinity matrix
    :return tuple: 1) fused_aff: affinity clustered samples, 2) fused_labels: cluster labels,
                    3) num_clusters: number of clusters, 4) silhouette: average silhouette score
    """
    affinities = []
    for (d, m) in data_tuples:
        affinities += [snf.make_affinity(d, metric=m, K=k, mu=mu)]
    fused_aff = snf.snf(affinities, K=k)
    if num_clusters is None:
        num_clusters, second = snf.get_n_clusters(fused_aff)
    fused_labels = cluster.spectral_clustering(fused_aff, n_clusters=num_clusters)
    fused_labels = [i+1 for i in fused_labels]
    silhouette = snf.metrics.silhouette_score(fused_aff, fused_labels)

    return (fused_aff, fused_labels, num_clusters, silhouette)


def run_snf(df_dict, index, num_clusters=None, distance_metric='euclidean', k_affinity=5, mu_affinity=0.5):
    """
    Runs Similarity Network Fusion: integration of multiple omics datasets to identify
    similar samples (clusters) (ref: https://www.ncbi.nlm.nih.gov/pubmed/24464287).
    We make use of the pyton version SNFpy (https://github.com/rmarkello/snfpy)

    :param df_dict: dictionary of datasets to be used (i.e {'rnaseq': rnaseq_data, 'proteomics': proteomics_data})
    :param index: how the datasets can be merged (common columns)
    :param num_clusters: number of clusters to be identified, if None, the algorithm finds the best number based on SNF algorithm (recommended)
    :param distance_metric: distance metric used to calculate the sample similarity network
    :param k_affinity: number of neighbors used to measure local affinity (KNN)
    :param mu_ffinity: normalization factor to scale similarity kernel when constructing affinity matrix
    :return tuple: 1) feature_df: SNF features and mutual information score (MIscore), 2) fused_aff: adjacent similarity matrix, 3)fused_labels: cluster labels,
                    4) silhouette: silhouette score
    """
    datasets = []
    dataset_labels = []
    common_samples = set()
    for dtype in df_dict:
        dataset_labels.append(dtype)
        df = df_dict[dtype]
        df = df.set_index(index)
        datasets.append(df)
        if len(common_samples) > 1:
            common_samples = common_samples.intersection(df.index)
        else:
            common_samples.update(df.index.tolist())

    data_tuples = [(d.loc[list(common_samples)].values, distance_metric) for d in datasets]

    fused_aff, fused_labels, num_clusters, silhouette_score = get_snf_clusters(data_tuples, num_clusters, metric=distance_metric, k=k_affinity, mu=mu_affinity)

    fused_labels = pd.DataFrame(fused_labels, index=common_samples, columns=['cluster'])

    snf_features = snf.metrics.rank_feature_by_nmi(data_tuples, fused_aff, K=k_affinity, mu=mu_affinity, n_clusters=num_clusters)

    feature_df = pd.DataFrame(columns=['MIscore'])
    indexes = [df.columns for df in datasets]
    i = 0
    for dtype in snf_features:
        df = pd.DataFrame(dtype, index = indexes[i], columns = ["MIscore"]).sort_values(by="MIscore", ascending=False)
        df['dataset'] = dataset_labels[i]
        i += 1
        feature_df = feature_df.append(df)

    feature_df = feature_df.sort_values(by='MIscore', ascending=False)

    return feature_df, fused_aff, fused_labels, silhouette_score


def run_km(data, time_col, event_col, group_col, args={}):
    kmfit, summary = kaplan_meierAnalysis.run_km(data, time_col, event_col, group_col, args)

    return kmfit, summary
