import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test


def get_data_ready_for_km(dfs_dict, args):
    kmdf = None
    for df_key in dfs_dict:
        if df_key == 'clinical':
            kmdf = dfs_dict[df_key]
        else:
            df = dfs_dict[df_key]
            if 'marker' in args and 'index_col' in args:
                how = 'half'
                value = None
                if 'how' in args:
                    how = args['how']
                if 'value' in args:
                    value = args['value']
                mdf = group_data_based_on_marker(df, args['marker'], args['index_col'], how, value)

    if kmdf is not None:
        if 'index_col' in args and args['index_col'] in kmdf:
            index_col = args['index_col']
            kmdf = kmdf.set_index(index_col).join(mdf.set_index(index_col), how='inner')

    return kmdf


def group_data_based_on_marker(df, marker, index_col, how, value):
    mdf = pd.DataFrame()
    if index_col is not None and marker is not None:
        if index_col in df and marker in df:
            mdf = df[[marker, index_col]]
            if how == 'cutoff':
                mdf['new_grouping'] = mdf.apply(lambda row: str(marker) + '+' if row[marker] >= value else str(marker)+'-')
            elif how == 'top' or how == 'top%':
                mdf = mdf.sort_values(by=marker, ascending=False)
                num_values = len(mdf[marker].values.tolist())
                if how == 'top%':
                    value = int(num_values * value / 100)
                if value < num_values:
                    labels = [str(marker)+'+'] * value
                    labels.extend([str(marker)+'-'] * (num_values - value))
                else:
                    print("Invalid value provided. Exceeded maximun number of samples {}".format(num_values))    
                mdf['new_grouping'] = labels
            else:
                print("Grouping method {} not implemented. Try with 'cutoff' or 'top'".format(how))

    return mdf


def run_km(data, time_col, event_col, group_col, args={}):
    kmdf = None
    kmf = pd.DataFrame()
    summary = None
    if isinstance(data, dict):
        kmdf = get_data_ready_for_km(data, args)
        group_col = 'new_grouping'
    elif isinstance(data, pd.DataFrame):
        kmdf = data

    if kmdf is not None:
        kmf, summary = get_km_results(kmdf, group_col, time_col, event_col)

    return kmf, summary


def get_km_results(df, group_col, time_col, event_col):
    models = []
    summary_ = None
    summary_result = None
    df = df[[event_col, time_col, group_col]].dropna()
    df[event_col] = df[event_col].astype('category')
    df[event_col] = df[event_col].cat.codes
    df[time_col] = df[time_col].astype('float')
    if not df.empty:
        for name, grouped_df in df.groupby(group_col):
            kmf = KaplanMeierFitter()
            t = grouped_df[time_col]
            e = grouped_df[event_col]
            kmf.fit(t, event_observed=e, label=name + " (N=" + str(len(t.tolist())) + ")")
            models.append(kmf)

        summary_ = multivariate_logrank_test(df[time_col].tolist(), df[group_col].tolist(), df[event_col].tolist(), alpha=99)

    if summary_ is not None:
        summary_result = "Multivariate logrank test: pval={}, t_statistic={}".format(summary_.p_value, summary_._test_statistic)

    return models, summary_result


def get_hazard_ratio_results(df, group_col, time_col, event_col):
    models = []
    summary_ = None
    summary_result = None
    df = df[[event_col, time_col, group_col]].dropna()
    df[event_col] = df[event_col].astype('category')
    df[event_col] = df[event_col].cat.codes
    df[time_col] = df[time_col].astype('float')
    if not df.empty:
        for name, grouped_df in df.groupby(group_col):
            hr = NelsonAalenFitter()
            t = grouped_df[time_col]
            e = grouped_df[event_col]
            hr.fit(t, event_observed=e, label=name + " (N=" + str(len(t.tolist())) + ")")
            models.append(hr)

    return models
