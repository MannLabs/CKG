from ckg.analytics_core.viz import viz
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly.subplots as tools
import pandas as pd
import numpy as np
from itertools import chain
from collections import defaultdict
from natsort import natsorted


def get_stats_data(filename, n=3):
    """
    Reads graph database stats file and filters for the last 'n' full and partial independent \
    imports, returning a Pandas DataFrame.

    :param str filename: path to stats file (including filename and '.hdf' extension).
    :param int n: number of independent imports to plot.
    :return: Pandas Dataframe with different entities and relationships as rows and columns:
    """
    df = pd.DataFrame()
    with pd.HDFStore(filename, 'r') as store:
        for key in store.keys():
            k = key.split('_')[0].replace('/', '')
            if df.empty:
                df = store[key].copy()
                df['Import_flag'] = k
            else:
                aux = store[key]
                aux['Import_flag']= k
                df = df.append(aux)
    
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['date']+' '+df['time'])
        imp = select_last_n_imports(df, n=n)
        df = df[df['import_id'].isin(imp)].reset_index(drop=True)

    return df


def select_last_n_imports(stats_file, n=3):
    """
    Selects which independent full and partial imports should be plotted based on n.

    :param stats_file: pandas DataFrame with stats data.
    :param int n: number of independent imports to select.
    :return: List of import ids to be plotted according to selection criterion.
    """
    partial = []
    full = []
    if 'datetime' in  stats_file and 'import_id' in stats_file and 'Import_flag' in stats_file:
        df = stats_file[['datetime', 'import_id', 'Import_flag']].sort_values('datetime', ascending=False).drop_duplicates(['import_id'], keep = 'first', inplace = False)
        full = df[df['Import_flag'] == 'full']
        full = full.iloc[:n, 1].tolist()
        partial = df[df['Import_flag'] == 'partial']
        partial = partial.iloc[:n, 1].tolist()
    
    return partial + full



def remove_legend_duplicates(figure):
    """
    Removes duplicated legend items.

    :param figure: plotly graph object figure.
    """
    seen = []
    if 'data' in figure:
        for n,i in enumerate(figure['data']):
            name = figure['data'][n]['name']
            if name in seen:
                figure.data[n].update(showlegend=False)
            else:
                figure.data[n].update(showlegend=True)
            seen.append(name)


def get_databases_entities_relationships(stats_file, key='full', options='databases'):
    """
    Builds dictionary from stats file. Depending on 'options', keys and values can differ. \
    If *options* is set to 'dates', keys are dates of the imports and values are databases imported at each date; \
    if 'databases', keys are databases and values are entities and relationships created from each database; \
    if 'entities', keys are databases and values are entities created from each database; \
    if 'relationships', keys are databases and values are relationships created from each database.

    :param stats_file: pandas DataFrame with stats data.
    :param str key: use only full, partial or both kinds of imports ('full', 'partial', 'all').
    :param str options: name of the variables to be used as keys in the output dictionary ('dates', \
                        'databases', 'entities' or 'relationships').
    :return: Dictionary.
    """
    d_ent = {}
    d_rel = {}
    d_dat = {}
    d_dbs_filename = None
    stats = pd.DataFrame()
    if 'Import_flag' in stats_file:
        if key == 'full':
            stats = stats_file[stats_file['Import_flag'] == 'full']
        elif key == 'partial':
            stats = stats_file[stats_file['Import_flag'] == 'partial']
        elif key == 'all':
            stats = stats_file
    if not stats.empty:
        if 'Import_type' in stats:
            mask = (stats['Import_type']=='entity')
            mask2 = (stats['Import_type']=='relationships')
            ent = list(set(list(zip(stats.loc[mask,'filename'], stats.loc[mask,'dataset']))))
            rel = list(set(list(zip(stats.loc[mask2,'filename'], stats.loc[mask2,'dataset']))))

        if 'import_id' in stats and 'datetime in stats':
            dat = []
            for i, j in stats.groupby('import_id'):
                date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
                for i in j['dataset'].unique():
                    dat.append((date, i))

            d_dat = defaultdict(list)
            for k, v in dat:
                d_dat[k].append(v)
            d_dat = {k: tuple(v) for k, v in d_dat.items()}
            d_dat = dict(natsorted(d_dat.items()))

            d_ent = defaultdict(list)
            for k, v in ent:
                d_ent[v].append(k)
            d_ent = {k: tuple(v) for k, v in d_ent.items()}
            d_ent = dict(natsorted(d_ent.items()))

            d_rel = defaultdict(list)
            for k, v in rel:
                d_rel[v].append(k)
            d_rel = {k: tuple(v) for k, v in d_rel.items()}
            d_rel = dict(natsorted(d_rel.items()))

        if 'dataset' in stats_file:
            for i in stats_file['dataset'].unique():
                if i not in d_ent.keys():
                    d_ent[i] = ''
                if i not in d_rel.keys():
                    d_rel[i] = ''

        d_dbs_filename = defaultdict(list)
        for k, v in chain(d_ent.items(), d_rel.items()):
            d_dbs_filename[k].append(v)
        d_dbs_filename = {k: tuple(v) for k, v in d_dbs_filename.items()}
        d_dbs_filename = dict(natsorted(d_dbs_filename.items()))

    if options == 'entities':
        return d_ent
    if options == 'relationships':
        return d_rel
    if options == 'databases':
        return d_dbs_filename
    if options == 'dates':
        return d_dat


def set_colors(dictionary):
    """
    This function takes the values in a dictionary and attributes them an RGB color.

    :param dict dictionary: dictionary with variables to be attributed a color, as values.
    :return: Dictionary where 'dictionary' values are keys and random RGB colors are the values.
    """
    colors = []
    for i in list(chain(*dictionary.values())):
        color = 'rgb' + str(tuple(np.random.choice(range(256), size=3)))
        colors.append((i, color))
    colors = dict(colors)

    return colors


def get_dropdown_menu(fig, options_dict, add_button=True, equal_traces=True, number_traces=2):
    """
    Builds a list for the dropdown menu, based on a plotly figure traces and a dictionary with \
    the options to be used in the dropdown.

    :param fig: plotly graph object figure.
    :param options_dict: dictionary where keys are used as dropdown options and values data points.
    :param bool add_button: add option to display all dropdown options simultaneously.
    :param bool equal_traces: defines if all dropdown options have the same number of traces each. \
                                If True, define 'number_traces' as well. If False, number of traces \
                                will be the same as the number of values for each 'options_dict' key.
    :param int number_traces: number of traces created for each 'options_dict' key.
    :return: List of nested structures. Each dictionary within *updatemenus[0]['buttons'][0]* corresponds \
            to one dropdown menu options and contains information on which traces are visible, label and method.
    """

    list_updatemenus = []
    updatemenus = []
    start = 0
    if options_dict is not None:
        for n, i in enumerate(options_dict.keys()):
            if equal_traces:
                visible = [False] * len(fig['data'])
                end = start + number_traces
                visible[start:end] = [True] * number_traces
                start += number_traces
            else:
                number_traces = len([element for tupl in options_dict[i] for element in tupl])*2
                visible = [False] * len(fig['data'])
                end = start + number_traces
                visible[start:end] = [True] * number_traces
                start += number_traces
            temp_dict = dict(label=str(i),
                            method='update',
                            args=[{'visible': visible},
                                    {'title': 'Date: '+i}])
            list_updatemenus.append(temp_dict)

        if add_button:
            button = [dict(label='All',
                            method='update',
                            args=[{'visible': [True] * len(fig['data'])}, {'title': 'All'}])]
            list_updatemenus = list_updatemenus + button
        else:
            pass

        updatemenus = list([dict(active=len(list_updatemenus)-1,
                                buttons=list_updatemenus,
                                direction='down',
                                showactive=True, x=-0.17, xanchor='left', y=1.1, yanchor='top'), ])

    return updatemenus


def get_totals_per_date(stats_file, key='full', import_types=False):
    """
    Summarizes stats file to a Pandas DataFrame with import dates and total number of \
    imported entities and relationships.

    :param stats_file: pandas DataFrame with stats data.
    :param str key: use only full or partial imports ('full', 'partial').
    :param bool import_types: breakdown importing stats into entities or relationships related.
    :return: Pandas DataFrame with independent import dates as rows and imported numbers as columns.
    """
    df = pd.DataFrame()
    stats = pd.DataFrame()
    if 'Import_flag' in stats_file:
        if key == 'full':
            stats = stats_file[stats_file['Import_flag'] == 'full']
        elif key == 'partial':
            stats = stats_file[stats_file['Import_flag'] == 'partial']

    cols = ['date', 'total']
    counts = []
    if not stats.empty:
        if 'import_id' in stats:
            for i, j in stats.groupby('import_id'):
                date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
                count = j['Imported_number'].sum()
                counts.append((date, count))

            df = pd.DataFrame(counts, columns=cols)
            df = df.set_index('date')

            if import_types:
                cols = ['date', 'entity', 'relationships']
                counts = []
                for i, j in stats.groupby(['import_id']):
                    date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
                    ent = j.loc[(j['Import_type'] == 'entity'), 'Imported_number'].sum()
                    rel = j.loc[(j['Import_type'] == 'relationships'), 'Imported_number'].sum()
                    counts.append((date, ent, rel))

                df = pd.DataFrame(counts, columns=cols)
                df = df.set_index('date')

    return df


def get_imports_per_database_date(stats_file):
    """
    Summarizes stats file to a Pandas DataFrame with import dates, databases and total number of \
    imported entities and relationships per database.

    :param stats_file: pandas DataFrame with stats data.
    :return: Pandas DataFrame with independent import dates and databases as rows and imported numbers as columns.
    """
    cols = ['date', 'dataset', 'entities', 'relationships', 'total']
    stats_sum = []
    df = pd.DataFrame()
    if not stats_file.empty:
        for i, j in stats_file.groupby(['import_id']):
            date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
            for a, b in j.groupby('dataset'):
                ent = b.loc[(b['Import_type'] == 'entity'), 'Imported_number'].sum()
                rel = b.loc[(b['Import_type'] == 'relationships'), 'Imported_number'].sum()
                total = b['Imported_number'].sum()
                stats_sum.append((date, a, ent, rel, total))

        df = pd.DataFrame(stats_sum, columns=cols)
        df = df.sort_values(['date', 'total'])
        df = df.set_index(['date', 'dataset'])
        df = df.drop('total', axis=1)

    return df


def plot_total_number_imported(stats_file, plot_title):
    """
    Creates plot with overview of imports numbers per date.

    :param stats_file: pandas DataFrame with stats data.
    :param str plot_title: title of the plot.
    :return: Line plot figure within the <div id="_dash-app-content">.
    """
    df_full = get_totals_per_date(stats_file, key='full', import_types=False).sort_index()
    df_partial = get_totals_per_date(stats_file, key='partial', import_types=False).sort_index()

    traces_f = viz.getPlotTraces(df_full, key='full', type='lines')
    traces_p = viz.getPlotTraces(df_partial, key='partial', type='lines')
    traces = traces_f + traces_p
    if len(traces) > 0:
        if type(traces[0]) == list:
            traces = list(chain.from_iterable(traces))
    
    layout = go.Layout(title='', xaxis=dict(title=''), yaxis={'title': 'Number of imports'},
                       legend={'font': {'size': 11}}, margin=go.layout.Margin(l=80, r=40, t=100, b=50),
                       annotations=[dict(text='<b>{}<b>'.format(plot_title), font=dict(family='Arial', size=18),
                       showarrow=False, xref='paper', x=-0.06, xanchor='left', yref='paper', y=1.15, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    return dcc.Graph(id='total imports', figure=fig)


def plot_total_numbers_per_date(stats_file, plot_title):
    """
    Plots number of entities and relationships imported per date, with scaled markers reflecting numbers rations.

    :param stats_file: pandas DataFrame with stats data.
    :param str plot_title: title of the plot.
    :return: Scatter plot figure within the <div id="_dash-app-content">, with scalled markers.
    """
    df_full = get_totals_per_date(stats_file, key='full', import_types=True)
    df_partial = get_totals_per_date(stats_file, key='partial', import_types=True)

    traces_f = viz.getPlotTraces(df_full, key='full', type='scaled markers', div_factor=float(10^1000))
    traces_p = viz.getPlotTraces(df_partial, key='partial', type='scaled markers', div_factor=float(10^1000))
    traces = traces_f + traces_p

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else:
        pass

    layout = go.Layout(title='',
                    xaxis={'showgrid': True},
                    yaxis={'title': 'Imported entities/relationships'},
                    legend={'font': {'size':11}},
                    height=550,
                    margin=go.layout.Margin(l=80, r=40, t=100, b=100),
                    annotations=[dict(text='<b>{}<b>'.format(plot_title), font=dict(family='Arial', size=18),
                    showarrow=False, xref='paper', x=-0.06, xanchor='left', yref='paper', y=1.15, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    return dcc.Graph(id='entities-relationships per date', figure=fig)


def plot_databases_numbers_per_date(stats_file, plot_title, key='full', dropdown=False, dropdown_options='dates'):
    """
    Grouped horizontal barplot showing the number of entities and relationships imported from each biomedical database.

    :param stats_file: pandas DataFrame with stats data.
    :param str plot_title: title of the plot.
    :param str key: use only full or partial imports ('full', 'partial').
    :param bool dropdown: add dropdown menu to figure or not.
    :param str dropdown_options: name of the variables to be used as options in the dropdown menu ('dates', \
                        'databases', 'entities' or 'relationships').
    :return: Horizontal barplot figure within the <div id="_dash-app-content">.
    """
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']
    else:
        
        ('Syntax error')

    dropdown_options = get_databases_entities_relationships(stats_file, key=key, options=dropdown_options)
    data = get_imports_per_database_date(stats)

    traces = []
    for i in dropdown_options.keys():
        df = data.iloc[data.index.get_level_values(0).str.contains(i)].droplevel(0)
        traces.append(viz.getPlotTraces(df, key=key, type='bars', horizontal=True))

    if len(traces) > 0:
        if type(traces[0]) == list:
            traces = list(chain.from_iterable(traces))

    layout = go.Layout(title='', xaxis = {'showgrid':True, 'type':'log','title':'Imported entities/relationships'},
                        legend={'font':{'size':11}}, height=600, margin=go.layout.Margin(l=40,r=40,t=80,b=100),
                        annotations=[dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
                        showarrow=False, xref = 'paper', x=-0.17, xanchor='left', yref = 'paper', y=1.2, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    if dropdown:
        updatemenus = get_dropdown_menu(fig, dropdown_options, add_button=True, equal_traces=True, number_traces=2)
        fig.layout.update(go.Layout(updatemenus = updatemenus))

    names = set([fig['data'][n]['name'] for n,i in enumerate(fig['data'])])
    colors = dict(zip(names, ['red', 'blue', 'green', 'yellow', 'orange']))

    for name in names:
        fig.for_each_trace(lambda trace: trace.update(marker=dict(color=colors[name])), selector=dict(name=name))

    # remove_legend_duplicates(fig) #Removes legend from individual plots.

    return dcc.Graph(id = 'databases imports {}'.format(key), figure = fig)


def plot_import_numbers_per_database(stats_file, plot_title, key='full', subplot_titles = ('',''), colors=True, plots_1='entities', plots_2='relationships', dropdown=True, dropdown_options='databases'):
    """
    Creates plotly multiplot figure with breakdown of imported numbers and size of the respective files, per database and \
    import type (entities or relationships).

    :param stats_file: pandas DataFrame with stats data.
    :param str plot_title: title of the plot.
    :param str key: use only full or partial imports ('full', 'partial').
    :param tuple subplot_titles: title of the subplots (tuple of strings, one for each subplot).
    :param bool colors: define standard colors for entities and for relationships.
    :param str plots_1: name of the variable plotted.
    :param str plots_2: name of the variable plotted.
    :param bool dropdown: add dropdown menu to figure or not.
    :param str dropdown_options: name of the variables to be used as options in the dropdown menu ('dates', \
                        'databases', 'entities' or 'relationships').
    :return: Multi-scatterplot figure within the <div id="_dash-app-content">.
    """
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']
    else:
        print('Syntax error')

    ent = get_databases_entities_relationships(stats_file, key=key, options=plots_1)
    rel = get_databases_entities_relationships(stats_file, key=key, options=plots_2)
    dropdown_options = get_databases_entities_relationships(stats_file, key=key, options=dropdown_options)

    if colors:
        ent_colors = set_colors(ent)
        rel_colors = set_colors(rel)

    fig = tools.make_subplots(2, 2, subplot_titles = subplot_titles, vertical_spacing = 0.18, horizontal_spacing = 0.2)

    for i, j in stats.groupby(['dataset', 'filename']):
        date = pd.Series(str(j['datetime'].sort_values().reset_index(drop=True)[0]))
        j = j.sort_values(['import_id', 'datetime']).drop_duplicates(['dataset', 'import_id', 'filename'], keep='first', inplace=False)
        entities_df = j[j['Import_type'] == 'entity']
        relationships_df = j[j['Import_type'] == 'relationships']

        if not entities_df['Imported_number'].empty:
            fig.append_trace(go.Scattergl(visible=True,
                                                  x=entities_df['datetime'],
                                                  y=entities_df['Imported_number'],
                                                  mode='markers+lines',
                                                  marker = dict(color = ent_colors[i[1]]),
                                                  name=i[1].split('.')[0]),1,1)
            fig.append_trace(go.Scattergl(visible=True,
                                                  x=entities_df['datetime'],
                                                  y=entities_df['file_size'],
                                                  mode='markers+lines',
                                                  marker = dict(color = ent_colors[i[1]]),
                                                  name=i[1].split('.')[0],
                                                  showlegend=False),2,1)

        if not relationships_df['Imported_number'].empty:
            fig.append_trace(go.Scattergl(visible=True,
                                                  x=relationships_df['datetime'],
                                                  y=relationships_df['Imported_number'],
                                                  mode='markers+lines',
                                                  marker = dict(color = rel_colors[i[1]]),
                                                  name=i[1].split('.')[0]),1,2)
            fig.append_trace(go.Scattergl(visible=True,
                                                  x=relationships_df['datetime'],
                                                  y=relationships_df['file_size'],
                                                  mode='markers+lines',
                                                  marker = dict(color = rel_colors[i[1]]),
                                                  name=i[1].split('.')[0],
                                                  showlegend=False),2,2)

    fig.layout.update(go.Layout(legend={'orientation':'v', 'font':{'size':11}},
                                height=700, margin=go.layout.Margin(l=20,r=20,t=150,b=60)))

    annotations = []
    annotations.append(dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
                            showarrow=False, xref = 'paper', x=-0.07, xanchor='left', yref = 'paper', y=1.3, yanchor='top'))
    annotations.append({'font':{'size': 14},'showarrow':False,'text':subplot_titles[0],'x':0.23,'xanchor':'center','xref':'paper','y':1.0,'yanchor':'bottom','yref':'paper'})
    annotations.append({'font':{'size': 14},'showarrow':False,'text':subplot_titles[1],'x':0.78,'xanchor':'center','xref':'paper','y':1.0,'yanchor':'bottom','yref':'paper'})
    annotations.append({'font':{'size': 14},'showarrow':False,'text':subplot_titles[2],'x':0.23,'xanchor':'center','xref':'paper','y':0.44,'yanchor':'bottom','yref':'paper'})
    annotations.append({'font':{'size': 14},'showarrow':False,'text':subplot_titles[3],'x':0.78,'xanchor':'center','xref':'paper','y':0.44,'yanchor':'bottom','yref':'paper'})

    fig.layout['annotations'] = annotations
    fig['layout']['template'] = 'plotly_white'

    if dropdown:
        updatemenus = get_dropdown_menu(fig, dropdown_options, add_button=True, equal_traces=False)
        fig.layout.update(go.Layout(updatemenus = updatemenus))


    return dcc.Graph(id = 'imports-breakdown per database {}'.format(key), figure = fig)
