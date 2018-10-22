from KnowledgeViewer.plots import basicFigures as figure
#Dash
import dash
import dash_core_components as dcc
import dash_html_components as html

#Plotly imports
import plotly.plotly as py
from IPython.display import display
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools

#Data manipulation imports
import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import chain
from collections import defaultdict
from natsort import natsorted, ns


def get_stats_data(filename):
    df = pd.read_hdf(filename)
    df[['Imported_number', 'file_size']] = df[['Imported_number', 'file_size']].apply(pd.to_numeric)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    return df

def get_databases_entities_relationships(stats_file, options='all'):
    mask = (stats_file['Import_type']=='entity')
    mask2 = (stats_file['Import_type']=='relationships')
    ent = list(set(list(zip(stats_file.loc[mask,'filename'], stats_file.loc[mask,'dataset']))))
    rel = list(set(list(zip(stats_file.loc[mask2,'filename'], stats_file.loc[mask2,'dataset']))))
    dat = list(set(list(zip(stats_file['date'], stats_file['dataset']))))

    d_dat = defaultdict(list)
    for k, v in dat: d_dat[k].append(v)
    d_dat = {k:tuple(v) for k, v in d_dat.items()}
    d_dat = dict(natsorted(d_dat.items()))

    d_ent = defaultdict(list)
    for k, v in ent: d_ent[v].append(k)
    d_ent = {k:tuple(v) for k, v in d_ent.items()}
    d_ent = dict(natsorted(d_ent.items()))

    d_rel = defaultdict(list)
    for k, v in rel: d_rel[v].append(k)
    d_rel = {k:tuple(v) for k, v in d_rel.items()}
    d_rel = dict(natsorted(d_rel.items()))

    for i in stats_file['dataset'].unique():
        if i not in d_ent.keys(): d_ent[i] = ''
        if i not in d_rel.keys(): d_rel[i] = ''

    d_dbs_filename = defaultdict(list)
    for k,v in chain(d_ent.items(), d_rel.items()): d_dbs_filename[k].append(v)
    d_dbs_filename = {k:tuple(v) for k, v in d_dbs_filename.items()}
    d_dbs_filename = dict(natsorted(d_dbs_filename.items()))

    if options == 'entities': return d_ent
    if options == 'relationships': return d_rel
    if options == 'all': return d_dbs_filename
    if options == 'dates': return d_dat


def set_colors(dictionary):
    colors = []
    for i in list(chain(*dictionary.values())):
        color = 'rgb' + str(tuple(np.random.choice(range(256), size=3)))
        colors.append((i, color))
    colors = dict(colors)

    return colors


def get_dropdown_menu(fig, options_dict, entities_dict=None, number_traces=2):

    if entities_dict == None:
        list_updatemenus = []
        start = 0
        for n, i in enumerate(options_dict.keys()):
            visible = [False] * len(fig['data'])
            end = start + number_traces
            visible[start:end] = [True] * number_traces
            start += number_traces
            temp_dict = dict(label = str(i),
                             method = 'update',
                             args = [{'visible': visible},
                                     {'title': 'Date: '+i}])
            list_updatemenus.append(temp_dict)

            updatemenus = list([dict(active = -1,
                                 buttons = list_updatemenus,
                                 direction='down',
                                 pad={'r':10, 't':10},
                                 showactive=True,x=-0.3,xanchor='left',y=1.15,yanchor='top'),])

    else:
        list_updatemenus = []

        n_entities = sum(len(v) for v in entities_dict.values())
        start = 0
        start2 = n_entities*2
        for n, i in enumerate(options_dict.keys()):
            visible = [False] * len(fig['data'])
            if len(options_dict[i][0]) or len(options_dict[i][1]) > 0:
                end = start + len(options_dict[i][0])*2
                end2 = start2 + len(options_dict[i][1])*2
                visible[start:end] = [True]*len(options_dict[i][0])*2
                visible[start2:end2] = [True]*len(options_dict[i][1])*2
            else: continue

            start += len(options_dict[i][0])*2
            start2 += len(options_dict[i][1])*2

            temp_dict = dict(label = str(i),
                             method = 'update',
                             args = [{'visible': visible},
                                     {'title': 'Database: '+i}])
            list_updatemenus.append(temp_dict)

            updatemenus = list([dict(active = -1,
                                 buttons = list_updatemenus,
                                 direction='down',
                                 pad={'r':10, 't':10},
                                 showactive=True,x=-0.3,xanchor='left',y=1.15,yanchor='top'),])

    return updatemenus


def get_totals_per_date(stats_file, import_types=False):
    cols = ['date', 'total']
    counts = []
    for i in stats_file['date'].unique():
        count = stats_file[stats_file.date == i]['Imported_number'].sum()
        counts.append((i, count))

    df = pd.DataFrame(counts, columns=cols)
    df = df.set_index('date')

    if import_types:
        cols = ['date', 'type', 'total']
        counts = []
        for i in stats_file['date'].unique():
            for j in stats_file['Import_type'].unique():
                mask = (stats_file.date == i) & (stats_file.Import_type == j)
                total = stats_file.loc[mask, 'Imported_number'].sum()
                counts.append((i,j,total))

        df = pd.DataFrame(counts, columns=cols)
        df = df.pivot(index='date', columns='type', values='total')

    return df

def get_imports_per_database_date(stats_file):
    cols = ['date', 'database', 'entity', 'relationships', 'total']
    counts = []
    for i in stats_file['date'].unique():
        for j in stats_file['dataset'].unique():
            #comment this line when imports are only once a week
            df = stats_file[stats_file['date'] == i].sort_values('datetime', ascending=False).drop_duplicates(subset=['dataset', 'name'], keep='first')
            mask_ent = (df['dataset'] == j) & (df['Import_type'] == 'entity')
            mask_rel = (df['dataset'] == j) & (df['Import_type'] == 'relationships')
            ent = df.loc[mask_ent, 'Imported_number'].sum()
            rel = df.loc[mask_rel, 'Imported_number'].sum()
            total = ent + rel
            counts.append((i,j,ent, rel, total))

    df = pd.DataFrame(counts, columns = cols)
    df = df.sort_values(['date','total'])
    df = df.set_index(['date', 'database'])
    df = df.drop('total', axis=1)

    return df


def plot_total_number_imported(stats_file, plot_title):
    data = get_totals_per_date(stats_file, import_types=False)
    traces = figure.getPlotTraces(data, type = 'lines')

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else: pass

    layout = go.Layout(title='', yaxis={'title':'Imported entities/relationships'}, height=500,
    annotations=[dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
    showarrow=False, xref = 'paper', x=-0.06, xanchor='left', yref = 'paper', y=1.2, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)

    return dcc.Graph(id = 'total imports', figure = fig)


def plot_total_numbers_per_date(stats_file, plot_title):
    data = get_totals_per_date(stats_file, import_types=True)
    traces = figure.getPlotTraces(data, type = 'scaled markers', div_factor=float(10^10000))

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else: pass

    layout = go.Layout(title='', xaxis={'showgrid':True}, yaxis={'title':'Imported entities/relationships'}, height=500, margin=go.Margin(b=100, pad=4),
    annotations=[dict(text='<b>{}<b>'.format(plot_title), font=dict(family='Arial', size = 18),
    showarrow=False, xref='paper', x=-0.06, xanchor='left', yref='paper', y=1.2, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)

    return dcc.Graph(id = 'entities-relationships per date', figure = fig)


def plot_databases_numbers_per_date(stats_file, plot_title, dropdown=False, dropdown_options='dates'):
    dropdown_options = get_databases_entities_relationships(stats_file, options=dropdown_options)
    data = get_imports_per_database_date(stats_file)

    traces = []
    for i in dropdown_options.keys():
        df = data.xs(i, level='date')
        traces.append(figure.getPlotTraces(df, type = 'bars', horizontal=True))

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else: pass

    layout = go.Layout(title = '', xaxis = {'title':'Imported entities/relationships'}, height=600, margin=go.Margin(b=100, pad=6),
    annotations=[dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
    showarrow=False, xref = 'paper', x=-0.31, xanchor='left', yref = 'paper', y=1.23, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)

    if dropdown:
        updatemenus = get_dropdown_menu(fig, dropdown_options, entities_dict=None, number_traces=2)
        fig.layout.update(go.Layout(updatemenus = updatemenus))

    return dcc.Graph(id = 'databases total imports', figure = fig)


def plot_import_numbers_per_database(stats_file, plot_title, subplot_titles = ('',''), colors=True, colors_1='entities', colors_2='relationships', dropdown=False, dropdown_options='all'):

    ent = get_databases_entities_relationships(stats_file, options=colors_1)
    rel = get_databases_entities_relationships(stats_file, options=colors_2)
    dropdown_options = get_databases_entities_relationships(stats_file, options=dropdown_options)

    fig = tools.make_subplots(2, 2, subplot_titles = subplot_titles)

    if colors:
        ent_colors = set_colors(ent)
        rel_colors = set_colors(rel)

        for database in dropdown_options.keys():
            df = stats_file[(stats_file['dataset'] == database) & (stats_file['Import_type'] == 'entity')]
            for entity in df.filename.unique():
                mask = (df.filename == entity)
                fig.append_trace(go.Scatter(visible = False,
                                        x=df.loc[mask, 'date'],
                                        y=df.loc[mask, 'Imported_number'],
                                        mode='markers+lines',
                                        marker = dict(color = ent_colors[entity]),
                                        name=entity.split('.')[0], legendgroup='entities'),1,1)
                fig.append_trace(go.Scatter(visible = False,
                                        x=df.loc[mask, 'date'],
                                        y=df.loc[mask, 'file_size'],
                                        mode='markers+lines',
                                        marker = dict(color = ent_colors[entity]),
                                        name=entity.split('.')[0], legendgroup='entities', showlegend=False),2,1)

        for database in dropdown_options.keys():
            df = stats_file[(stats_file['dataset'] == database) & (stats_file['Import_type'] == 'relationships')]
            for relationship in df.filename.unique():
                mask = (df.filename == relationship)
                fig.append_trace(go.Scatter(visible = False,
                                        x=df.loc[mask, 'date'],
                                        y=df.loc[mask, 'Imported_number'],
                                        mode='markers+lines',
                                        marker = dict(color = rel_colors[relationship]),
                                        name=relationship.split('.')[0], legendgroup='relationships'),1,2)
                fig.append_trace(go.Scatter(visible = False,
                                        x=df.loc[mask, 'date'],
                                        y=df.loc[mask, 'file_size'],
                                        mode='markers+lines',
                                        marker = dict(color = rel_colors[relationship]),
                                        name=relationship.split('.')[0], legendgroup='relationships', showlegend=False),2,2)


    fig.layout.update(go.Layout(title='', legend={'orientation':'v'}, height=600,
    annotations=[dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
    showarrow=False, xref = 'paper', x=-0.31, xanchor='left', yref = 'paper', y=1.23, yanchor='top')]))

    if dropdown:
        updatemenus = get_dropdown_menu(fig, dropdown_options, entities_dict=ent)
        fig.layout.update(go.Layout(updatemenus = updatemenus))

    return dcc.Graph(id = 'imports-breakdown per database', figure = fig)
