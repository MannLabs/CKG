from report_manager.plots import basicFigures as figure
#Dash
import dash
import dash_core_components as dcc
import dash_html_components as html

#Plotly imports
import chart_studio.plotly as py
from IPython.display import display
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.subplots as tools

#Data manipulation imports
import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import chain
from collections import defaultdict
from natsort import natsorted, ns

def get_stats_data(filename, n=3):
    store = pd.HDFStore(filename, 'r')
    full, partial = list(store.keys())
    df_full = store[full]
    df_partial = store[partial]
    store.close()

    df_full['Import_flag'] = 'full'
    df_partial['Import_flag'] = 'partial'
    df = pd.concat([df_full, df_partial])
    df['datetime'] = pd.to_datetime(df['date']+' '+df['time'])
    imp = select_last_n_imports(df, n=n)
    df = df[df['import_id'].isin(imp)].reset_index(drop=True)
    return df

def select_last_n_imports(stats_file, n=3):
    df = stats_file[['datetime', 'import_id', 'Import_flag']].sort_values('datetime').drop_duplicates(['import_id'], keep = 'first', inplace = False) 
    f = df[df['Import_flag'] == 'full']
    f = f.iloc[:n,1].tolist()
    p = df[df['Import_flag'] == 'partial']
    p = p.iloc[:n,1].tolist()
    return p+f

def get_databases_entities_relationships(stats_file, key='full', options='databases'):
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']
    elif key == 'all':
        stats = stats_file

    mask = (stats['Import_type']=='entity')
    mask2 = (stats['Import_type']=='relationships')
    ent = list(set(list(zip(stats.loc[mask,'filename'], stats.loc[mask,'dataset']))))
    rel = list(set(list(zip(stats.loc[mask2,'filename'], stats.loc[mask2,'dataset']))))
    # dat = list(set(list(zip(stats['date'].apply(str).str.split(' ').str[0], stats['dataset']))))
    dat = []
    for i, j in stats.groupby('import_id'):
        date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
        for i in j['dataset'].unique():
            dat.append((date, i))

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
    if options == 'databases': return d_dbs_filename
    if options == 'dates': return d_dat

def set_colors(dictionary):
    colors = []
    for i in list(chain(*dictionary.values())):
        color = 'rgb' + str(tuple(np.random.choice(range(256), size=3)))
        colors.append((i, color))
    colors = dict(colors)

    return colors

def get_dropdown_menu(fig, options_dict, add_button=True, entities_dict=None, number_traces=2):
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

        if add_button:
            button = [dict(label = 'All',
                            method = 'update',
                            args = [{'visible': [True]*len(fig['data'])}, {'title': 'All'}])]
            list_updatemenus = list_updatemenus + button
        else: pass

        updatemenus = list([dict(active = len(list_updatemenus)-1,
                                 buttons = list_updatemenus,
                                 direction='down',
                                 #pad={'r':10, 't':10},
                                 showactive=True,x=-0.17,xanchor='left',y=1.1,yanchor='top'),])

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

        if add_button:
            button = [dict(label = 'All',
                            method = 'update',
                            args = [{'visible': [True]*len(fig['data'])}, {'title': 'All'}])]
            list_updatemenus = list_updatemenus + button
        else: pass

        updatemenus = list([dict(active = len(list_updatemenus)-1,
                                 buttons = list_updatemenus,
                                 direction='down',
                                 #pad={'r':10, 't':10},
                                 showactive=True,x=-0.07,xanchor='left',y=1.2,yanchor='top'),])

    return updatemenus

def get_totals_per_date(stats_file, key='full', import_types=False):
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']

    cols = ['date', 'total']
    counts = []
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
    cols = ['date', 'dataset', 'entities', 'relationships', 'total']
    stats_sum = []
    for i, j in stats_file.groupby(['import_id']):
        date = str(j['datetime'].sort_values().reset_index(drop=True)[0])
        for a, b in j.groupby('dataset'):
            ent = b.loc[(b['Import_type'] == 'entity'), 'Imported_number'].sum()
            rel = b.loc[(b['Import_type'] == 'relationships'), 'Imported_number'].sum()
            total = b['Imported_number'].sum()
            stats_sum.append((date, a, ent, rel, total))

    df = pd.DataFrame(stats_sum, columns=cols)
    df = df.sort_values(['date','total'])
    df = df.set_index(['date', 'dataset'])
    df = df.drop('total', axis=1)

    return df

def plot_total_number_imported(stats_file, plot_title):
    df_full = get_totals_per_date(stats_file, key='full', import_types=False).sort_index()
    df_partial = get_totals_per_date(stats_file, key='partial', import_types=False).sort_index()

    traces_f = figure.getPlotTraces(df_full, key='full', type='lines')
    traces_p = figure.getPlotTraces(df_partial, key='partial', type='lines')
    traces = traces_f + traces_p

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else: pass

    layout = go.Layout(title='', xaxis=dict(title=''), yaxis={'title':'Number of imports'},
                       legend={'font':{'size':11}}, margin=go.layout.Margin(l=80,r=40,t=100,b=50),
                       annotations=[dict(text='<b>{}<b>'.format(plot_title), font=dict(family='Arial', size = 18),
                       showarrow=False, xref='paper', x=-0.06, xanchor='left', yref='paper', y=1.15, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    return dcc.Graph(id = 'total imports', figure = fig)

def plot_total_numbers_per_date(stats_file, plot_title):
    df_full = get_totals_per_date(stats_file, key='full', import_types=True)
    df_partial = get_totals_per_date(stats_file, key='partial', import_types=True)

    traces_f = figure.getPlotTraces(df_full, key='full', type='scaled markers', div_factor=float(10^1000))
    traces_p = figure.getPlotTraces(df_partial, key='partial', type='scaled markers', div_factor=float(10^1000))
    traces = traces_f + traces_p

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else: pass

    layout = go.Layout(title='', 
                    xaxis={'showgrid':True}, 
                    yaxis={'title':'Imported entities/relationships'},
                    legend={'font':{'size':11}}, 
                    height=550, 
                    margin=go.layout.Margin(l=80,r=40,t=100,b=100),
                    annotations=[dict(text='<b>{}<b>'.format(plot_title), font=dict(family='Arial', size = 18),
                        showarrow=False, xref='paper', x=-0.06, xanchor='left', yref='paper', y=1.15, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    return dcc.Graph(id = 'entities-relationships per date', figure = fig)

def plot_databases_numbers_per_date(stats_file, plot_title, key='full', dropdown=False, dropdown_options='dates'):
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']
    else:
        print('Syntax error')

    dropdown_options = get_databases_entities_relationships(stats_file, key=key, options=dropdown_options)
    data = get_imports_per_database_date(stats)

    traces = []
    for i in dropdown_options.keys():
        df = data.iloc[data.index.get_level_values(0).str.contains(i)].droplevel(0)
        traces.append(figure.getPlotTraces(df, key=key, type = 'bars', horizontal=True))

    if type(traces[0]) == list:
        traces = list(chain.from_iterable(traces))
    else:
        pass

    layout = go.Layout(title = '', xaxis = {'showgrid':True, 'type':'log','title':'Imported entities/relationships'},
                        legend={'font':{'size':11}}, height=600, margin=go.layout.Margin(l=40,r=40,t=80,b=100),
                        annotations=[dict(text='<b>{}<b>'.format(plot_title), font = dict(family='Arial', size = 18),
                        showarrow=False, xref = 'paper', x=-0.17, xanchor='left', yref = 'paper', y=1.2, yanchor='top')])

    fig = go.Figure(data=traces, layout=layout)
    fig['layout']['template'] = 'plotly_white'

    if dropdown:
        updatemenus = get_dropdown_menu(fig, dropdown_options, add_button=True, entities_dict=None, number_traces=2)
        fig.layout.update(go.Layout(updatemenus = updatemenus))
        
    names = set([fig['data'][n]['name'] for n,i in enumerate(fig['data'])])
    colors = dict(zip(names, ['red', 'blue', 'green', 'yellow', 'orange']))
    # colors = {}
    # for name in names:
    #     color = 'rgb' + str(tuple(np.random.choice(range(256), size=3)))
    #     colors[name] = color

    for name in names:
        fig.for_each_trace(lambda trace: trace.update(marker=dict(color=colors[name])), selector=dict(name=name))

    return dcc.Graph(id = 'databases total imports {}'.format(key), figure = fig)


def plot_import_numbers_per_database(stats_file, plot_title, key='full', subplot_titles = ('',''), colors=True, color1='entities', color2='relationships', dropdown=True, dropdown_options='databases'):
    if key == 'full':
        stats = stats_file[stats_file['Import_flag'] == 'full']
    elif key == 'partial':
        stats = stats_file[stats_file['Import_flag'] == 'partial']
    else:
        print('Syntax error')

    ent = get_databases_entities_relationships(stats_file, key=key, options=color1)
    rel = get_databases_entities_relationships(stats_file, key=key, options=color2)
    dropdown_options = get_databases_entities_relationships(stats_file, key=key, options=dropdown_options)

    if colors:
        ent_colors = set_colors(ent)
        rel_colors = set_colors(rel)

    fig = tools.make_subplots(2, 2, subplot_titles = subplot_titles, vertical_spacing = 0.18, horizontal_spacing = 0.2)

    for i, j in stats.groupby('import_id'):
        date = pd.Series(str(j['datetime'].sort_values().reset_index(drop=True)[0]))
        j = j[j['Import_type'] == 'entity']
        for a, b in j.groupby('dataset'):
            for file in b['filename']:
                mask = (b['filename'] == file)
                fig.append_trace(go.Scatter(visible=True,
                                            x=date,
                                            y=b.loc[mask, 'Imported_number'],
                                            mode='markers+lines',
                                            marker = dict(color = ent_colors[file]),
                                            name=file.split('.')[0]),1,1)
                fig.append_trace(go.Scatter(visible=True,
                                            x=date,
                                            y=b.loc[mask, 'file_size'],
                                            mode='markers+lines',
                                            marker = dict(color = ent_colors[file]),
                                            name=file.split('.')[0],
                                            showlegend=False),1,2)
    for i, j in stats.groupby('import_id'):
        date = pd.Series(str(j['datetime'].sort_values().reset_index(drop=True)[0]))
        j = j[j['Import_type'] == 'relationships']
        for a, b in j.groupby('dataset'):
            for file in b['filename']:
                mask = (b['filename'] == file)
                fig.append_trace(go.Scatter(visible=True,
                                            x=date,
                                            y=b.loc[mask, 'Imported_number'],
                                            mode='markers+lines',
                                            marker = dict(color = rel_colors[file]),
                                            name=file.split('.')[0]),2,1)
                fig.append_trace(go.Scatter(visible=True,
                                            x=date,
                                            y=b.loc[mask, 'file_size'],
                                            mode='markers+lines',
                                            marker = dict(color = rel_colors[file]),
                                            name=file.split('.')[0],
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
        updatemenus = get_dropdown_menu(fig, dropdown_options, add_button=True, entities_dict=ent)
        fig.layout.update(go.Layout(updatemenus = updatemenus))
            

    return dcc.Graph(id = 'imports-breakdown per database {}'.format(key), figure = fig)
