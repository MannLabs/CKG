import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from ckg import ckg_utils
from ckg.graphdb_connector import connector
from ckg.analytics_core.viz import viz
from ckg.analytics_core import utils

def size_converter(value):
    """
    Converts a given value to the highest possible unit, maintaining two decimals.

    :param int or float value:
    :return: String with converted value and units.
    """
    unit = 'KB'
    val = np.round(value*0.001, 2)
    if len(str(val).split('.')[0]) > 3:
        unit = 'MB'
        val = np.round(val*0.001, 2)
        if len(str(val).split('.')[0]) > 3:
            unit = 'GB'
            val = np.round(val*0.001, 2)
    return str(val)+' '+unit


def get_query():
    """
       Reads the YAML file containing the queries relevant for graph database stats, parses the given stream and \
       returns a Python object (dict[dict]).

    :return: Nested dictionary.
    """
    try:
        queries_path = "../queries/dbstats_cypher.yml"
        directory = os.path.dirname(os.path.abspath(__file__))
        data_upload_cypher = ckg_utils.get_queries(os.path.join(directory, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise Exception("Erro: {}. Reading queries from file {}: {}, file: {},line: {}".format(err, queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
    return data_upload_cypher


def get_db_schema():
    """
    Retrieves the database schema

    :return: network with all the database nodes and how they are related
    """
    style = [{'selector': 'node',
            'style': {'label': 'data(name)',
                      'background-color': 'data(color)',
                      'text-valign': 'center',
                      'text-halign': 'center',
                      'border-color': 'gray',
                      'border-width': '1px',
                      'width': 55,
                      'height': 55,
                      'opacity': 0.8,
                      'font-size': '14'}},
           {'selector': 'edge',
            'style': {'label': 'data(label)',
                     'curve-style': 'bezier',
                     'opacity': 0.7,
                     'width': 0.4,
                     'font-size': '5'}}]
    layout = {'name': 'cose',
              'idealEdgeLength': 100,
              'nodeOverlap': 20,
              'refresh': 20,
              'randomize': False,
              'componentSpacing': 100,
              'nodeRepulsion': 400000,
              'edgeElasticity': 100,
              'nestingFactor': 5,
              'gravity': 80,
              'numIter': 1000,
              'initialTemp': 200,
              'coolingFactor': 0.95,
              'minTemp': 1.0}

    query_name = 'db_schema'
    cypher = get_query()
    driver = connector.getGraphDatabaseConnectionConfiguration()
    
    if driver is not None:
        if query_name in cypher:
            if 'query' in cypher[query_name]:
                query = cypher[query_name]['query']
                try:
                    path = connector.sendQuery(driver, query, parameters={})
                    G = utils.neo4j_schema_to_networkx(path)
                    args = {'height': '1000px'}
                    args['stylesheet'] = style
                    args['layout'] = layout
                    args['title'] = "Database Schema"
                    net, mouseover = utils.networkx_to_cytoscape(G)
                    plot = viz.get_cytoscape_network(net, "db_schema", args)
                except Exception as err:
                    plot = html.Div(children=html.H1("Error accessing the database statistics", className='error_msg'))
        else:
            plot = html.Div(children=html.H1("Error: Cypher query {} for accessing the database statistics does not exist".format(query_name), className='error_msg'))
    else:
        plot = html.Div(children=html.H1("Database is offline", className='error_msg'))

    return plot


def get_db_stats_data():
    """
    Retrieves all the stats data from the graph database and returns them as a dictionary.

    :return: Dictionary of dataframes.
    """
    query_names = ['unique_projects', 'get_db_stats',
                   'get_db_store_size', 'get_db_transactions', 'get_db_kernel']
    df_names = ['projects', 'meta_stats',
                'store_size', 'transactions', 'kernel_monitor']

    dfs = {}
    cypher = get_query()
    driver = connector.getGraphDatabaseConnectionConfiguration()
    
    if driver is not None:
        for i, j in zip(df_names, query_names):
            query = cypher[j]['query']
            try:
                data = connector.getCursorData(driver, query)
                if i == 'store_size':
                    data = data.T
                    data['size'] = [size_converter(i) for i in data[0]]
                dfs[i] = data.to_json(orient='records')
            except Exception:
                pass
    return dfs


def plot_store_size_components(dfs, title, args):
    """
    Plots the store size of different components of the graph database, as a Pie Chart.

    :param dict dfs: dictionary of json objects.
    :param str title: title of the Dash div where plot is located.
    :param dict args: see below.
    :Arguments:
        * **valueCol** (str) -- name of the column with the values to be plotted.
        * **textCol** (str) -- name of the column containing information for the hoverinfo parameter.
        * **height** (str) -- height of the plot.
        * **width** (str) -- width of the plot.
    :return: New Dash div containing title and pie chart.
    """
    fig = None
    if len(dfs) > 0:
        if 'store_size' in dfs:
            data = pd.read_json(dfs['store_size'], orient='records')
            data.index = ['Array store', 'Logical Log', 'Node store', 'Property store',
                        'Relationship store', 'String store', 'Total store size']
            data.columns = ['value', 'size']
            data = data.iloc[:-1]
            fig = viz.get_pieplot(data, identifier='store_size_pie', args=args)

    return html.Div([html.H3(title), fig], style={'margin': '0%', 'padding': '0%'})


def plot_node_rel_per_label(dfs, title, args, focus='nodes'):
    """
    Plots the number of nodes or relationships (depending on 'focus') per label, contained in the \
    grapha database.

    :param dict dfs: dictionary of json objects.
    :param str title: title of the Dash div where plot is located.
    :paeam str focus: plot number of nodes per label ('nodes') or the number of relationships \
                                            per type ('relationships').
    :return: New Dash div containing title and barplot.
    """
    fig = None
    if len(dfs) > 0:
        if 'meta_stats' in dfs:
            data = pd.read_json(dfs['meta_stats'], orient='records')
            if focus == 'nodes':
                data = pd.DataFrame.from_dict(data['labels'][0], orient='index', columns=[
                                            'number']).reset_index()
            elif focus == 'relationships':
                data = pd.DataFrame.from_dict(
                    data['relTypesCount'][0], orient='index', columns=['number']).reset_index()

            data = data.sort_values('number')
            
            if not data.empty:
                fig = viz.get_barplot(data, identifier='node_rel_per_label_{}'.format(focus), args=args)
                fig.figure['layout'] = go.Layout(barmode='relative',
                                                height=args['height'],
                                                xaxis={'type': 'log', 'range': [0, np.log10(data['number'].iloc[-1])]},
                                                yaxis={'showline': True, 'linewidth': 1, 'linecolor': 'black'},
                                                font={'family': 'MyriadPro-Regular', 'size': 12},
                                                template='plotly_white',
                                                bargap=0.2)

    return html.Div([html.H3(title), fig], style={'margin': '0%', 'padding': '0%'})


def indicator(color, text, id_value):
    """
    Builds a new Dash div styled as a container, with borders and background.

    :param str color: background color of the container (RGB or Hex colors).
    :param str text: name to be plotted inside the container.
    :param str id_value: identifier of the container.
    :return: Dash div containing title and an html.P element.
    """
    return html.Div([html.H4(id=id_value),
                     html.P(text)], style={'border-radius': '5px',
                                           'background-color': '#f9f9f9',
                                                            'margin': '0.3%',
                                                            'padding': '1%',
                                                            'position': 'relative',
                                                            'box-shadow': '2px 2px 2px lightgrey',
                                                            'width': '19%',
                                                            # 'height': '15%',
                                                            # 'width':'230px',
                                                            'height': '140px',
                                                            'display': 'inline-block',
                                                            'vertical-align': 'middle'})


def quick_numbers_panel():
    """
    Creates a panel of Dash containers where an overviem of the graph database numbers can be plotted.

    :return: List of Dash components.
    """
    project_ids = []
    project_links = [html.H4('No available Projects')]
    try:
        driver = connector.getGraphDatabaseConnectionConfiguration()
    
        if driver is not None:
            projects = connector.find_nodes(driver, node_type='Project', parameters={})
            for project in projects:
                project_ids.append((project['n']['name'], project['n']['id']))
            project_links = [html.H4('Available Projects:')]
    except Exception:
        pass

    for project_name, project_id in project_ids:
        project_links.append(html.A(project_name.title(),
                                    id='link-internal',
                                    href='/apps/project?project_id={}&force=0'.format(project_id),
                                    target='',
                                    n_clicks=0,
                                    className="button_link"))

    project_dropdown = [html.H6('Project finder:'),
                        dcc.Dropdown(id='project_option',
                                     options=[{'label': name, 'value': (name, value)} for name, value in project_ids],
                                     value='',
                                     multi=False,
                                     clearable=True,
                                     placeholder='Search...',
                                     style={'width': '50%'}),
                        html.H4('', id='project_url')]

    navigation_links = [html.H4('Navigate to:'),
                        html.A("Database Imports", href="/apps/imports", className="nav_link"),
                        html.A("Project Creation", href="/apps/projectCreationApp", className="nav_link"),
                        html.A("Data Upload", href="/apps/dataUploadApp", className="nav_link"),
                        html.A("Admin", href="/apps/admin", className="nav_link")]

    layout = [html.Div(children=navigation_links),
              html.Div(children=project_links[0:5]),
              html.Div(children=project_dropdown),
              html.Div(children=get_db_schema()),
              dcc.Store(id='db_stats_df', data=get_db_stats_data()),
              html.Div(id='db-creation-date'),
              html.Br(),
              html.H3('Overview'),
              html.Div(children=[indicator("#EF553B", "No. of Entities", "db_indicator_1"),
                                 indicator("#EF553B", "No. of Labels",
                                           "db_indicator_2"),
                                 indicator(
                                     "#EF553B", "No. of Relationships", "db_indicator_3"),
                                 indicator(
                                     "#EF553B", "No. of Relationship Types", "db_indicator_4"),
                                 indicator("#EF553B", "No. of Property Keys", "db_indicator_5")]),
              html.Div(children=[indicator("#EF553B", "Entities store", "db_indicator_6"),
                                 indicator(
                                     "#EF553B", "Relationships store", "db_indicator_7"),
                                 indicator("#EF553B", "Property store",
                                           "db_indicator_8"),
                                 indicator("#EF553B", "String store",
                                           "db_indicator_9"),
                                 indicator("#EF553B", "Array store", "db_indicator_10")]),
              html.Div(children=[indicator("#EF553B", "Logical Log size", "db_indicator_11"),
                                 indicator(
                                     "#EF553B", "No. of Transactions (opened)", "db_indicator_12"),
                                 indicator(
                                     "#EF553B", "No. of Transactions (committed)", "db_indicator_13"),
                                 indicator("#EF553B", "No. of Projects", "db_indicator_14")]),
              html.Br(),
              html.Br()
              ]

    return layout
