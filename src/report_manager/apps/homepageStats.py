import os
import sys
import pandas as pd
import numpy as np
import json
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.subplots as tools
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import config.ckg_config as ckg_config
import ckg_utils
import logging
import logging.config
from graphdb_connector import connector
from graphdb_builder import builder_utils
from report_manager.queries import query_utils
from report_manager.plots import basicFigures as figure

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="db_stats")

try:
	cwd = os.path.abspath(os.path.dirname(__file__))
	config = builder_utils.setup_config('experiments')
	driver = connector.getGraphDatabaseConnectionConfiguration()
except Exception as err:
	logger.error("Reading configuration > {}.".format(err))

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
		queries_path = "../queries/project_cypher.yml"
		data_upload_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
	return data_upload_cypher

def get_db_stats_data():
	"""
	Retrieves all the stats data from the graph database and returns them as a dictionary.

	:return: Dictionary of dataframes.
	"""
	query_names = ['unique_projects', 'get_db_stats', 'get_db_store_size', 'get_db_transactions', 'get_db_kernel']
	df_names = ['projects', 'meta_stats', 'store_size', 'transactions', 'kernel_monitor']

	dfs = {}
	cypher = get_query()
	for i,j in zip(df_names, query_names):
		query = cypher[j]['query']
		data = connector.getCursorData(driver, query)
		if i == 'store_size':
			data = data.T
			data['size'] = [size_converter(i) for i in data[0]]
		dfs[i] = data.to_json(orient='records')
	return dfs

def plot_store_size_components(dfs, title):
	"""
	Plots the store size of different components of the graph database, as a Pie Chart.

	:param dict dfs: dictionary of json objects.
	:param str title: title of the Dash div where plot is located.
	:return: New Dash div containing title and pie chart.
	"""
	data = pd.read_json(dfs['store_size'], orient='records')
	data.index = ['Array store', 'Logical Log', 'Node store', 'Property store', 'Relationship store', 'String store', 'Total store size']
	data.columns = ['value', 'size']
	
	fig = go.Figure(data=[go.Pie(labels=data.index[:-1], values=data['value'][:-1], hovertext=data['size'][:-1], hoverinfo='label+text+percent')])
	fig['layout']['template'] = 'plotly_white'

	return html.Div([html.H3(title), dcc.Graph(id = 'store_size_pie', figure = fig)], style={'margin': '0%',
																							'padding': '0%'})
def plot_node_rel_per_label(dfs, title, focus='nodes'):
	"""
	Plots the number of nodes or relationships (depending on 'focus') per label, contained in the \
	grapha database.

	:param dict dfs: dictionary of json objects.
	:param str title: title of the Dash div where plot is located.
	:paeam str focus: plot number of nodes per label ('nodes') or the number of relationships \
						per type ('relationships').
	:return: New Dash div containing title and barplot.
	"""
	data = pd.read_json(dfs['meta_stats'], orient='records')
	if focus == 'nodes':
		data = pd.DataFrame.from_dict(data['labels'][0], orient='index', columns=['number']).reset_index()
		xaxis_name = 'Labels'
	elif focus == 'relationships':
		data = pd.DataFrame.from_dict(data['relTypesCount'][0], orient='index', columns=['number']).reset_index()
		xaxis_name = 'Types'

	# traces = figure.getPlotTraces(df, key='', type='bars')
	# fig = go.Figure(data=traces, layout=layout)
	fig = px.bar(data, x='index', y='number')
	fig.update_layout(xaxis_tickangle=-60, xaxis_title=xaxis_name, yaxis_title='', yaxis_type='log')
	fig['layout']['template'] = 'plotly_white'

	return html.Div([html.H3(title), dcc.Graph(id = 'node_rel_per_label_{}'.format(focus), figure = fig)], style={'margin': '0%',
																												'padding': '0%'})

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
								'vertical-align':'middle'})

def quick_numbers_panel():
	"""
	Creates a panel of Dash containers where an overviem of the graph database numbers can be plotted.

	:return: List of Dash components.
	"""
	project_ids = [(d['id']) for d in driver.nodes.match("Project")]

	layout = [dcc.Store(id='db_stats_df', data=get_db_stats_data()),
			  html.Div(id='db-creation-date'),
			  html.Br(),
			  html.H3('Overview'),
			  html.Div(children=[indicator("#EF553B", "No. of Entities", "db_indicator_1"),
			  					 indicator("#EF553B", "No. of Labels", "db_indicator_2"),
								 indicator("#EF553B", "No. of Relationships", "db_indicator_3"),
								 indicator("#EF553B", "No. of Relationship Types", "db_indicator_4"),
								 indicator("#EF553B", "No. of Property Keys", "db_indicator_5")]),
			  html.Div(children=[indicator("#EF553B", "Entities store", "db_indicator_6"),
								 indicator("#EF553B", "Relationships store", "db_indicator_7"),
								 indicator("#EF553B", "Property store", "db_indicator_8"),
								 indicator("#EF553B", "String store", "db_indicator_9"),
								 indicator("#EF553B", "Array store", "db_indicator_10")]),
			  html.Div(children=[indicator("#EF553B", "Logical Log size", "db_indicator_11"),
								 indicator("#EF553B", "No. of Transactions (opened)", "db_indicator_12"),
								 indicator("#EF553B", "No. of Transactions (committed)", "db_indicator_13"),
								 indicator("#EF553B", "No. of Projects", "db_indicator_14")]),
			  html.Br(),
			  html.Div(children=[html.H4('Project URL finder:'),
			  					 dcc.Dropdown(id='project_option', options=[{'label':i, 'value':i} for i in project_ids], value='', multi=False, clearable=True, placeholder='Search...', style={'width':'50%'}),
			  					 html.H4('',id='project_url')]),
			  html.Br()
			  ]

	return layout
	