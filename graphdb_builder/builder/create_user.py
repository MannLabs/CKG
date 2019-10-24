import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ckg_utils
import config.ckg_config as ckg_config
from graphdb_connector import connector
from graphdb_builder import builder_utils

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key='user_creation')

try:
	config = builder_utils.setup_config('builder')
except Exception as err:
	logger.error("Reading configuration > {}.".format(err))


cwd = os.path.abspath(os.path.dirname(__file__))
driver = connector.getGraphDatabaseConnectionConfiguration()

def get_user_creation_queries():
	"""
	Reads the YAML file containing the queries relevant to user creation, parses the given stream and \
	returns a Python object (dict[dict]).
	"""
	try:
		queries_path = 'user_creation_cypher.yml'
		user_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading queries from file {}: {}, file: {},line: {}, error: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return user_creation_cypher

def get_new_user_identifier(driver):
	"""
	Queries the database for the last user identifier and returns a new sequential identifier.

	:param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :return: User identifier.
    :rtype: str
	"""
	query_name = 'increment_user_id'
	try:
		user_creation_cypher = get_user_creation_queries()
		query = user_creation_cypher[query_name]['query']
		user_identifier = connector.getCursorData(driver, query).values[0][0]
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return user_identifier

def check_if_node_exists(driver, node_property, value):
	"""
	Queries the graph database and checks if a node with a specific property and property value already exists.

	:param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
	:param str node_property: property of the node.
	:param value: property value.
	:type value: str, int, float or bool
	:return: Pandas dataframe with user identifier if User with node_property and value already exists, \
			if User does not exist, returns and empty dataframe.
	"""
	query_name = 'find_user'
	try:
		user_creation_cypher = get_user_creation_queries()
		query = user_creation_cypher[query_name]['query'].replace('PROPERTY', node_property)
		for q in query.split(';')[0:-1]:
			if '$' in q:
				result = connector.getCursorData(driver, q+';', parameters={'value':value})
			else:
				result = connector.getCursorData(driver, q+';')
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return result

def create_db_user(driver, data):
	"""
	Creates and assigns role to new graph database user, if user not in list of local users.

	:param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
	:param Series data: pandas Series with required user information (see set_arguments()).
	"""
	query_name_add = 'create_db_user'
	query_name_role = 'add_role_to_db_user'
	query_list_db_users =  'list_db_users'

	try:
		cypher = get_user_creation_queries()
		db_query = cypher[query_name_add]['query'] + cypher[query_name_role]['query']
		db_users = connector.getCursorData(driver, cypher[query_list_db_users]['query'],{})
		if data['username'] not in db_users['username'].to_list() or db_users.empty:
			for q in db_query.split(';')[0:-1]:
				result = connector.getCursorData(driver, q+';', parameters=data.to_dict())
			logger.info("New user created: {}. Result: {}".format(data['username'], result))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno, err))

def create_user_node(driver, data):
	"""
	Creates graph database node for new user and adds respective properties to node.

	:param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
	:param Series data: pandas Series with new user identifier and required user information (see set_arguments()).
	"""
	query_name_node = 'create_user_node'
	try:
		cypher = get_user_creation_queries()
		query = cypher[query_name_node]['query']
		for q in query.split(';')[0:-1]:
			result = connector.getCursorData(driver, q+';', parameters=data.to_dict())
		logger.info("New user node created: {}. Result: {}".format(data['username'], result))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno, err))


#To use with builder.py
def extractUsersInfo(filepath, expiration=365):
	"""
	Creates new user in the graph database and corresponding node, through the following steps:
	
		1. Generates new user identifier
		2. Checks if a user with given properties already exists in the database. If not:
		3. Creates new user node
		4. Creates new local user (access to graph database)
		5. Saves data to users.tsv

	:param str filepath: filepath and filename containing users information.
	:param int expiration: number of days a user is given access.
	:return: Writes relevant .tsv file for the users in the provided file.

	.. warning:: This function must be used within *builder.py*.
	"""
	data = pd.read_excel(filepath).applymap(str)
	driver = connector.getGraphDatabaseConnectionConfiguration()
	date = datetime.today() + timedelta(days=expiration)
	df = []
	try:
		user_identifier = get_new_user_identifier(driver)
		if user_identifier is None:
			user_identifier = 'U1'
		new_id = int(re.search('\d+', user_identifier).group())

		for index, row in data.iterrows():
			username = check_if_node_exists(driver, 'username', row['username'])
			name = check_if_node_exists(driver, 'name', row['name'])
			email = check_if_node_exists(driver, 'email', row['email'])
			if username.empty and name.empty and email.empty:
				row['ID'] = 'U{}'.format(new_id)
				row['acronym'] = ''.join([c for c in row['name'] if c.isupper()])
				row['password'] = row['username']
				row['rolename'] = 'reader'
				row['expiration_date'] = date.strftime('%Y-%m-%d')
				row['image'] = ''
				df.append(row)
				create_db_user(driver, row)
			new_id += 1

	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Extracting users info: {}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno))
	if len(df) > 0:
		data = pd.DataFrame(df)
		data = data[['ID', 'acronym', 'name', 'username', 'email', 'secondary_email', 'phone_number', 'affiliation', 'expiration_date', 'rolename', 'image']]
		GenerateGraphFiles(data)


#To use outside builder.py
def create_user_from_command_line(args, expiration):
	"""
	Creates new user in the graph database and corresponding node, from a terminal window (command line). \
	Arguments as in set_arguments().

	:param args: object. Contains all the parameters neccessary to create a user ('username', 'name', 'email', \
				'secondary_email', 'phone_number' and 'affiliation').
	:type args: any object with __dict__ attribute
	:param int expiration: number of days users is given access.

	.. note:: This function can be used directly with *python create_user_from_command_line.py -u username \
				-n user_name -e email -s secondary_email -p phone_number -a affiliation* .
	"""
	data = vars(args)
	data = pd.DataFrame.from_dict(data, orient='index').T
	result = create_user(data, expiration)
	return result


def create_user_from_file(filepath, expiration):
	"""
	Creates new user in the graph database and corresponding node, from an excel file. \
	Rows in the file must be users, and columns must follow set_arguments() fields.

	:param str filepath: filepath and filename containing users information.
	:param int expiration: number of days users is given access.

	.. note:: This function can be used directly with *python create_user_from_file.py -f path_to_file* .
	"""
	data = pd.read_excel(filepath).applymap(str)
	result = create_user(data, expiration)    
	return result

def create_user(data, expiration=365):
	"""
	Creates new user in the graph database and corresponding node, through the following steps:
	
		1. Checks if a user with given properties already exists in the database. If not:
		2. Generates new user identifier
		3. Creates new user node
		4. Creates new local user (access to graph database)
		5. Saves data to users.tsv

	:param data: pandas dataframe with users as rows and arguments and columns.
	:param int expiration: number of days users is given access.
	:return: Writes relevant .tsv file for the users in data.
	"""
	driver = connector.getGraphDatabaseConnectionConfiguration()
	date = datetime.today() + timedelta(days=expiration)
	df = []

	try:
		for index, row in data.iterrows():
			username = check_if_node_exists(driver, 'username', row['username'])
			name = check_if_node_exists(driver, 'name', row['name'])
			email = check_if_node_exists(driver, 'email', row['email'])
			if username.empty and name.empty and email.empty:
				user_identifier = get_new_user_identifier(driver)
				if user_identifier is None:
					user_identifier = 'U1'
				row['ID'] = user_identifier
				row['acronym'] = ''.join([c for c in row['name'] if c.isupper()])
				row['password'] = row['username']
				row['rolename'] = 'reader'
				row['expiration_date'] = date.strftime('%Y-%m-%d')
				row['image'] = ''
				create_user_node(driver, row)
				create_db_user(driver, row)
				df.append(row)
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Creating users: file: {},line: {}, error: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno, err))
	
	if len(df) > 0:
		data = pd.DataFrame(df)
		data = data[['ID', 'acronym', 'name', 'username', 'email', 'secondary_email', 'phone_number', 'affiliation', 'expiration_date', 'rolename', 'image']]
		GenerateGraphFiles(data)

def GenerateGraphFiles(data):
	"""
	Saves pandas dataframe to users.tsv.
	If file already exists, appends new lines. \
	Else, creates file and writes dataframe to it.
	
	:param data: pandas dataframe to be written to .tsv file.
	"""
	importDir = os.path.join(cwd,'../../../data/imports/users')
	ckg_utils.checkDirectory(importDir)
	ifile = os.path.join(importDir, 'users.tsv')

	if os.path.exists(ifile):
		with open(ifile, 'a') as f:
			data.to_csv(path_or_buf = f, sep='\t',
						header=False, index=False, quotechar='"',
						line_terminator='\n', escapechar='\\')
	else:
		with open(ifile, 'w') as f:
			data.to_csv(path_or_buf = f, sep='\t',
						header=True, index=False, quotechar='"',
						line_terminator='\n', escapechar='\\')

def set_arguments():
	"""
    This function sets the arguments to be used as input for **create_user.py** in the command line.
    """
	parser = argparse.ArgumentParser('Use an excel file (multiple new users) or the function arguments (one new user at a time) to create new users in the database')
	parser.add_argument('-f', '--file', help='define path to file with users creation information', type=str, required=False)
	parser.add_argument('-u', '--username', help='define the username to be created', type=str, required=False)
	parser.add_argument('-n', '--name', help='define the name of the user', type=str, required=False)
	parser.add_argument('-e', '--email', help='define the email of the user being created', type=str, required=False)
	parser.add_argument('-s', '--secondary_email', help='define an alternative email for the user', type=str, required=False)
	parser.add_argument('-p', '--phone_number', help='define a phone number where the user can be reached', type=str, required=False)
	parser.add_argument('-a', '--affiliation', help="define the user's affiliation (University, Group)", type=str, required=False)
	parser.add_argument('-i', '--image', help='define path to a picture of the user', type=str, default='', required=False)

	return parser

if __name__ == "__main__":
	parser =  set_arguments()
	args = parser.parse_args()

	if args.file is None and args.username is None:
		print('Please specify a file path or use the function arguments. See help (--help).')

	if args.file != None:
		logger.info('Creating users, from file, in the database')
		print('Creating users, from file, in the database')
		create_user_from_file(args.file, expiration=365)
		print('Done')

	if args.file is None and args.username != None:
		logger.info('Creating user in the database')
		print('Creating user in the database')
		create_user_from_command_line(args, expiration=365)
		print('Done')



