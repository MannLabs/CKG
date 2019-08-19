import os
import sys
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
	try:
		queries_path = 'user_creation_cypher.yml'
		user_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading queries from file {}: {}, file: {},line: {}, error: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return user_creation_cypher

def get_new_user_identifier(driver):
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
	query_name = 'find_user'
	try:
		user_creation_cypher = get_user_creation_queries()
		query = user_creation_cypher[query_name]['query'].replace('PROPERTY', node_property)
		for q in query.split(';')[0:-1]:
			if '$' in q:
				result = connector.getCursorData(driver, q+';', parameters={'value':str(value)}).values
			else:
				result = connector.getCursorData(driver, q+';').values
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return result

def create_user_db(driver, args):
	query_name_add = 'create_user'
	query_name_role = 'add_role_to_user'
	try:
		cypher = get_user_creation_queries()
		query = cypher[query_name_add]['query'] + cypher[query_name_role]['query']
		for q in query.split(';')[0:-1]:
			if '$' in q:
				result = connector.getCursorData(driver, q+';', parameters=args)
			else:
				result = connector.getCursorData(driver, q+';')
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno, err))
	return 'Done'

def create_user_from_command_line(args, expiration=365):
	data = vars(args)
	result = create_user(data, expiration)

	return result


def create_user_from_file(filepath, expiration):
    data = pd.read_excel(filepath).applymap(str)
    result = create_user(data, expiration)
    
    return result

def create_user(data, expiration=365):
	query_name_add = 'create_db_user'
	query_name_role = 'add_role_to_db_user'
	query_name_node = 'create_user_node'
	query_list_db_users =  'list_db_users'

	driver = connector.getGraphDatabaseConnectionConfiguration()
	date = datetime.today() + timedelta(days=expiration)
	df = []
	done = 0

	try:
		cypher = get_user_creation_queries()
		db_query = cypher[query_name_add]['query'] + cypher[query_name_role]['query']
		for index, row in data.iterrows():
			username = check_if_node_exists(driver, 'username', row['username'])
			name = check_if_node_exists(driver, 'name', row['name'])
			email = check_if_node_exists(driver, 'email', row['email'])
			if len(username) == 0 and len(name) == 0 and len(email) == 0:
				user_id = get_new_user_identifier(driver)
				if user_id is None:
					user_id = 'U1'
				row['ID'] = user_id
				row['acronym'] = ''.join([c for c in row['name'] if c.isupper()])
				row['password'] = row['username']
				row['rolename'] = 'reader'
				row['expiration_date'] = date.strftime('%Y-%m-%d')
				row['image'] = ''
				
				for q in cypher[query_name_node]['query'].split(';')[0:-1]:
					parameters = {}
					if '$' in q:
						parameters = row.to_dict()
					result = connector.getCursorData(driver, q+';', parameters=parameters)
				logger.info("New user created: {}. Result: {}".format(row['username'], result))
				df.append(row)
				done +=1
				db_users = connector.getCursorData(driver, cypher[query_list_db_users]['query'],{})
				if row['username'] not in db_users['username'].to_list() or db_users.empty:
					print("IN")
					for q in db_query.split(';')[0:-1]:
						result = connector.getCursorData(driver, q+';', parameters=row.to_dict())
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno, err))

	if len(df) > 0:
		#Save new user to file
		usersDir = os.path.join(cwd,'../../../data/imports/users')
		ifile = os.path.join(usersDir, 'users.tsv')
		data = pd.DataFrame(df)
		data = data[['ID', 'acronym', 'name', 'username', 'email', 'secondary_email', 'phone_number', 'affiliation', 'expiration_date', 'rolename', 'image']]

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
	return done
    

def set_arguments():
	parser = argparse.ArgumentParser('Use an excel file (multiple new users) or the function arguments (one new user at a time) to create new users in the database')
	parser.add_argument('-f', '--file', help='define path to file with users creation information', type=str, required=False)
	parser.add_argument('-u', '--username', help='define the username to be created', type=str, required=False)
	parser.add_argument('-n', '--name', help='define the name of the user', type=str, required=False)
	parser.add_argument('-e', '--email', help='define the email of the user being created', type=str, required=False)
	parser.add_argument('-s', '--second_email', help='define an alternative email for the user', type=str, required=False)
	parser.add_argument('-p', '--phone', help='define a phone number where the user can be reached', type=str, required=False)
	parser.add_argument('-a', '--affiliation', help="define the user's affiliation (University, Group)", type=str, required=False)
	parser.add_argument('-i', '--image', help='define path to a picture of the user', type=str, required=False)

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
		create_user_from_command_line(driver, args, expiration=365)
		print('Done')



