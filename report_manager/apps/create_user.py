import os
import sys
import argparse
import pandas as pd
import numpy as np
import ckg_utils
import config.ckg_config as ckg_config
from graphdb_connector import connector
from graphdb_builder import builder_utils

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key='user_creation')

try:
    config = ckg_utils.get_configuration(ckg_config.builder_config_file)
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


cwd = os.path.abspath(os.path.dirname(__file__))
driver = connector.getGraphDatabaseConnectionConfiguration()

def get_user_creation_queries():
	try:
		queries_path = '../queries/user_creation_cypher.yml'
		user_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
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
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
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
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
	return result

def create_user_db(driver, args):
	query_name_add = 'create_user'
	query_name_role = 'add_role_to_user'
	try:
		cypher = get_user_creation_queries()
		query = cypher[query_name_add]['query'] + cypher[query_name_role]['query']
		arguments = {'username':args.username, 'password':args.username, 'rolename':'reader'}
		for q in query.split(';')[0:-1]:
			if '$' in q:
				result = connector.getCursorData(driver, q+';', parameters=arguments)
			else:
				result = connector.getCursorData(driver, q+';')
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno))
	return 'Done'

def create_user_node(driver, args):
	query_name = 'create_user_node'
	try:
		user_id = get_new_user_identifier(driver)
		data = vars(args)
		data['ID'] = user_id
		data['acronym'] = ''.join([c for c in data['name'] if c.isupper()])
		user_creation_cypher = get_user_creation_queries()
		query = user_creation_cypher[query_name]['query']
		for q in query.split(';')[0:-1]:
			if '$' in q:
				result = connector.getCursorData(driver, q+';', parameters=data)
			else:
				result = connector.getCursorData(driver, q+';')
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))

	#Save new user to file
	usersDir = os.path.join(os.getcwd(),config["usersDirectory"])
	file = os.path.join(usersDir, 'CKG_users.csv')
	data = pd.DataFrame.from_dict(data, orient='index').T
	data = data[['ID', 'acronym', 'name', 'username', 'email', 'second_email', 'phone', 'affiliation']]
	with open(file, 'a') as f:
		data.to_csv(path_or_buf = f,
                    header=False, index=False, quotechar='"',
                    line_terminator='\n', escapechar='\\')
	return result


def create_user_from_file(driver, args):
	query_name_add = 'create_user'
	query_name_role = 'add_role_to_user'
	query_name_node = 'create_user_node'
	df = []
	done = 0
	try:
		data = pd.read_excel(args.file).applymap(str)
		cypher = get_user_creation_queries()
		query = cypher[query_name_add]['query'] + cypher[query_name_role]['query'] + cypher[query_name_node]['query']
		for index, row in data.iterrows():
			user_id = get_new_user_identifier(driver)
			if user_id is None:
				user_id = 'U1'
			else:
				pass
			username = check_if_node_exists(driver, 'username', row['username'])
			name = check_if_node_exists(driver, 'name', row['name'])
			email = check_if_node_exists(driver, 'email', row['email'])
			if username.size == 0 and name.size == 0 and email.size == 0:
				row['ID'] = user_id
				row['acronym'] = ''.join([c for c in row['name'] if c.isupper()])
				row['password'] = row['username']
				row['rolename'] = 'reader'
				for q in query.split(';')[0:-1]:
					if '$' in q:			
						result = connector.getCursorData(driver, q+';', parameters=row.to_dict())
					else:
						result = connector.getCursorData(driver, q+';')
					done += 1
				df.append(row)
			if username.size != 0:
				print('A user with the same username "{}" already exists. Modify username.'.format(row['username']))
				continue
			if name.size != 0:
				print('A user with the same name "{}" already exists. Modify name.'.format(row['name']))
				continue
			if email.size != 0:
				print('A user with the same email "{}" already exists. Modify email.'.format(row['email']))
				continue
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name_add, sys.exc_info(), fname, exc_tb.tb_lineno))

	#Save new user to file
	usersDir = os.path.join(os.getcwd(),config["usersDirectory"])
	file = os.path.join(usersDir, 'CKG_users.csv')
	data = pd.DataFrame(df)
	data = data[['ID', 'acronym', 'name', 'username', 'email', 'second_email', 'phone', 'affiliation', 'rolename']]
	with open(file, 'a') as f:
		data.to_csv(path_or_buf = f,
                    header=False, index=False, quotechar='"',
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
	# # parser.add_argument('-d', '--expiration_date', help='define the email of the user being created', type=str, required=True)
	# parser.add_argument('-i', '--image', help='define path to a picture of the user', type=str, required=True)

	return parser

if __name__ == "__main__":
	parser =  set_arguments()
	args = parser.parse_args()

	if args.file is None and args.username is None:
		print('Please specify a file path or use the function arguments. See help (--help).')

	if args.file != None:
		logger.info('Creating users, from file, in the database')
		print('Creating users, from file, in the database')
		print(create_user_from_file(driver, args))
		print('Done')

	if args.file is None and args.username != None:
		username = check_if_node_exists(driver, 'username', args.username)
		name = check_if_node_exists(driver, 'name', args.name)
		email = check_if_node_exists(driver, 'email', args.email)

		if username.size != 0:
			print('A user with the same username already exists. Modify username.')
			pass
		if name.size != 0:
			print('A user with the same name already exists. Modify name.')
			pass
		if email.size != 0:
			print('A user with the same email already exists. Modify email.')
			pass
		if username.size == 0 and name.size == 0 and email.size == 0:
			logger.info('Creating user in the database')
			print('Creating user in the database')
			create_user_db(driver, args)
			create_user_node(driver, args)
			print('Done')



