import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from passlib.hash import bcrypt
from ckg import ckg_utils
from ckg.graphdb_connector import connector
from ckg.graphdb_builder import builder_utils



try:
    ckg_config = ckg_utils.read_ckg_config()
    cwd = os.path.dirname(os.path.abspath(__file__))
    config = builder_utils.setup_config('users')
    log_config = ckg_config['graphdb_builder_log']
    logger = builder_utils.setup_logging(log_config, key='users_controller')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))




def parseUsersFile(expiration=365):
    """
    Creates new user in the graph database and corresponding node, through the following steps:

        1. Generates new user identifier
        2. Checks if a user with given properties already exists in the database. If not:
        3. Creates new local user (access to graph database)
        4. Saves data to tab-delimited file.

    :param int expiration: number of days a user is given access.
    :return: Writes relevant .tsv file for the users in the provided file.
    """
    usersDir = ckg_config['users_directory']
    usersFile = os.path.join(usersDir, config['usersFile'])
    usersImportDir = ckg_config['imports_users_directory']
    usersImportFile = os.path.join(usersImportDir, config['import_file'])

    driver = connector.getGraphDatabaseConnectionConfiguration(database=None)

    data = pd.read_excel(usersFile).applymap(str)
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
                row['rolename'] = 'reader'
                row['expiration_date'] = date.strftime('%Y-%m-%d')
                row['image'] = ''
                #create_db_user(driver, row)
                row['password'] = bcrypt.encrypt(row['password'])
                df.append(row)
                new_id += 1

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Extracting users info: {}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno))
    if len(df) > 0:
        data = pd.DataFrame(df)
        data['phone_number'] = data['phone_number'].str.split('.').str[0]
        data = data[['ID', 'acronym', 'name', 'username', 'password', 'email', 'secondary_email', 'phone_number', 'affiliation', 'expiration_date', 'rolename', 'image']]
        GenerateGraphFiles(data, usersImportFile)

def get_user_creation_queries():
    """
    Reads the YAML file containing the queries relevant to user creation, parses the given stream and \
    returns a Python object (dict[dict]).
    """
    try:
        queries_path = config['cypher_queries_file']
        user_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading queries from file {}: {}, file: {},line: {}, error: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))
    return user_creation_cypher

def get_new_user_identifier(driver):
    """
    Queries the database for the last user identifier and returns a new sequential identifier.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
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

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str node_property: property of the node.
    :param value: property value.
    :type value: str, int, float or bool
    :return: Pandas dataframe with user identifier if User with node_property and value already exists, \
            if User does not exist, returns and empty dataframe.
    """
    query_name = 'check_node'
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

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
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

def GenerateGraphFiles(data, output_file):
    """
    Saves pandas dataframe to users.tsv.
    If file already exists, appends new lines. \
    Else, creates file and writes dataframe to it.

    :param data: pandas dataframe to be written to .tsv file.
    :param str output_file: path to output csv file.
    """

    if os.path.exists(output_file):
        with open(output_file, 'a') as f:
            data.to_csv(path_or_buf = f, sep='\t',
                        header=False, index=False, quotechar='"',
                        line_terminator='\n', escapechar='\\')
    else:
        with open(output_file, 'w') as f:
            data.to_csv(path_or_buf = f, sep='\t',
                        header=True, index=False, quotechar='"',
                        line_terminator='\n', escapechar='\\')


if __name__ == "__main__":
    pass
