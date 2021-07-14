import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from passlib.hash import bcrypt
from ckg import ckg_utils
from ckg.graphdb_connector import connector
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder.users import users_controller as uh


try:
    ckg_config = ckg_utils.read_ckg_config()
    log_config = ckg_config['graphdb_builder_log']
    logger = builder_utils.setup_logging(log_config, key='user_creation')
    uconfig = builder_utils.setup_config('users')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


def create_user_from_dict(driver, data):
    """
    Creates graph database node for new user and adds properties to the node.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :param dict data: dictionary with the user information).
    """
    query_name_node = 'create_user_node'
    result = None
    
    user_id = get_new_user_id(driver)
    if 'ID' in data and data['ID'] is None:
        data['ID'] = user_id
    elif 'ID' not in data:
        data['ID'] = user_id

    cypher = uh.get_user_creation_queries()
    query = cypher[query_name_node]['query']
    for q in query.split(';')[0:-1]:
        try:
            result = connector.commitQuery(driver, q+';', parameters=data)
            logger.info("New user node created: {}. Result: {}".format(data['username'], result))
            print("New user node created: {}. Result: {}".format(data['username'], result))
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_node, sys.exc_info(), fname, exc_tb.tb_lineno, err))
            print("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name_node, sys.exc_info(), fname, exc_tb.tb_lineno, err))
    return result


def create_user_node(driver, data):
    """
    Creates graph database node for new user and adds respective properties to node.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param Series data: pandas Series with new user identifier and required user information (see set_arguments()).
    """
    result = create_user_from_dict(driver, data.to_dict())

    return result


def create_user_from_command_line(args, expiration):
    """
    Creates new user in the graph database and corresponding node, from a terminal window (command line), \
    and adds the new user information to the users excel and import files. Arguments as in set_arguments().

    :param args: object. Contains all the parameters neccessary to create a user ('username', 'name', 'email', \
                'secondary_email', 'phone_number' and 'affiliation').
    :type args: any object with __dict__ attribute
    :param int expiration: number of days users is given access.

    .. note:: This function can be used directly with *python create_user_from_command_line.py -u username \
                -n user_name -e email -s secondary_email -p phone_number -a affiliation* .
    """
    usersImportDirectory = ckg_config['imports_users_directory']
    usersFile = os.path.join(usersImportDirectory, uconfig['usersFile'])

    builder_utils.checkDirectory(usersImportDirectory)
    import_file = os.path.join(usersImportDirectory, uconfig['import_file'])

    data = vars(args)
    df = pd.DataFrame.from_dict(data, orient='index').T.drop('file', axis=1)
    create_user(df, import_file, expiration)

    if os.path.exists(usersFile):
        excel = pd.read_excel(usersFile, index=0)
        excel = excel.append(data, ignore_index=True)
        excel.to_excel(usersFile, index=False)
    else:
        df.to_excel(usersFile, index=False)


def create_user_from_file(filepath, expiration):
    """
    Creates new user in the graph database and corresponding node, from an excel file. \
    Rows in the file must be users, and columns must follow set_arguments() fields.

    :param str filepath: filepath and filename containing users information.
    :param str output_file: path to output csv file.
    :param int expiration: number of days users is given access.

    .. note:: This function can be used directly with *python create_user_from_file.py -f path_to_file* .
    """
    usersImportDirectory = ckg_config['imports_users_directory']
    usersFile = os.path.join(usersImportDirectory, uconfig['usersFile'])

    builder_utils.checkDirectory(usersImportDirectory)
    import_file = os.path.join(usersImportDirectory, uconfig['import_file'])

    data = vars(args)
    data = pd.read_excel(data['file']).applymap(str)
    create_user(data, import_file, expiration)

    if os.path.exists(usersFile):
        excel = pd.read_excel(usersFile, index=0)
        excel = excel.append(data.drop('file', axis=1), ignore_index=True)
        excel.to_excel(usersFile, index=False)
    else:
        data.to_excel(usersFile, index=False)


def validate_user(driver, username, email):
    username_found = uh.check_if_node_exists(driver, 'username', username)
    email_found = uh.check_if_node_exists(driver, 'email', email)

    return not username_found.empty or not email_found.empty


def get_new_user_id(driver):
    user_identifier = uh.get_new_user_identifier(driver)
    if user_identifier is None:
        user_identifier = 'U1'

    return user_identifier


def create_user(data, output_file, expiration=365):
    """
    Creates new user in the graph database and corresponding node, through the following steps:

        1. Checks if a user with given properties already exists in the database. If not:
        2. Generates new user identifier
        3. Creates new local user (access to graph database)
        4. Creates new user node
        5. Saves data to users.tsv

    :param data: pandas dataframe with users as rows and arguments and columns.
    :param str output_file: path to output csv file.
    :param int expiration: number of days users is given access.
    :return: Writes relevant .tsv file for the users in data.
    """
    driver = connector.getGraphDatabaseConnectionConfiguration(database=None)
    date = datetime.today() + timedelta(days=expiration)
    df = []

    try:
        for index, row in data.iterrows():
            found = validate_user(driver, row['username'], row['email'])
            if not found:
                user_identifier = get_new_user_id(driver)
                row['ID'] = user_identifier
                row['acronym'] = ''.join([c for c in row['name'] if c.isupper()])
                row['rolename'] = 'reader'
                row['expiration_date'] = date.strftime('%Y-%m-%d')
                row['image'] = ''
                uh.create_db_user(driver, row)
                row['password'] = bcrypt.hash(row['password'])
                create_user_node(driver, row)
                df.append(row)
            else:
                print("User already in the database. Check username and email address.")
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Creating users: file: {},line: {}, error: {}".format(fname, exc_tb.tb_lineno, err))

    if len(df) > 0:
        data = pd.DataFrame(df)
        data['phone_number'] = data['phone_number'].str.split('.').str[0]
        data = data[['ID', 'acronym', 'name', 'username', 'password', 'email', 'secondary_email', 'phone_number', 'affiliation', 'expiration_date', 'rolename', 'image']]
        uh.GenerateGraphFiles(data, output_file)


def set_arguments():
    """
    This function sets the arguments to be used as input for **create_user.py** in the command line.
    """
    parser = argparse.ArgumentParser('Use an excel file (multiple new users) or the function arguments (one new user at a time) to create new users in the database')
    parser.add_argument('-f', '--file', help='define path to file with users creation information', type=str, required=False)
    parser.add_argument('-u', '--username', help='define the username to be created', type=str, required=True)
    parser.add_argument('-d', '--password', help='define the user password', type=str, required=False)
    parser.add_argument('-n', '--name', help='define the name of the user', type=str, required=True)
    parser.add_argument('-e', '--email', help='define the email of the user being created', type=str, required=False)
    parser.add_argument('-s', '--secondary_email', help='define an alternative email for the user', type=str, required=False)
    parser.add_argument('-p', '--phone_number', help='define a phone number where the user can be reached', type=str, required=True)
    parser.add_argument('-a', '--affiliation', help="define the user's affiliation (University, Group)", type=str, required=True)
    parser.add_argument('-i', '--image', help='define path to a picture of the user', type=str, default='', required=False)

    return parser


if __name__ == "__main__":
    parser = set_arguments()
    args = parser.parse_args()

    if args.file is None and args.username is None:
        print('Please specify a file path or use the function arguments. See help (--help).')

    if args.file is not None:
        logger.info('Creating users, from file, in the database')
        print('Creating users, from file, in the database')
        create_user_from_file(args.file, expiration=365)
        print('Done')

    if args.file is None and args.username is not None:
        logger.info('Creating user in the database')
        print('Creating user in the database')
        create_user_from_command_line(args, expiration=365)
        print('Done')
