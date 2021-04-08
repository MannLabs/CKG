import os
import sys
from ckg import ckg_utils
from ckg.graphdb_connector import connector
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder.builder import loader
from ckg.graphdb_builder.experiments import experiments_controller as eh

try:
    ckg_config = ckg_utils.read_ckg_config()
    log_config = ckg_config['report_manager_log']
    logger = builder_utils.setup_logging(log_config, key="project_creation")
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


def get_project_creation_queries():
    """
    Reads the YAML file containing the queries relevant to user creation, parses the given stream and \
    returns a Python object (dict[dict]).

    :return: Nested dictionary.
    """
    try:
        directory = os.path.dirname(os.path.abspath(__file__))
        queries_path = "../queries/project_creation_cypher.yml"
        project_creation_cypher = ckg_utils.get_queries(os.path.join(directory, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading queries from file {}: {}, file: {},line: {}, err: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))
    return project_creation_cypher


def check_if_node_exists(driver, node, node_property, value):
    """
    Queries the graph database and checks if a node with a specific property and property value already exists.
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str node: node to be matched in the database.
    :param str node_property: property of the node.
    :param value: property value.
    :type value: str, int, float or bool
    :return: Pandas dataframe with user identifier if User with node_property and value already exists, \
            if User does not exist, returns and empty dataframe.
    """
    query_name = 'check_node'
    try:
        cypher = get_project_creation_queries()
        query = cypher[query_name]['query'].replace('NODE', node).replace('PROPERTY', node_property)
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'value': value})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))

    return result


def get_new_project_identifier(driver, projectId):
    """
    Queries the database for the last project external identifier and returns a new sequential identifier.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :return: Project external identifier.
    :rtype: str
    """
    query_name = 'increment_project_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        last_project, new_id = connector.getCursorData(driver, query).values[0]
        if last_project is None and new_id is None:
            external_identifier = 'P0000001'
        else:
            length = len(last_project.split('P')[-1])
            new_length = len(str(new_id))
            external_identifier = 'P'+'0'*(length-new_length)+str(new_id)
    except Exception as err:
        external_identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}, err: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))

    return external_identifier


def get_subject_number_in_project(driver, projectId):
    """
    Extracts the number of subjects included in a given project.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str projectId: external project identifier (from the graph database).
    :return: Integer with the number of subjects.
    """
    query_name = 'subject_number'
    try:
        cypher = get_project_creation_queries()
        query = cypher[query_name]['query']
        result = connector.getCursorData(driver, query, parameters={'external_id': projectId}).values[0][0]
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Reading query {}: {}, file: {},line: {}".format(err, query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return result


def create_new_project(driver, projectId, data, separator='|'):
    """
    Creates a new project in the graph database, following the steps:

    1. Retrieves new project external identifier and creates project node and relationships in the graph database.
    2. Creates subjects, timepoints and intervention nodes.
    3. Saves all the entities and relationships to tab-delimited files.
    4. Returns the number of projects created and the project external identifier.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :param data: pandas Dataframe with project as row and other attributes as columns.
    :param str separator: character used to separate multiple entries in a project attribute.
    :return: Two strings: number of projects created and the project external identifier.
    """
    query_name = 'create_project'
    external_identifier = None
    done = None
    try:
        db_project = check_if_node_exists(driver, 'Project', 'name', data['name'][0])
        if db_project.empty:
            external_identifier = get_new_project_identifier(driver, projectId)
            if external_identifier is None:
                external_identifier = 'P0000001'
            data['external_id'] = external_identifier

            projectDir = os.path.join(ckg_config['experiments_directory'], os.path.join(external_identifier, 'project'))
            ckg_utils.checkDirectory(projectDir)
            data.to_excel(os.path.join(projectDir, 'ProjectData_{}.xlsx'.format(external_identifier)), index=False, encoding='utf-8')

            datasetPath = os.path.join(os.path.join(ckg_config['imports_experiments_directory'], external_identifier), 'project')
            ckg_utils.checkDirectory(datasetPath)
            eh.generate_dataset_imports(external_identifier, 'project', datasetPath)
            loader.partialUpdate(imports=['project'], specific=[external_identifier])
            done = 1
        else:
            done = 0
            external_identifier = ''
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}, err: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))
    return done, external_identifier
