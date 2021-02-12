import sys
import os
import json
import neo4j
import pandas as pd
import ckg.ckg_utils
from ckg.config import ckg_config
from ckg.graphdb_builder import builder_utils

log_config = ckg_config.graphdb_connector_log
logger = builder_utils.setup_logging(log_config, key="connector")

try:
    cwd = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(cwd, ckg_config.connector_config_file)
    config = ckg_utils.get_configuration(path)
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


def getGraphDatabaseConnectionConfiguration(configuration=None, database=None):
    if configuration is None:
        configuration = config # TODO this will fail if this function is imported
    host = configuration['db_url']
    port = configuration['db_port']
    user = configuration['db_user']
    password = configuration['db_password']

    if database is not None:
        host = host+'/'+database

    driver = connectToDB(host, port, user, password)

    return driver


def connectToDB(host="localhost", port=7687, user="neo4j", password="password"):
    try:
        uri = "bolt://{}:{}".format(host, port)
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno)
        raise Exception("Unexpected error:{}.\n{}".format(err, sys_error))

    return driver


def removeRelationshipDB(entity1, entity2, relationship):
    driver = getGraphDatabaseConnectionConfiguration()
    countCy = cy.COUNT_RELATIONSHIPS
    deleteCy = cy.REMOVE_RELATIONSHIPS
    countst = countCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    deletest = deleteCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    print("Removing %d entries in the database" % sendQuery(driver, countst).data()[0]['count'])
    sendQuery(driver, deletest)
    print("Existing entries after deletion: %d" % sendQuery(driver, countst).data()[0]['count'])


def modifyEntityProperty(parameters):
    '''parameters: tuple with entity name, entity id, property name to modify, and value'''

    driver = getGraphDatabaseConnectionConfiguration()
    entity, entityid, attribute, value = parameters

    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "./queries.yml"
        project_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
        for query_name in project_cypher:
            title = query_name.lower().replace('_', ' ')
            if title == 'modify':
                query = project_cypher[query_name]['query'] % (entity, entityid, attribute, value)
                sendQuery(driver, query)
                print("Property successfully modified")
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Reading queries from file {}: {}, file: {},line: {}".format(err, queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))


def do_cypher_tx(tx, cypher, parameters):
    result = tx.run(cypher, **parameters)
    values = result.data()
    return values


def commitQuery(driver, query, parameters={}):
    result = None
    try:
        with driver.session() as session:
            result = session.run(query, parameters)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno)
        raise Exception("Connection error:{}.\n{}".format(err, sys_error))

    return result


def sendQuery(driver, query, parameters={}):
    result = None
    try:
        with driver.session() as session:
            result = session.read_transaction(do_cypher_tx, query, parameters)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno)
        raise Exception("Connection error:{}.\n{}".format(err, sys_error))

    return result


def getCursorData(driver, query, parameters={}):
    result = sendQuery(driver, query, parameters)
    df = pd.DataFrame(result)

    return df


def find_node(driver, node_type, parameters={}):
    query = "MATCH (n:TYPE) WHERE RETURN n".replace('TYPE', node_type)
    where_clause = ''
    if len(parameters) > 0:
        where_clause = "WHERE "+'AND '.join(["n.{}='{}'".format(k,v) for k, v in parameters.items()])
    query = query.replace("WHERE", where_clause)
    result = sendQuery(driver, query)
    result = result.pop()['n']

    return result


def find_nodes(driver, node_type, parameters={}):
    query = "MATCH (n:TYPE) WHERE RETURN n".replace('TYPE', node_type)
    where_clause = ''
    if len(parameters) > 0:
        where_clause = "WHERE "+'AND '.join(["n.{}='{}'".format(k,v) for k, v in parameters.items()])
    query = query.replace("WHERE", where_clause)
    result = sendQuery(driver, query)

    return result


def run_query(query, parameters={}):
    driver = getGraphDatabaseConnectionConfiguration(configuration=None, database=None)
    data = getCursorData(driver, query, parameters=parameters)

    return data


def generate_virtual_graph(graph_json):
    query = "CALL apoc.graph.fromDocument('JSON', {write:false}) YIELD graph RETURN *".replace("JSON", json.dumps(graph_json))
    #driver = getGraphDatabaseConnectionConfiguration()
    #neo4j = sendQuery(driver, query)
    return query
