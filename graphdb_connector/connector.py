import sys
import os
import py2neo
import pandas as pd
import ckg_utils
from config import ckg_config
from graphdb_builder import builder_utils

import logging
import logging.config

log_config = ckg_config.graphdb_connector_log
logger = builder_utils.setup_logging(log_config, key="connector")

try:
    cwd = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(cwd, ckg_config.connector_config_file)
    config = ckg_utils.get_configuration(path)
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

def getGraphDatabaseConnectionConfiguration(configuration=None):
    if configuration is None:
        configuration = config
    host = configuration['db_url']
    port = configuration['db_port']
    user = configuration['db_user']
    password = configuration['db_password']

    driver = connectToDB(host, port, user, password)

    return driver

def connectToDB(host="localhost", port=7687, user="neo4j", password="password"):
    try:
        driver = py2neo.Graph(host=host, port=port, user=user, password=password)
    except py2neo.database.DatabaseError as err:
        raise py2neo.database.DatabaseError("Database failed to service the request. {}".format(err))
    except py2neo.database.ClientError as err:
        raise py2neo.ClientError("The client sent a bad request. {}".format(err))
    except py2neo.database.TransientError as err:
        raise py2neo.TransientError("Database cannot service the request right now. {}".format(err))
    except py2neo.GraphError as err:
        raise py2neo.GraphError("{}".format(err))
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

    driver = connector.getGraphDatabaseConnectionConfiguration()
    entity, entityid, attribute, value = parameters

    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "./queries.yml"
        project_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
        for query_name in project_cypher:
            title = query_name.lower().replace('_',' ')
            if title == 'modify':
                query = project_cypher[query_name]['query'] % (entity, entityid, attribute, value)
                sendQuery(driver, query)
                print("Property successfully modified")
    except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))


def sendQuery(driver, query, parameters={}):
    #print(query)
    result = None
    try:
        result = driver.run(query, parameters)
    except py2neo.database.DatabaseError as err:
        raise py2neo.database.DatabaseError("Database failed to service the request. {}".format(err))
    except py2neo.database.ClientError as err:
        raise py2neo.ClientError("The client sent a bad request. {}".format(err))
    except py2neo.GraphError as err:
        raise py2neo.GraphError("{}".format(err))
    except py2neo.database.TransientError as err:
        raise py2neo.TransientError("Database cannot service the request right now. {}".format(err))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exc_tb.tb_lineno)
        raise Exception("Unexpected error:{}.\n{}".format(err, sys_error))

    return result

def getCursorData(driver, query, parameters={}):
    result = sendQuery(driver, query, parameters)
    df = pd.DataFrame(result.data())

    return df
