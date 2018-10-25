import sys
import py2neo
import pandas as pd
from KnowledgeConnector import graph_config as config

def getGraphDatabaseConnectionConfiguration():
    host = config.dbURL
    port = config.dbPort
    user = config.dbUser
    password = config.dbPassword

    driver = connectToDB(host, port, user, password)

    return driver

def connectToDB(host="localhost", port=7687, user="neo4j", password="password"):
    try:
        driver = py2neo.Graph(host=host, port=port, user=user, password=password)
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
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exec_tb.tb_lineno)
        raise Exception("Unexpected error:{}.\n{}".format(err, sys_err))

    return driver

def removeRelationshipDB(entity1, entity2, relationship):
    driver = getGraphDatabaseConnectionConfiguration()

    countCy = cy.COUNT_RELATIONSHIPS
    deleteCy = cy.REMOVE_RELATIONSHIPS
    countst = countCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    deletest = deleteCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    print(countst)
    print(sendQuery(driver, countst).data()[0])
    print("Removing %d entries in the database" % sendQuery(driver, countst).data()[0]['count'])
    sendQuery(driver, deletest)
    print("Existing entries after deletion: %d" % sendQuery(driver, countst).data()[0]['count'])

def sendQuery(driver, query):
    #print(query)
    result = None
    try:
        result = driver.run(query)
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
        sys_error = "{}, file: {},line: {}".format(sys.exc_info(), fname, exec_tb.tb_lineno)
        raise Exception("Unexpected error:{}.\n{}".format(err, sys_err))

    return result

def getCursorData(driver, query):
    result = sendQuery(driver, query)
    df = pd.DataFrame(result.data())
    
    return df
