import sys
import py2neo
import pandas as pd
from KnowledgeConnector import graph_config as config
import ckgError

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
        raise ckgError.DatabaseError("Database failed to service the request. {}".format(err))
    except py2neo.database.ClientError as err:
        raise ckgError.ClientError("The client sent a bad request. {}".format(err))
    except py2neo.GraphError as err:
        raise ckgError.GraphError("{}".format(err))
    except py2neo.database.TransientError as err:
        raise ckgError.TransientError("Database cannot service the request right now. {}".format(err))
    except:
        raise ckgError.Error("Unexpected error:", sys.exc_info()[0])

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
        raise ckgError.DatabaseError("Database failed to service the request. {}".format(err))
    except py2neo.database.ClientError as err:
        raise ckgError.ClientError("The client sent a bad request. {}".format(err))
    except py2neo.GraphError as err:
        raise ckgError.GraphError("{}".format(err))
    except py2neo.database.TransientError as err:
        raise ckgError.TransientError("Database cannot service the request right now. {}".format(err))
    except:
        raise ckgError.Error("Unexpected error: {}".format(sys.exc_info()[0]))

    return result

def getCursorData(driver, query):
    result = sendQuery(driver, query)
    df = pd.DataFrame(result.data())
    
    return df
