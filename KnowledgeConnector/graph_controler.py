import sys
import py2neo
import config_graph as config


def getGraphDatabaseConnectionConfiguration():
    host = config.dbURL
    port = config.dbPort
    user = config.dbUser
    password = config.dbPassword

    driver = connectToDB(host, port, user, password)

    return driver

def connectToDB(host="localhost", port=7687, user="neo4j", password="password"):
    driver = py2neo.Graph(host=host, port=port, user=user, password=password)

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
    try:
        result = driver.run(query)
    except py2neo.database.ClientError as err:
        print("Error: {}".format(err))
    except py2neo.GraphError as err:
        print("Error: {}".format(err))
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return result
