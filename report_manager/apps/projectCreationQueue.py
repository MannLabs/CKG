import os
import sys
import pandas as pd
from datetime import datetime
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder.experiments import experiments_controller as eh
from graphdb_builder import builder_utils
from graphdb_builder.builder import loader
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="grapher")
START_TIME = datetime.now()

cwd = os.path.abspath(os.path.dirname(__file__))

# Define project creation cypher queries and builders
#Create create_project_from_app class

def project_app_importer(projectId):
    filepath = os.path.join(cwd, '../../../data/experiments/PROJECTID/clinical/ProjectData.xlsx').replace("PROJECTID", projectId)
    project_data = None
    if os.path.isfile(filepath):
        project_data = eh.readDataset(filepath)

    #Create .csv file in /imports directory
    if project_data is not None:
        dataRows = eh.extractProjectInfo(project_data)
        if dataRows is not None:
            importDir = os.path.join(os.path.join(cwd, '../../../data/imports/experiments'), os.path.join(projectId,'clinical'))
            ckg_utils.checkDirectory(importDir)
            outputfile = os.path.join(importDir, projectId+".csv")
            with open(outputfile, 'w') as f:
                dataRows.to_csv(path_or_buf = f,
                            header=True, index=False, quotechar='"',
                            line_terminator='\n', escapechar='\\')


def project_app_loader(driver, projectId):
    #Queries for external id and project creation
    create_external_id = "MATCH (p:Project) WITH toInteger(SPLIT(max(p.id), 'P')[1])+1 AS new_external_id, SIZE(SPLIT(max(p.id), 'P')[1]) AS length, SIZE(toString(toInteger(SPLIT(max(p.id), 'P')[1])+1)) AS new_length RETURN SUBSTRING('P', 0, 1) + SUBSTRING('00000000000', 0, length-new_length) + new_external_id AS Project_external_id;"
    create_project = "CREATE CONSTRAINT ON (p:Project) ASSERT p.internal_id IS UNIQUE;CREATE CONSTRAINT ON (p:Project) ASSERT p.name IS UNIQUE;USING PERIODIC COMMIT 10000 LOAD CSV WITH HEADERS FROM 'file:///IMPORTDIR/PROJECTID.csv' AS line MERGE (p:Project {internal_id:line.internal_id}) ON CREATE SET p.external_id='EXTERNALID',p.name=line.name,p.acronym=line.acronym,p.description=line.description,p.type=line.type,p.tissue=line.tissue,p.responsible=line.responsible,p.participant=line.participant,p.start_date=line.start_date,p.end_date=line.end_date,p.status=line.status RETURN COUNT(p) AS PROJECTID_project;"

    #Load entities into database  
    importDir = os.path.join(cwd, '../../../data/imports/experiments')
    projectDir = os.path.join(importDir, projectId)
    projectDir = os.path.join(projectDir, 'clinical')

    #Get external id from database
    result = connector.sendQuery(driver, create_external_id)
    project_external_id = [record['Project_external_id'] for record in result][0]
    
    #Load project data into database
    queries = []
    logger.info("Loading {} into the database".format('project'))

    try:
        queries.extend(create_project.replace("IMPORTDIR", projectDir).replace('PROJECTID', projectId).replace('EXTERNALID', project_external_id).split(';')[0:-1])
        loader.load_into_database(driver, queries, 'project')
    except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Loading: {}: {}, file: {}, line: {}".format('project', err, fname, exc_tb.tb_lineno))
    
    

           



# # Add to the index.py to start the queueing
# from rq import Queue
# from redis import Redis

# q = Queue(connection=Redis())

# q.enqueue(projectCreationQueue.project_app_importer, projectId)
# q.enqueue(projectCreationQueue.project_app_loader, driver, projectId)

# external_id = q.dequeue()
# project = q.dequeue()





# class EventQueue:
#     def __init__(self):
#         self._queue = self.Queue()

#     def enqueue(self, func, args = [], kwargs = {}, highPriority = False):
#         element = self.packCall(func, args, kwargs)
#         return self._queue.enqueue(element, highPriority)

#     def packCall(self, func, args = [], kwargs = {}):
#         return (func, args, kwargs)

#     class Queue():
#         def __init__(self):
#             self._list = []

#         def enqueue(self, element, highPriority):
#             if element not in self._list:
#                 if highPriority:
#                     self._list.insert(0, element)
#                 else:
#                     self._list.append(element)
#                 return True
#             return False
        
#         # def hasMore(self):
#         #     return len(self._list) > 0
 
#         def dequeue(self):
#             if len(self._list) > 0:
#                 return self._list.pop(0)
#             return ("Queue Empty!")

    