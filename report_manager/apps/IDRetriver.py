import os
import sys
from datetime import datetime
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder.experiments import experiments_controller as eh
from graphdb_builder import builder_utils
from graphdb_builder.builder import loader
from report_manager import queries
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="project_creation")
START_TIME = datetime.now()

cwd = os.path.abspath(os.path.dirname(__file__))

# Define project creation cypher queries and builders
#Create create_project_from_app class

def get_project_creation_queries():
    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "queries/project_creation_cypher.yml"
        project_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))

    return project_creation_cypher

def get_new_project_identifier(driver, projectId):
    query_name = 'increment_project_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        external_identifier = connector.getCursorData(driver, query)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    return external_identifier


def create_new_project(driver, projectId, data):
    query_name = 'create_project'
    try:
        external_identifier = get_new_project_identifier(driver, projectId)
        data['external_identifier'] = external_identifier

        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                for parameters in data.to_dict(orient='records'):
                    result = connector.getCursorData(driver, q+';', parameters=parameters)
            else:
                result = connector.getCursorData(driver, q+';', parameters=parameters)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    store_new_project_as_file(external_identifier, data)

    return result

def store_new_project_as_file(identifier, data):
    if data is not None:
        importDir = os.path.join(os.path.join(cwd, '../../../data/imports/experiments'), os.path.join(identifier,'clinical'))
        ckg_utils.checkDirectory(importDir)
        outputfile = os.path.join(importDir, projectId+".csv")
        with open(outputfile, 'w') as f:
            data.to_csv(path_or_buf = f,
                        header=True, index=False, quotechar='"',
                        line_terminator='\n', escapechar='\\')

def retrieve_identifiers_from_database(driver, projectId):
    #Queries
    project_identifier = "MATCH (p:Project) WHERE p.external_id = 'EXTERNALID' RETURN p.internal_id AS result"
    subject_identifier = "MATCH (s:Subject) WITH max(toInteger(SPLIT(s.id, 'S')[1]))+1 as new_id RETURN SUBSTRING('S',0,1) + new_id AS result"
    biosample_identifier = "MATCH (b:Biological_sample) WITH max(toInteger(SPLIT(b.id, 'BS')[1]))+1 as new_id RETURN SUBSTRING('BS',0,2) + new_id AS result"
    anasample_identifier = "MATCH (a:Analytical_sample) WITH max(toInteger(SPLIT(a.id, 'AS')[1]))+1 as new_id RETURN SUBSTRING('AS',0,2) + new_id AS result"

    #Get external id from database
    project_id = connector.sendQuery(driver, project_identifier.replace('EXTERNALID', projectId))
    project_id = [record['result'] for record in project_id][0]
        
    subject_id = connector.sendQuery(driver, subject_identifier)
    subject_id = [record['result'] for record in subject_id][0]

    biosample_id = connector.sendQuery(driver, biosample_identifier)
    biosample_id = [record['result'] for record in biosample_id][0]

    anasample_id = connector.sendQuery(driver, anasample_identifier)
    anasample_id = [record['result'] for record in anasample_id][0]

    return project_id, subject_id, biosample_id, anasample_id
