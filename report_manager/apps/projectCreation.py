import os
import sys
import re
import pandas as pd
import numpy as np
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder import builder_utils
from report_manager import queries
import logging
import logging.config

log_config = ckg_config.report_manager_log
logger = builder_utils.setup_logging(log_config, key="project_creation")

cwd = os.path.abspath(os.path.dirname(__file__))

def get_project_creation_queries():
    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "../queries/project_creation_cypher.yml"
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
        external_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    return external_identifier

def get_new_subject_identifier(driver, projectId):
    query_name = 'increment_subject_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        subject_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    return subject_identifier

def create_new_project(driver, projectId, data):
    query_name = 'create_project'
    external_identifier='No Identifier Assigned'

    try:
        external_identifier = get_new_project_identifier(driver, projectId)
        data['external_id'] = external_identifier
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                for parameters in data.to_dict(orient='records'):
                    result = connector.getCursorData(driver, q+';', parameters=parameters)
            else:
                result = connector.getCursorData(driver, q+';')

        subjects = create_new_subject(driver, external_identifier, data['subjects'][0])

        if data['timepoints'][0] == '':
            pass
        else:
            timepoints = create_new_timepoint(driver, external_identifier, data['timepoints'][0])

        if data['intervention'][0] == '':
            pass
        else:
            intervention_rel = create_intervention_relationship(driver, external_identifier, data['intervention'][0])

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    

    store_new_project_as_file(external_identifier, data, '')
    store_project_relationships_as_file(external_identifier, data['responsible'][0].split(','), [external_identifier], 'IS_RESPONSIBLE', '_responsibles')
    store_project_relationships_as_file(external_identifier, data['participant'][0].split(','), [external_identifier], 'PARTICIPATES_IN', '_participants')
    store_project_relationships_as_file(external_identifier, [external_identifier], data['disease'][0].split(','), 'STUDIES_DISEASE', '_studies_disease')
    store_project_relationships_as_file(external_identifier, [external_identifier], data['tissue'][0].split(','), 'STUDIES_TISSUE', '_studies_tissue')
    store_project_relationships_as_file(external_identifier, [external_identifier], data['intervention'][0].split(','), 'STUDIES_INTERVENTION', '_studies_intervention')

    return result.values[0], external_identifier


def create_new_subject(driver, projectId, subjects):
    data = pd.DataFrame(index=np.arange(1), columns=np.arange(subjects))
    query_name = 'create_subjects'
    subject_identifier='No Identifier Assigned'
    try:
        subject_identifier = get_new_subject_identifier(driver, projectId)
        #Creates dataframe with sequential subject numbers
        number = int(subject_identifier.split('S')[1])
        for i in data.columns:
            data[i] = number
            number += 1
        data = data.T
        data[0] = 'S' + data[0].astype(str)

        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                done = 0
                for index, row in data.iterrows():
                    result = connector.getCursorData(driver, q+';', parameters={'subject_id': str(row[0]), 'external_id': str(projectId)})
                    done += 1
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    #Add node and relationship type to dataframe and store as csv file
    data.insert(loc=0, column='', value=projectId)
    data.columns = ['START_ID', 'END_ID']
    data['TYPE'] = 'HAS_ENROLLED'
    store_new_project_as_file(projectId, data, '_project')

    return done

def create_new_timepoint(driver, projectId, timepoints):
    query_name = 'create_timepoint'
    timepoints = timepoints.replace(' ', '').split(',')
    data = []
    for i in timepoints:
        value = int(''.join(filter(str.isdigit, i)))
        units = re.sub('[^a-zA-Z]+', '', i)
        data.append({'ID': value, 'units': units})  

    data = pd.DataFrame(data)
    data['type'] = 'Timepoint'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                done = 0
                for index, row in data.iterrows():
                    result = connector.getCursorData(driver, q+';', parameters={'timepoint': str(row['ID']), 'units': str(row['units'])})
                    done += 1
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    store_new_project_as_file(projectId, data, '_timepoint')

    return done


def create_intervention_relationship(driver, projectId, intervention):
    query_name = 'create_intervention_relationship'
    interventions = intervention.split(',')
    
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                for i in interventions:
                    result = connector.getCursorData(driver, q+';', parameters={'external_id': projectId, 'intervention': i})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))

    return result.values[0][0]


def store_new_project_as_file(identifier, data, name):
    if data is not None:
        importDir = os.path.join(os.path.join(cwd, '../../../data/imports/experiments'), os.path.join(identifier,'clinical'))
        ckg_utils.checkDirectory(importDir)
        outputfile = os.path.join(importDir, identifier+name+".csv")
        with open(outputfile, 'w') as f:
            data.to_csv(path_or_buf = f,
                        header=True, index=False, quotechar='"',
                        line_terminator='\n', escapechar='\\')


def store_project_relationships_as_file(projectId, start_node_list, end_node_list, relationship, filename):
    length = int(len(max([start_node_list, end_node_list], key=len)))
    data = pd.DataFrame(index=np.arange(length), columns=['START_ID', 'END_ID', 'TYPE'])
    if len(start_node_list) == len(data.index):
        data['START_ID'] = start_node_list
    else: data['START_ID'] = start_node_list[0]
    if len(end_node_list) == len(data.index):
        data['END_ID'] = end_node_list
    else: data['END_ID'] = end_node_list[0]
    data['TYPE'] = relationship 
    store_new_project_as_file(projectId, data, filename)



# def retrieve_identifiers_from_database(driver, projectId):
#     #Queries
#     project_identifier = "MATCH (p:Project) WHERE p.id = 'EXTERNALID' RETURN p.internal_id AS result"
#     subject_identifier = "MATCH (s:Subject) WITH max(toInteger(SPLIT(s.id, 'S')[1]))+1 as new_id RETURN SUBSTRING('S',0,1) + new_id AS result"
#     biosample_identifier = "MATCH (b:Biological_sample) WITH max(toInteger(SPLIT(b.id, 'BS')[1]))+1 as new_id RETURN SUBSTRING('BS',0,2) + new_id AS result"
#     anasample_identifier = "MATCH (a:Analytical_sample) WITH max(toInteger(SPLIT(a.id, 'AS')[1]))+1 as new_id RETURN SUBSTRING('AS',0,2) + new_id AS result"

#     #Get external id from database
#     project_id = connector.sendQuery(driver, project_identifier.replace('EXTERNALID', projectId))
#     project_id = [record['result'] for record in project_id][0]
        
#     subject_id = connector.sendQuery(driver, subject_identifier)
#     subject_id = [record['result'] for record in subject_id][0]

#     biosample_id = connector.sendQuery(driver, biosample_identifier)
#     biosample_id = [record['result'] for record in biosample_id][0]

#     anasample_id = connector.sendQuery(driver, anasample_identifier)
#     anasample_id = [record['result'] for record in anasample_id][0]

#     return project_id, subject_id, biosample_id, anasample_id

