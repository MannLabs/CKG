import os
import sys
import re
import pandas as pd
import numpy as np
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder import builder_utils
from graphdb_builder.experiments import experiments_controller as eh
from report_manager.queries import query_utils
import logging
import logging.config

log_config = ckg_config.report_manager_log
logger = builder_utils.setup_logging(log_config, key="project_creation")

cwd = os.path.abspath(os.path.dirname(__file__))
experimentDir = os.path.join(cwd, '../../../data/experiments')
importDir = os.path.join(cwd, '../../../data/imports/experiments')

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


def get_subjects_in_project(driver, projectId):
    query_name = 'extract_project_subjects'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'external_id': str(projectId)})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return result.values


def create_new_project(driver, projectId, data, separator='|'):
    query_name = 'create_project'
    external_identifier='No Identifier Assigned'
    disease_ids = []
    tissue_ids = []
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
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))

    subjects = create_new_subject(driver, external_identifier, data['subjects'][0])
    if pd.isnull(data['timepoints'][0]):
        pass
    else:
        timepoints = create_new_timepoint(driver, external_identifier, data, separator)
    if pd.isnull(data['intervention'][0]):
        pass
    else:
        interventions = create_intervention_relationship(driver, external_identifier, data, separator)
    
    for disease in data['disease'][0].split(separator):
        disease_ids.append(query_utils.map_node_name_to_id(driver, 'Disease', str(disease)))
    for tissue in data['tissue'][0].split(separator):
        tissue_ids.append(query_utils.map_node_name_to_id(driver, 'Tissue', str(tissue)))

    store_new_project(external_identifier, data, experimentDir, 'xlsx')
    store_as_file(external_identifier, data, external_identifier, importDir, 'tsv')
    store_new_relationships(external_identifier, data['responsible'][0].split(separator), [external_identifier], 'IS_RESPONSIBLE', '_responsibles', importDir, 'tsv')
    store_new_relationships(external_identifier, data['participant'][0].split(separator), [external_identifier], 'PARTICIPATES_IN', '_participants', importDir, 'tsv')
    store_new_relationships(external_identifier, [external_identifier], disease_ids, 'STUDIES_DISEASE', '_studies_disease', importDir, 'tsv')
    store_new_relationships(external_identifier, [external_identifier], tissue_ids, 'STUDIES_TISSUE', '_studies_tissue', importDir, 'tsv')
    return result.values[0], external_identifier


def create_new_subject(driver, projectId, subjects):
    query_name = 'create_subjects'
    subject_identifier='No Identifier Assigned'
    subject_ids = []
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for subject in list(np.arange(subjects)):
            done = 0
            subject_identifier = get_new_subject_identifier(driver, projectId)
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'subject_id': str(subject_identifier), 'external_id': str(projectId)})
                else:
                    result = connector.getCursorData(driver, q+';')
            subject_ids.append(subject_identifier)
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))

    data = pd.DataFrame(subject_ids)
    data.insert(loc=0, column='', value=projectId)
    data.columns = ['START_ID', 'END_ID']
    data['TYPE'] = 'HAS_ENROLLED'
    store_as_file(projectId, data, projectId+'_project', importDir, 'tsv')
    return done

def create_new_timepoint(driver, projectId, data, separator='|'):
    query_name = 'create_timepoint'
    df = eh.extractTimepoints(data, separator=separator)
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        done = 0
        for index, row in df.iterrows():
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'timepoint': str(row['ID']), 'units': str(row['units'])})
                else:
                    result = connector.getCursorData(driver, q+';')
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
 
    store_as_file(projectId, df, projectId+'_timepoint', importDir, 'tsv')
    return done

def create_intervention_relationship(driver, projectId, data, separator='|'):
    query_name = 'create_intervention_relationship'
    data = eh.extractProjectInterventionRelationships(data, separator=separator)
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        done = 0
        for i in data['END_ID'].astype(str):
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'external_id': projectId, 'intervention_id': i})
                else:
                    result = connector.getCursorData(driver, q+';')
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))

    store_as_file(projectId, data, projectId+'_studies_intervention', importDir, 'tsv')
    return done

def store_as_file(projectId, data, filename, folder, file_format):
    if data is not None:
        outputDir = os.path.join(folder, os.path.join(projectId,'clinical'))
        ckg_utils.checkDirectory(outputDir)
        outputfile = os.path.join(outputDir, filename+'.{}'.format(file_format))
        if file_format == 'tsv':
            with open(outputfile, 'w') as f:
                data.to_csv(path_or_buf = f, sep='\t',
                            header=True, index=False, quotechar='"',
                            line_terminator='\n', escapechar='\\')
        if file_format == 'xlsx':
            with pd.ExcelWriter(outputfile, mode='w') as e:
                data.to_excel(e, index=False)

def store_new_project(projectId, data, folder, file_format):
    if data is not None:
        filename = 'ProjectData_{}'.format(projectId)
        store_as_file(projectId, data, filename, folder, file_format)

def store_new_relationships(projectId, start_node_list, end_node_list, relationship, filename, folder, file_format):
    length = int(len(max([start_node_list, end_node_list], key=len)))
    data = pd.DataFrame(index=np.arange(length), columns=['START_ID', 'END_ID', 'TYPE'])
    if len(start_node_list) == len(data.index):
        data['START_ID'] = start_node_list
    else: data['START_ID'] = start_node_list[0]
    if len(end_node_list) == len(data.index):
        data['END_ID'] = end_node_list
    else: data['END_ID'] = end_node_list[0]
    data['TYPE'] = relationship
    
    filename = projectId+'_{}'.format(filename)
    store_as_file(projectId, data, filename, folder, file_format)



