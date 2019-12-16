import os
import sys
import re
import pandas as pd
import numpy as np
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder import builder_utils
from graphdb_builder.builder import loader
from graphdb_builder.experiments.parsers import clinicalParser as cp
from graphdb_connector import query_utils
import logging
import logging.config
 
log_config = ckg_config.report_manager_log
logger = builder_utils.setup_logging(log_config, key="project_creation")
 
cwd = os.path.abspath(os.path.dirname(__file__))
experimentDir = os.path.join(cwd, '../../../data/experiments')
importDir = os.path.join(cwd, '../../../data/imports/experiments')
 
def get_project_creation_queries():
    """
    Reads the YAML file containing the queries relevant to user creation, parses the given stream and \
    returns a Python object (dict[dict]).
 
    :return: Nested dictionary.
    """
    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "../queries/project_creation_cypher.yml"
        project_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
    return project_creation_cypher
 
def check_if_node_exists(driver, node_property, value):
    """
    Queries the graph database and checks if a node with a specific property and property value already exists.
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str node_property: property of the node.
    :param value: property value.
    :type value: str, int, float or bool
    :return: Pandas dataframe with user identifier if User with node_property and value already exists, \
            if User does not exist, returns and empty dataframe.
    """
    query_name = 'check_node'
    try:
        cypher = get_project_creation_queries()
        query = cypher[query_name]['query'].replace('PROPERTY', node_property)
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'value':value})
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
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :return: Project external identifier.
    :rtype: str
    """
    query_name = 'increment_project_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        external_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        external_identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return external_identifier
 
def get_new_subject_identifier(driver, projectId):
    """
    Queries the database for the last subject identifier and returns a new sequential identifier.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :return: Subject identifier.
    :rtype: str
    """
    query_name = 'increment_subject_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        subject_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        subject_identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return subject_identifier
 
 
def get_subjects_in_project(driver, projectId):
    """
    Extracts the number of subjects included in a given project.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :return: Number of subjects.
    :rtype: Numpy ndarray
    """
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
    """
    Creates a new project in the graph database, following the steps:
     
    1. Retrieves new project external identifier and creates project node and relationships in the graph database.
    2. Creates subjects, timepoints and intervention nodes.
    3. Saves all the entities and relationships to tab-delimited files.
    4. Returns the number of projects created and the project external identifier.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :param data: pandas Dataframe with project as row and other attributes as columns.
    :param str separator: character used to separate multiple entries in a project attribute.
    :return: Two strings: number of projects created and the project external identifier.
    """
    query_name = 'create_project'
    external_identifier='No Identifier Assigned'
    done = None
     
    try:
        db_project = check_if_node_exists(driver, 'name', data['name'][0])
        if db_project.empty:
            external_identifier = get_new_project_identifier(driver, projectId)
            if external_identifier is None:
                external_identifier = 'P0000001'
            data['external_id'] = external_identifier

            dataRows = cp.extract_project_info(data)
            if dataRows is not None:
                generateGraphFiles(dataRows,'info', external_identifier, d='clinical')
            dataRows = cp.extract_responsible_rels(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'responsibles', external_identifier, d='clinical')
            dataRows = cp.extract_participant_rels(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'participants', external_identifier, d='clinical')
            dataRows = cp.extract_project_tissue_rels(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'studies_tissue', external_identifier, d='clinical')
            dataRows = cp.extract_project_disease_rels(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'studies_disease', external_identifier, d='clinical')
            dataRows = cp.extract_project_intervention_rels(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'studies_intervention', external_identifier, d='clinical')
            dataRows = cp.extract_timepoints(data, separator=separator)
            if dataRows is not None:
                generateGraphFiles(dataRows,'timepoint', external_identifier, d='clinical')

            loader.partialUpdate(imports=['project'])
            subjects = create_new_subjects(driver, external_identifier, data['subjects'][0])
            done = 1

            projectDir = os.path.join(experimentDir, os.path.join(external_identifier,'clinical'))
            ckg_utils.checkDirectory(projectDir)
            data.to_excel(os.path.join(projectDir, 'ProjectData_{}.xlsx'.format(external_identifier)), index=False, encoding='utf-8')
            
        else:
            done = 0
            external_identifier = ''
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return done, external_identifier   
        
def create_new_subjects(driver, projectId, subjects):
    """
    Creates new graph database nodes for subjects participating in a project.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :param int subjects: number of subjects participating in the project.
    :return: Integer for the number of subjects created.
    """
    query_name = 'create_subjects'
    subject_identifier='No Identifier Assigned'
    # subject_ids = []
    try:
        #Save tsv file with new subject identifiers
        subject_identifier = get_new_subject_identifier(driver, projectId)
        if subject_identifier is None:
            subject_identifier = '1'

        subject_ids = ['S'+str(i) for i in np.arange(int(subject_identifier), int(subject_identifier)+subjects)]
        data = pd.DataFrame(subject_ids)
        data.insert(loc=0, column='', value=projectId)
        data.columns = ['START_ID', 'END_ID']
        data['TYPE'] = 'HAS_ENROLLED'
        generateGraphFiles(data,'project', projectId, d='clinical')
        
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        projectDir = os.path.join(importDir, projectId)
        projectDir = os.path.join(projectDir, 'clinical')
        query = query.replace("IMPORTDIR", projectDir).replace('PROJECTID', projectId)
        for q in query.split(';')[0:-1]:
            result = connector.getCursorData(driver, q+';')

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return result.values[0][0]

def generateGraphFiles(data, dataType, projectId, ot = 'w', d = 'clinical'):
    """
    Saves data provided as a Pandas DataFrame to a tab-delimited file.
    
    :param data: pandas DataFrame.
    :param str dataType: type of data in 'data'.
    :param str projectId: external project identifier (from the graph database).
    :param str ot: mode while opening file.
    :param str d: data type ('proteomics', 'clinical', 'wes').
    """
    importsDir = os.path.join(importDir, os.path.join(projectId,d))
    ckg_utils.checkDirectory(importsDir)
    outputfile = os.path.join(importsDir, projectId+"_"+dataType.lower()+".tsv")
    with open(outputfile, ot) as f:
        data.to_csv(path_or_buf = f, sep='\t',
                    header=True, index=False, quotechar='"',
                    line_terminator='\n', escapechar='\\')
    logger.info("Project {} - Number of {} relationships: {}".format(projectId, dataType, data.shape[0]))
