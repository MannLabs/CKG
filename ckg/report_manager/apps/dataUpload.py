import os
import sys
import re
import pandas as pd
import numpy as np
from ckg import ckg_utils
from ckg.graphdb_connector import connector
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_connector import query_utils
from ckg.analytics_core.viz import viz

ckg_config = ckg_utils.read_ckg_config()
log_config = ckg_config['graphdb_builder_log']
logger = builder_utils.setup_logging(log_config, key="data_upload")


def get_data_upload_queries():
    """
    Reads the YAML file containing the queries relevant to parsing of clinical data and \
    returns a Python object (dict[dict]).

    :return: Nested dictionary.
    """
    try:
        queries_path = "../queries/data_upload_cypher.yml"
        directory = os.path.dirname(os.path.abspath(__file__))
        data_upload_cypher = ckg_utils.get_queries(os.path.join(directory, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Reading queries from file {}: {}, file: {},line: {}".format(err, queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))

    return data_upload_cypher


def get_new_subject_identifier(driver):
    """
    Queries the database for the last subject identifier and returns a new sequential identifier.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str projectId: external project identifier (from the graph database).
    :return: Subject identifier.
    :rtype: str
    """
    query_name = 'increment_subject_id'
    query = ''
    try:
        cypher = get_data_upload_queries()
        query = cypher[query_name]['query']
        subject_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        subject_identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Getting new subject identifiers: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))
    return subject_identifier


def get_new_biosample_identifier(driver):
    """
    Queries the database for the last biological sample internal identifier and returns a new sequential identifier.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.

    :return: Biological sample identifier.
    """
    query_name = 'increment_biosample_id'
    query = ''
    try:
        cypher = get_data_upload_queries()
        query = cypher[query_name]['query']
        identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Getting new biological sample identifiers: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    return identifier


def get_new_analytical_sample_identifier(driver):
    """
    Queries the database for the last analytical sample internal identifier and returns a new sequential identifier.
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.

    :return: Analytical sample identifier.
    """
    query_name = 'increment_analytical_sample_id'
    query = ''
    try:
        cypher = get_data_upload_queries()
        query = cypher[query_name]['query']
        identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        identifier = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Getting new analytical sample identifiers: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    return identifier


def get_subjects_enrolled_in_project(driver, projectId):
    """
    Extracts the number of subjects included in a given project.

    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param str projectId: external project identifier (from the graph database).
    :return: Number of subjects.
    :rtype: Numpy ndarray
    """
    query_name = 'extract_enrolled_subjects'
    query = ''
    try:
        data_upload_cypher = get_data_upload_queries()
        query = data_upload_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'external_id': str(projectId)})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Getting new subjects enrolled in project: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))
    return result.values


def check_samples_in_project(driver, projectId):
    """
    """
    query_name = 'extract_samples_numbers'
    query = ''
    result = pd.DataFrame()
    try:
        data_upload_cypher = get_data_upload_queries()
        query = data_upload_cypher[query_name]['query']
        result = connector.getCursorData(driver, query, parameters={'external_id': str(projectId)})
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Checking whether samples exist in project: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    return result


def check_external_ids_in_db(driver, projectId):
    """
    """
    query_name = 'check_external_ids'
    query = ''
    result = pd.DataFrame()
    try:
        data_upload_cypher = get_data_upload_queries()
        query = data_upload_cypher[query_name]['query']
        result = connector.getCursorData(driver, query, parameters={'external_id': str(projectId)})
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Checking if external identifiers exist in the database: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    return result


def remove_samples_nodes_db(driver, projectId):
    """
    """
    result = None
    query_name = 'remove_project'
    query = ''
    try:
        queries_path = "../queries/project_cypher.yml"
        directory = os.path.dirname(os.path.abspath(__file__))
        project_cypher = ckg_utils.get_queries(os.path.join(directory, queries_path))
        query = project_cypher[query_name]['query'].replace('PROJECTID', projectId).split(';')[:-2]
        for q in query:
            result = connector.commitQuery(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Removing nodes associated to project: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    return result


def create_new_subjects(driver, data, projectId):
    """
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :param data: pandas Dataframe with clinical data as columns and samples as rows.
    :param string projectId: project identifier.
    :return: Pandas DataFrame where new biological sample internal identifiers have been added.
    """
    external_ids = data['subject external_id'].unique()
    subject_id = get_new_subject_identifier(driver)
    if subject_id is None:
        subject_id = '1'
    subject_ids = ['S'+str(i) for i in np.arange(int(subject_id), int(subject_id) + len(external_ids))]
    subject_dict = dict(zip(external_ids, subject_ids))
    query_name = 'create_project_subject'
    for external_id, subject_id in subject_dict.items():
        parameters = {'external_id': str(external_id), 'project_id': projectId, 'subject_id': subject_id}
        try:
            query = ''
            data_upload_cypher = get_data_upload_queries()
            queries = data_upload_cypher[query_name]['query'].split(';')[:-1]
            for query in queries:
                res = connector.commitQuery(driver, query+';', parameters=parameters)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error: {}. Creating new subjects: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    data['subject id'] = data['subject external_id'].map(subject_dict)

    return data


def create_new_biosamples(driver, data):
    """
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :param data: pandas Dataframe with clinical data as columns and samples as rows.

    :return: Pandas DataFrame where new biological sample internal identifiers have been added.
    """
    external_ids = data['biological_sample external_id'].unique()
    subject_ids = data['subject id']
    biosample_id = get_new_biosample_identifier(driver)
    if biosample_id is None:
        biosample_id = '1'

    biosample_ids = ['BS'+str(i) for i in np.arange(int(biosample_id), int(biosample_id) + len(external_ids))]
    biosample_dict = dict(zip(external_ids, biosample_ids))
    biosample_subject_dict = dict(zip(external_ids, subject_ids))
    query_name = 'create_subject_biosamples'
    for external_id, biosample_id in biosample_dict.items():
        subject_id = biosample_subject_dict[external_id]
        parameters = {'external_id': str(external_id), 'biosample_id':biosample_id, 'subject_id': subject_id}
        try:
            query = ''
            data_upload_cypher = get_data_upload_queries()
            queries = data_upload_cypher[query_name]['query'].split(';')[:-1]
            for query in queries:
                res = connector.commitQuery(driver, query+';', parameters=parameters)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error: {}. Creating biological samples: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    data['biological_sample id'] = data['biological_sample external_id'].map(biosample_dict)

    return data


def create_new_ansamples(driver, data):
    """
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :param data: pandas Dataframe with clinical data as columns and samples as rows.

    :return: Pandas DataFrame where new analytical sample internal identifiers have been added.
    """
    data = data.rename(columns={'analytical_sample external_id': 'external_id', 'biological_sample id': 'biosample_id'})
    data['external_id'] = data['external_id'].astype(str)
    num_samples = data['external_id'].shape[0]
    if 'grouping2' not in data:
        data['grouping2'] = None
    if 'batch' not in data:
        data['batch'] = None
    ansample_id = get_new_analytical_sample_identifier(driver)
    if ansample_id is None:
        ansample_id = '1'

    ansample_ids = ['AS' + str(i) for i in np.arange(int(ansample_id), int(ansample_id) + num_samples)]
    data['asample_id'] = ansample_ids
    query_name = 'create_asamples_biosamples'
    for parameters in data.to_dict('records'):
        print(parameters)
        try:
            query = ''
            data_upload_cypher = get_data_upload_queries()
            queries = data_upload_cypher[query_name]['query'].split(';')[:-1]
            for query in queries:
                res = connector.commitQuery(driver, query+';', parameters=parameters)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error: {}. Creating analytical samples: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    data = data.rename(columns={'asample_id': 'analytical_sample id', 'external_id': 'analytical_sample external_id', 'biosample_id': 'biological_sample id'})

    return data


def create_experiment_internal_identifiers(driver, projectId, data, directory, filename):
    done = 0
    df = create_new_subjects(driver, data, projectId)
    df1 = create_new_biosamples(driver, df)
    df2 = create_new_ansamples(driver, df1)
    builder_utils.export_contents(df2, directory, filename)
    done += 1

    return done


def create_mapping_cols_clinical(driver, data, directory, filename, separator='|'):
    """
    :param driver: neo4j driver, which provides the connection to the neo4j graph database.
    :type driver: neo4j driver
    :param data: pandas Dataframe with clinical data as columns and samples as rows.
    :param str separator: character used to separate multiple entries in an attribute.

    :return: Pandas Dataframe with all clinical data and graph database internal identifiers.
    """
    tissue_dict = {}
    disease_dict = {}
    intervention_dict = {}
    if 'disease' in data:
        for disease in data['disease'].dropna().unique():
            if len(disease.split(separator)) > 1:
                ids = []
                for i in disease.split(separator):
                    disease_id = query_utils.map_node_name_to_id(driver, 'Disease', str(i.strip()))
                    if disease_id is not None:
                        ids.append(disease_id)
                    disease_dict[disease] = '|'.join(ids)
            else:
                disease_id = query_utils.map_node_name_to_id(driver, 'Disease', str(disease.strip()))
                disease_dict[disease] = disease_id
        data['disease id'] = data['disease'].map(disease_dict)

    if 'tissue' in data:
        for tissue in data['tissue'].dropna().unique():
            tissue_id = query_utils.map_node_name_to_id(driver, 'Tissue', str(tissue.strip()))
            tissue_dict[tissue] = tissue_id

        data['tissue id'] = data['tissue'].map(tissue_dict)

    if 'studies_intervention' in data:
        for interventions in data['studies_intervention'].dropna().unique():
            for intervention in str(interventions).split('|'):
                if len(intervention.split()) > 1:
                    intervention_dict[intervention] = re.search(r'\(([^)]+)', intervention.split()[-1]).group(1)
                else:
                    intervention_dict[intervention] = intervention

        data['intervention id'] = data['studies_intervention'].map(intervention_dict)

    builder_utils.export_contents(data, directory, filename)


def get_project_information(driver, project_id):
    query_name = 'project_graph'
    queries = []
    data = []
    res = []
    try:
        query = ''
        data_upload_cypher = get_data_upload_queries()
        for section in data_upload_cypher[query_name]:
            code = section['query']
            queries.extend(code.replace("PROJECTID", project_id).split(';')[0:-1])
        for query in queries:
            result = connector.sendQuery(driver, query+";")[0]
            data.append(result)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Creating analytical samples: Query name ({}) - Query ({}), error info: {}, file: {},line: {}".format(err, query_name, query, sys.exc_info(), fname, exc_tb.tb_lineno))

    if data:
        for i, j in enumerate(data):
            df = pd.DataFrame([data[i]], columns=data[i].keys())
            header = '_'.join(df.columns[0].split('_', 1)[1:]).capitalize()
            df.rename(columns={df.columns[0]: 'project'}, inplace=True)
            res.append(viz.get_table(df, identifier='new_project_{}'.format(header), args={'title':'{} data uploaded for project {}'.format(header, project_id)}))
    else:
        res = None
        logger.error("Error: No data was uploaded for project: {}. Review your experimental design and data files and the logs for errors.".format(project_id))

    return res
