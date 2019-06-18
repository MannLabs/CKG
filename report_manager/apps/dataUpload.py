import os
import sys
import re
import pandas as pd
import numpy as np
from natsort import natsorted
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder import builder_utils
from graphdb_builder.experiments import experiments_controller as eh
from report_manager.queries import query_utils
from apps import projectCreation as pc
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="data_upload")

try:
	cwd = os.path.abspath(os.path.dirname(__file__))
	config = builder_utils.setup_config('experiments')
except Exception as err:
	logger.error("Reading configuration > {}.".format(err))


def get_data_upload_queries():
	try:
		queries_path = "../queries/data_upload_cypher.yml"
		data_upload_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
	return data_upload_cypher

def get_new_biosample_identifier(driver):
	query_name = 'increment_biosample_id'
	try:
		cypher = get_data_upload_queries()
		query = cypher[query_name]['query']
		identifier = connector.getCursorData(driver, query).values[0][0]
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
	return identifier

def get_new_analytical_sample_identifier(driver):
	query_name = 'increment_analytical_sample_id'
	try:
		cypher = get_data_upload_queries()
		query = cypher[query_name]['query']
		identifier = connector.getCursorData(driver, query).values[0][0]
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
	return identifiers

def create_new_biosamples(driver, projectId, data):
	query_name = 'create_biosample'
	biosample_dict = {}
	done = 0
	try:
		cypher = get_data_upload_queries()
		query = cypher[query_name]['query']
		for subject, sub_external_id, bio_external_id in zip(data['subject id'], data['subject external_id'], data['biological_sample external_id'].unique()):
			biosample_id = get_new_biosample_identifier(driver)
			biosample_dict[bio_external_id] = biosample_id
			for q in query.split(';')[0:-1]:
				if '$' in q:
					result = connector.getCursorData(driver, q+';', parameters={'biosample_id': str(biosample_id), 'bio_external_id': str(bio_external_id), 'subject_id': str(subject), 'sub_external_id': str(sub_external_id)})
				else:
					result = connector.getCursorData(driver, q+';')
			done += 1
		data.insert(1, 'biological_sample id', data['biological_sample external_id'].map(biosample_dict))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
	return data

def create_new_ansamples(driver, projectId, data):
	query_name = 'create_analytical_sample'
	ansample_dict = {}
	done = 0
	try:
		cypher = get_data_upload_queries()
		query = cypher[query_name]['query']
		for biosample_id, an_external_id, group in zip(data['biological_sample id'], data['analytical_sample external_id'].unique(), data['grouping1']):
			ansample_id = get_new_analytical_sample_identifier(driver)
			ansample_dict[an_external_id] = ansample_id
			for q in query.split(';')[0:-1]:
				if '$' in q:
					result = connector.getCursorData(driver, q+';', parameters={'project_id': str(projectId), 'ansample_id': str(ansample_id), 'an_external_id': str(an_external_id), 'group': str(group), 'biosample_id': str(biosample_id)})
				else:
					result = connector.getCursorData(driver, q+';')
			done += 1
		data.insert(2, 'analytical_sample id', data['analytical_sample external_id'].map(ansample_dict))
	except Exception as err:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
	return data	

def create_csv(driver, projectId, data):
	disease_dict = {}
	tissue_dict = {}
	intervention_dict = {}

	for disease in data['disease'].dropna().unique():
		if len(disease.split('|')) > 1:
			ids = []
			for i in disease.split('|'):
				disease_id = query_utils.map_node_name_to_id(driver, 'Disease', str(i))
				ids.append(disease_id)
			disease_dict[disease] = '|'.join(ids)
		else:
			disease_id = query_utils.map_node_name_to_id(driver, 'Disease', str(disease))
			disease_dict[disease] = disease_id

	for tissue in data['tissue'].dropna().unique():
		if tissue is not np.nan:
			tissue_id = query_utils.map_node_name_to_id(driver, 'Tissue', str(tissue))
			tissue_dict[tissue] = tissue_id

	for intervention in data['intervention'].dropna().unique():
		if len(intervention.split('|')) > 1:
			ids = []
			for i in intervention.split('|'):
				intervention_id = query_utils.map_node_name_to_id(driver, 'Clinical_variable', str(i))
				ids.append(intervention_id)
			intervention_dict[intervention] = '|'.join(ids)
		else:
			intervention_id = query_utils.map_node_name_to_id(driver, 'Clinical_variable', str(intervention))
			intervention_dict[intervention] = intervention_id

	data.insert(1, 'tissue id', data['tissue'].map(tissue_dict))
	data.insert(2, 'disease id', data['disease'].map(disease_dict))
	data.insert(3, 'intervention id', data['intervention'].map(intervention_dict))

	df = create_new_biosamples(driver, projectId, data)
	df2 = create_new_ansamples(driver, projectId, df)

	return df2

	# print(df2)

	# cols = [col for col in df2 if col.startswith('biological_sample')]
	# store_as_file(projectId, df2[cols], '_biological_samples', 'csv', d='clinical', columns=[col.replace('biological_sample ', '') for col in cols], relationship=False)  
	# cols = [col for col in df2 if col.startswith('analytical_sample')]+['grouping1', 'grouping2']
	# store_as_file(projectId, df2[cols], '_analytical_samples', 'csv', d='clinical', columns=[col.replace('analytical_sample ','').replace('grouping1','group').replace('grouping2','secondary_group') for col in cols], relationship=False)  
	# store_as_file(projectId, df2[['subject id','subject external_id']], '_subjects', 'csv', d='clinical', columns=['ID', 'external_id'], relationship=False)
	# store_as_file(projectId, df2[['biological_sample id','subject id']], '_subject_biosample', 'csv', d='clinical', columns=None, relationship=True, relationship_type='BELONGS_TO_SUBJECT')
	# store_as_file(projectId, df2[['subject id','disease id']], '_disease', 'csv', d='clinical', columns=None, relationship=True, relationship_type='HAS_DISEASE')
	# store_as_file(projectId, df2[['biological_sample id','tissue id']], '_biosample_tissue',   'csv', d='clinical', columns=None, relationship=True, relationship_type='FROM_TISSUE')
	# store_as_file(projectId, df2[['biological_sample id', 'analytical_sample id', 'analytical_sample quantity', 'analytical_sample quantity_units']], '_biosample_analytical', 'csv', d='clinical', relationship=True, relationship_type='SPLITTED_INTO', columns=['START_ID', 'END_ID', 'quantity', 'quantity_units'])
	# store_as_file(projectId, df2[['biological_sample id', 'timepoint', 'timepoint units', 'intervention id']], '_biological_sample_sample_at_timepoint', 'csv', d='clinical', relationship=True, relationship_type='SAMPLE_AT_TIMEPOINT', columns=['START_ID', 'END_ID', 'timepoint_units', 'intervention'])

	# groups = df2.copy()
	# if df2['grouping2'].isnull().all():
	# 	groups = df2[['biological_sample id', 'grouping1']]
	# 	groups.columns = ['START_ID', 'END_ID']
	# 	groups['TYPE'] = 'BELONGS_TO_GROUP'
	# 	groups['primary'] = True
	# else:
	# 	groups = df2[['biological_sample id', 'grouping1', 'grouping2']]
	# 	groups = pd.melt(df, id_vars=['biological_sample id'], value_vars=['grouping1', 'grouping2'])
	# 	groups['primary'] = groups['variable'].map(lambda x: x=='grouping1')
	# 	groups = groups.drop(['variable'], axis=1).dropna(subset=['value']).sort_values('biological_sample ID').reset_index(drop=True)
	# 	groups.columns = ['START_ID', 'END_ID', 'primary']
	# 	groups.insert(loc=2, column='TYPE', value='BELONGS_TO_GROUP')
	# 	groups = groups.drop_duplicates(keep='first').sort_values(by=['START_ID'], ascending=True).reset_index(drop=True)
	# store_as_file(projectId, groups, '_groups', 'csv', d='clinical', relationship=False)
	

	# clinical_quant = df2.set_index('biological_sample id').copy()
	# clinical_quant = clinical_quant._get_numeric_data()
	# cols = [i for i in clinical_quant.columns if str(i).endswith(' id')]
	# clinical_quant = clinical_quant.drop(cols, axis=1)
	# clinical_quant = clinical_quant.stack().reset_index()
	# clinical_quant.columns = ['START_ID', 'END_ID', 'value']
	# clinical_quant.insert(loc=2, column='TYPE', value='HAS_QUANTIFIED_CLINICAL')
	# clinical_quant['END_ID'] = clinical_quant['END_ID'].apply(lambda x: int(x) if isinstance(x,float) else x)
	# clinical_quant = clinical_quant.drop_duplicates(keep='first').dropna()
	# store_as_file(projectId, clinical_quant, '_clinical_quant', 'csv', d='clinical', relationship=False)

	# clinical_state = df2.copy()
	# intervention = None
	# if 'intervention id' in clinical_state.columns and clinical_state['intervention id'].dropna().empty != True:
	# 	intervention = clinical_state['intervention id'].to_frame().dropna(subset=['intervention id'])
	# 	intervention = intervention.reset_index()
	# 	intervention.columns = ['START_ID', 'END_ID']
	# 	intervention['END_ID'] = intervention['END_ID'].astype('int64')
	# 	intervention['value'] = True
	# clinical_state = clinical_state.set_index('subject id')
	# clinical_state = df.drop([i for i in clinical_state.columns if str(i).endswith(' id')], axis=1)
	# clinical_state = clinical_state.stack()
	# clinical_state = clinical_state.reset_index()
	# clinical_state.columns = ['START_ID', 'END_ID', "value"]
	# if intervention is not None:
	# 	clinical_state = clinical_state.append(intervention, sort=True)
	# clinical_state['TYPE'] = "HAS_CLINICAL_STATE"
	# clinical_state = clinical_state[['START_ID', 'END_ID','TYPE', 'value']]
	# clinical_state['END_ID'] = clinical_state['END_ID'].apply(lambda x: int(x) if isinstance(x,float) else x)
	# clinical_state = clinical_state[clinical_state['value'].apply(lambda x: isinstance(x, str))]
	# clinical_state = clinical_state.drop_duplicates(keep='first').dropna()
	# store_as_file(projectId, clinical_state, '_clinical_state', 'csv', d='clinical', relationship=False)



# def store_as_file(projectId, data, filename, file_format, columns=None, d='clinical', relationship=False, relationship_type=None):
# 	if data is not None:
# 		if relationship:
# 			if len(data.columns) > 2:
# 				df = data.copy()
# 				df.columns = columns
# 				df.insert(2, 'TYPE', relationship_type)
# 			else:
# 				df = data.copy()
# 				df.columns = ['START_ID', 'END_ID']
# 				df['TYPE'] = relationship_type
# 		else:
# 			df = data.copy()
# 			if columns is None:
# 				pass
# 			else:
# 				df.columns = columns

# 		importDir = os.path.join(os.path.join(cwd, '../../../data/imports/experiments'), os.path.join(projectId,d))
# 		ckg_utils.checkDirectory(importDir)
# 		outputfile = os.path.join(importDir, projectId+filename+'.{}'.format(file_format))
# 		# df.drop_duplicates(subset=['A', 'C'], keep='first', inplace=True)
# 		if file_format == 'csv':
# 			with open(outputfile, 'w') as f:
# 				df.to_csv(path_or_buf = f,
# 						header=True, index=False, quotechar='"',
# 						line_terminator='\n', escapechar='\\')
# 		if file_format == 'xlsx':
# 			with pd.ExcelWriter(outputfile, mode='w') as e:
# 				df.to_excel(e, index=False)
	

































