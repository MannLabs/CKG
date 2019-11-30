import os
import re
import pandas as pd
import numpy as np
from graphdb_builder import builder_utils
from graphdb_connector import connector

def parser(projectId):
    data = {}
    config = builder_utils.get_config(config_name="clinical.yml", data_type='experiments')
    directory = '../../../data/experiments/PROJECTID/clinical/'
    separator = config["separator"]
    if 'directory' in config:
        directory = config['directory']
    directory = directory.replace('PROJECTID', projectId)
    driver = connector.getGraphDatabaseConnectionConfiguration()
    
    project_data = parse_dataset(projectId, config, directory, key='project')
    clinical_data = parse_dataset(projectId, config, directory, key='clinical')
    if project_data is not None and clinical_data is not None:
        data[('info', 'w')] = extract_project_info(project_data)
        data[('responsibles', 'w')] = extract_responsible_rels(project_data, separator=separator)
        data[('participants', 'w')] = extract_participant_rels(project_data, separator=separator)
        data[('studies_tissue', 'w')] = extract_project_tissue_rels(driver, project_data, separator=separator)
        data[('studies_disease', 'w')] = extract_project_disease_rels(driver, project_data, separator=separator)
        data[('studies_intervention', 'w')] = extract_project_intervention_rels(project_data, separator=separator)
        data[('timepoint', 'w')] = extract_timepoints(project_data, separator=separator)
        data[('project', 'w')] = extract_project_subject_rels(project_data, clinical_data)
        data[('subjects', 'w')] = extract_subject_identifiers(project_data, clinical_data)
        data[('subject_biosample', 'w')] = extract_biological_sample_subject_rels(clinical_data)
        data[('biological_samples', 'w')] = extract_biological_samples_info(clinical_data)
        data[('analytical_samples', 'w')] = extract_analytical_samples_info(clinical_data)
        data[('biosample_analytical', 'w')] = extract_biological_sample_analytical_sample_rels(clinical_data)
        data[('biological_sample_at_timepoint', 'w')] = extract_biological_sample_timepoint_rels(clinical_data)
        data[('biosample_tissue', 'w')] = extract_biological_sample_tissue_rels(clinical_data)
        data[('disease', 'w')] = extract_subject_disease_rels(clinical_data, separator=separator)
        data[('groups', 'w')] = extract_biological_sample_group_rels(clinical_data)
        clinical_state, clinical_quant = extract_biological_sample_clinical_variables_rels(clinical_data)
        data[('clinical_state', 'w')] = clinical_state
        data[('clinical_quant', 'w')] = clinical_quant
        
    return data


def parse_dataset(projectId, configuration, dataDir, key='project'):
    '''This function parses clinical data from subjects in the project
    Input: uri of the clinical data file. Format: Subjects as rows, clinical variables as columns
    Output: pandas DataFrame with the same input format but the clinical variables mapped to the
    right ontology (defined in config), i.e. type = -40 -> SNOMED CT'''
    data = None
    if 'file_'+key in configuration:
        data_file = configuration['file_'+key].replace('PROJECTID', projectId)
    
        filepath = os.path.join(dataDir, data_file)
        if os.path.isfile(filepath):
            data = builder_utils.readDataset(filepath)

    return data

def extract_project_info(project_data):
    df = project_data.copy()
    df.columns = ['internal_id', 'name', 'acronym', 'description', 'subjects', 'datatypes', 'timepoints', 'disease', 'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date', 'status', 'external_id']
    return df

def extract_responsible_rels(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['responsible'][0]):
        return None
    else:
        df = pd.DataFrame(data.responsible.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
        df['END_ID'] = data['external_id'][0]
        df['TYPE'] = 'IS_RESPONSIBLE'
        return df

def extract_participant_rels(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['participant'][0]):
        return None
    else:
        df = pd.DataFrame(data.participant.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
        df['END_ID'] = data['external_id'][0]
        df['TYPE'] = 'PARTICIPATES_IN'
        return df
    
def extract_project_tissue_rels(driver, project_data, separator='|'):
    data = project_data.copy()
    
    if pd.isna(data['tissue'][0]):
        return None
    else:
        tissues = data['tissue'][0].split(separator)
        df = pd.DataFrame(tissues, columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=data['external_id'][0])
        df['TYPE'] = 'STUDIES_TISSUE'
        return df
    
def extract_project_disease_rels(driver, project_data, separator='|'):
    data = project_data.copy()
    
    if pd.isna(data['disease'][0]):
        return None
    else:
        diseases = data['disease'][0].split(separator)
        df = pd.DataFrame(diseases, columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=data['external_id'][0])
        df['TYPE'] = 'STUDIES_DISEASE'
        return df

def extract_project_intervention_rels(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['intervention'][0]):
        return pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    else:
        interventions = data['intervention'][0].split(separator)
        ids = [re.search(r'\(([^)]+)',x.split()[-1]).group(1) for x in interventions]
        df = pd.DataFrame(ids, columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=data['external_id'][0])
        df['TYPE'] = 'STUDIES_INTERVENTION'
        return df

def extract_timepoints(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['timepoints'][0]):
        return pd.DataFrame(columns=['ID', 'units', 'type'])
    else:
        df = pd.DataFrame(data['timepoints'][0].replace(' ','').split(separator))
        df = df[0].str.extract(r'([\-\d]+)([a-zA-Z]+)', expand=True)
        df.columns = ['ID', 'units']
        df['type'] = 'Timepoint'
        return df

def extract_project_subject_rels(project_data, clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['subject id']).any():
        return None
    else:
        df = pd.DataFrame(data['subject id'].dropna().unique(), columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
        df['TYPE'] = 'HAS_ENROLLED'
        return df

def extract_subject_identifiers(project_data, clinical_data):
    data = clinical_data.set_index('subject id').copy()
    if pd.isna(data['subject external_id']).any():
        return None
    else:
        df = data[['subject external_id']].dropna(axis=0).reset_index()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['ID', 'external_id']
        if int(project_data['subjects'][0]) != len(df['ID']):
            df = None
        return df

def extract_biological_sample_subject_rels(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['biological_sample id']).any():
        return None
    else:
        df = data[['biological_sample id', 'subject id']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID']
        df['TYPE'] = 'BELONGS_TO_SUBJECT'
        return df

def extract_biological_samples_info(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['biological_sample id']).any():
        return None
    else:
        cols = [i for i in data.columns if str(i).startswith('biological_sample')]
        df = data[cols]
        df.columns=[col.replace('biological_sample ', '') for col in cols]
        df = df.rename(columns={'id':'ID'})
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        return df

def extract_analytical_samples_info(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['analytical_sample id']).any():
        return None
    else:
        cols = [i for i in data.columns if str(i).startswith('analytical_sample')]
        df = data[cols]
        df.columns = [col.replace('analytical_sample ', '') for col in cols]
        df = df.rename(columns={'id':'ID'})
        df[['group', 'secondary_group']] = clinical_data[['grouping1', 'grouping2']]
        return df

def extract_biological_sample_analytical_sample_rels(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['analytical_sample id']).any():
        return None
    else:
        df = data[['biological_sample id', 'analytical_sample id', 'analytical_sample quantity', 'analytical_sample quantity_units']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID', 'quantity', 'quantity_units']
        df.insert(loc=2, column='TYPE', value='SPLITTED_INTO')
        return df

def extract_biological_sample_timepoint_rels(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['timepoint']).all():
        return pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE','timepoint_units', 'intervention'])
    else:
        df = data[['biological_sample id', 'timepoint', 'timepoint units', 'intervention id']].drop_duplicates(keep='first').reset_index(drop=True)
        df['intervention id'] = df['intervention id'].replace(np.nan, 0).astype('int64').astype('str').replace('0', np.nan)
        df.columns = ['START_ID', 'END_ID', 'timepoint_units', 'intervention']
        df.insert(loc=2, column='TYPE', value='SAMPLE_AT_TIMEPOINT')
        return df

def extract_biological_sample_tissue_rels(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['tissue id']).all():
        return None
    else:
        df = data[['biological_sample id', 'tissue id']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID']
        df['TYPE'] = 'FROM_TISSUE'
        return df

def extract_subject_disease_rels(clinical_data, separator='|'):
    data = clinical_data.copy()
    if pd.isna(data['disease id']).all():
        return None
    else:
        data = data.astype(str)
        df = pd.DataFrame(data['disease id'].str.split(separator).tolist(), index=data['subject id']).stack()
        df = df.reset_index([0, 'subject id']).replace('nan', np.nan).dropna().drop_duplicates(keep='first')
        df.columns = ['START_ID', 'END_ID']
        df['TYPE'] = 'HAS_DISEASE'
        return df

def extract_biological_sample_group_rels(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['grouping1']).any():
        return None
    else:
        df = data[['biological_sample id', 'grouping1', 'grouping2']]
        df = pd.melt(df, id_vars=['biological_sample id'], value_vars=['grouping1', 'grouping2'])
        df['primary'] = df['variable'].map(lambda x: x=='grouping1')
        df = df.drop(['variable'], axis=1).dropna(subset=['value']).drop_duplicates(keep='first').sort_values('biological_sample id').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID', 'primary']
        df.insert(loc=2, column='TYPE', value='BELONGS_TO_GROUP')
        return df

def extract_biological_sample_clinical_variables_rels(clinical_data):
    data = clinical_data.set_index('biological_sample id').copy()
    df = data.loc[:,'grouping2':].drop('grouping2', axis=1)
    df.columns = [i.split()[-1] for i in df.columns]
    df.columns = df.columns.str.extract(r'.*\((.*)\).*')[0].tolist()
    df_quant = df._get_numeric_data()
    df_state = df.loc[:,~df.columns.isin(df_quant.columns.tolist())]
    if df_quant.empty:
        df_quant = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE','value'])
    else:
        df_quant = df_quant.stack().reset_index().drop_duplicates(keep='first').dropna()
        df_quant.columns = ['START_ID', 'END_ID', 'value']
        df_quant.insert(loc=2, column='TYPE', value='HAS_QUANTIFIED_CLINICAL')
    if df_state.empty:
        df_state = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE','value'])
    else:
        df_state = df_state.stack().reset_index().drop_duplicates(keep='first').dropna()
        df_state.columns = ['START_ID', 'END_ID', 'value']
        df_state.insert(loc=2, column='TYPE', value='HAS_CLINICAL_STATE')
    
    return df_state, df_quant