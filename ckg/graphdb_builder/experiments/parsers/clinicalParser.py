import os
import re
import pandas as pd
import numpy as np
from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils


def parser(projectId, type='clinical'):
    data = {}
    experiments_directory = ckg_utils.read_ckg_config(key='experiments_directory')
    project_directory = os.path.join(experiments_directory, 'PROJECTID/project/')
    clinical_directory = os.path.join(experiments_directory, 'PROJECTID/clinical/')
    design_directory = os.path.join(experiments_directory, 'PROJECTID/experimental_design/')
    project_directory = project_directory.replace('PROJECTID', projectId)
    clinical_directory = clinical_directory.replace('PROJECTID', projectId)
    design_directory = design_directory.replace('PROJECTID', projectId)
    config = builder_utils.get_config(config_name="clinical.yml", data_type='experiments')
    if type == 'project':
        project_dfs = project_parser(projectId, config, project_directory)
        data.update(project_dfs)
    elif type == 'experimental_design':
        design_dfs = experimental_design_parser(projectId, config, design_directory)
        data.update(design_dfs)
    elif type == 'clinical':
        clinical_dfs = clinical_parser(projectId, config, clinical_directory)
        data.update(clinical_dfs)

    return data


def project_parser(projectId, config, directory):
    data = {}
    project_data = parse_dataset(projectId, config, directory, key='project')
    if project_data is not None:
        data[('info', 'w')] = extract_project_info(project_data)
        data[('responsibles', 'w')] = extract_responsible_rels(project_data, separator=config['separator'])
        data[('participants', 'w')] = extract_participant_rels(project_data, separator=config['separator'])
        data[('studies_tissue', 'w')] = extract_project_tissue_rels(project_data, separator=config['separator'])
        data[('studies_disease', 'w')] = extract_project_disease_rels(project_data, separator=config['separator'])
        data[('studies_intervention', 'w')] = extract_project_intervention_rels(project_data, separator=config['separator'])
        data[('follows_up_project', 'w')] = extract_project_rels(project_data, separator=config['separator'])
        data[('timepoint', 'w')] = extract_timepoints(project_data, separator=config['separator'])

    return data


def experimental_design_parser(projectId, config, directory):
    data = {}
    design_data = parse_dataset(projectId, config, directory, key='design')
    if design_data is not None:
        data[('project', 'w')] = extract_project_subject_rels(projectId, design_data)
        data[('subjects', 'w')] = extract_subject_identifiers(design_data)
        data[('biological_samples', 'w')] = extract_biosample_identifiers(design_data)
        data[('analytical_samples', 'w')] = extract_analytical_sample_identifiers(design_data)
        data[('analytical_samples_info', 'w')] = extract_analytical_samples_info(design_data)
        data[('subject_biosample', 'w')] = extract_biological_sample_subject_rels(design_data)
        data[('biosample_analytical', 'w')] = extract_biological_sample_analytical_sample_rels(design_data)

    return data


def clinical_parser(projectId, config, clinical_directory):
    data = {}
    clinical_data = parse_dataset(projectId, config, clinical_directory, key='clinical')
    if clinical_data is not None:
        data[('biosamples_info', 'w')] = extract_biological_samples_info(clinical_data)
        data[('biosample_analytical_attributes', 'w')] = extract_biosample_analytical_sample_relationship_attributes(clinical_data)
        data[('biological_sample_at_timepoint', 'w')] = extract_biological_sample_timepoint_rels(clinical_data)
        data[('biosample_tissue', 'w')] = extract_biological_sample_tissue_rels(clinical_data)
        data[('disease', 'w')] = extract_subject_disease_rels(clinical_data, separator=config['separator'])
        data[('subject_had_intervention', 'w')] = extract_subject_intervention_rels(clinical_data, separator=config['separator'])
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
        files = os.listdir(dataDir)
        regex = r"{}.+".format(data_file)
        r = re.compile(regex)
        filename = list(filter(r.match, files))
        if len(filename) > 0:
            filepath = os.path.join(dataDir, filename.pop())
            if os.path.isfile(filepath):
                data = builder_utils.readDataset(filepath)

    return data


def extract_project_info(project_data):
    cols = ['internal_id', 'name', 'acronym', 'description', 'related_to', 'datatypes', 'timepoints', 'disease', 'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date', 'status', 'external_id']
    n = len(cols)
    df = project_data.copy()
    if len(df.columns) == n:
        df.columns = cols
    else:
        raise Exception("Project data requires {} columns, {} provided.\n Provide the following columns: {}".format(n, len(df.columns), ",".join(cols)))

    return df


def extract_responsible_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'responsible' in project_data:
        if not pd.isna(project_data['responsible'][0]):
            df = pd.DataFrame(project_data.responsible.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
            df['END_ID'] = project_data['external_id'][0]
            df['TYPE'] = 'IS_RESPONSIBLE'

    return df


def extract_participant_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'participant' in project_data:
        if not pd.isna(project_data['participant'][0]):
            df = pd.DataFrame(project_data.participant.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
            df['END_ID'] = project_data['external_id'][0]
            df['TYPE'] = 'PARTICIPATES_IN'

    return df


def extract_project_tissue_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'tissue' in project_data:
        if not pd.isna(project_data['tissue'][0]):
            tissues = project_data['tissue'][0].split(separator)
            df = pd.DataFrame(tissues, columns=['END_ID'])
            df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
            df['TYPE'] = 'STUDIES_TISSUE'

    return df


def extract_project_disease_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'disease' in project_data:
        if not pd.isna(project_data['disease'][0]):
            diseases = project_data['disease'][0].split(separator)
            df = pd.DataFrame(diseases, columns=['END_ID'])
            df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
            df['TYPE'] = 'STUDIES_DISEASE'

    return df


def extract_project_intervention_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'intervention' in project_data:
        if not pd.isna(project_data['intervention'][0]):
            interventions = project_data['intervention'][0].split(separator)
            ids = [re.search(r'\(([^)]+)', x.split()[-1]).group(1) for x in interventions]
            df = pd.DataFrame(ids, columns=['END_ID'])
            df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
            df['TYPE'] = 'STUDIES_INTERVENTION'

    return df


def extract_project_rels(project_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'related_to' in project_data:
        if not pd.isna(project_data['related_to'][0]):
            related_projects = project_data['related_to'][0].split(separator)
            df = pd.DataFrame(related_projects, columns=['END_ID'])
            df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
            df['TYPE'] = 'FOLLOWS_UP_PROJECT'

    return df


def extract_timepoints(project_data, separator='|'):
    df = pd.DataFrame(columns=['ID', 'units', 'type'])
    if 'timepoints' in project_data:
        if not pd.isna(project_data['timepoints'][0]):
            df = pd.DataFrame(project_data['timepoints'][0].replace(' ', '').split(separator))
            df = df[0].str.extract(r'([\-\d]+)([a-zA-Z]+)', expand=True)
            df.columns = ['ID', 'units']
            df['type'] = 'Timepoint'

    return df


def extract_project_subject_rels(projectId, design_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'subject id' in design_data:
        if not pd.isna(design_data['subject id']).any():
            df = pd.DataFrame(design_data['subject id'].dropna().unique(), columns=['END_ID'])
            df.insert(loc=0, column='START_ID', value=projectId)
            df['TYPE'] = 'HAS_ENROLLED'

    return df


def extract_subject_identifiers(design_data):
    df = pd.DataFrame(columns=['ID', 'external_id'])
    data = design_data.set_index('subject id').copy()
    if not pd.isna(data['subject external_id']).any():
        df = data[['subject external_id']].dropna(axis=0).reset_index()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['ID', 'external_id']

    return df


def extract_biosample_identifiers(design_data):
    df = pd.DataFrame(columns=['ID', 'external_id'])
    data = design_data.set_index('biological_sample id').copy()
    if not pd.isna(data['biological_sample external_id']).any():
        df = data[['biological_sample external_id']].dropna(axis=0).reset_index()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['ID', 'external_id']

    return df


def extract_analytical_sample_identifiers(design_data):
    df = pd.DataFrame(columns=['ID', 'external_id'])
    data = design_data.set_index('analytical_sample id').copy()
    if not pd.isna(data['analytical_sample external_id']).any():
        df = data[['analytical_sample external_id']].dropna(axis=0).reset_index()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['ID', 'external_id']

    return df


def extract_biological_sample_subject_rels(design_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'biological_sample id' in design_data:
        if not pd.isna(design_data['biological_sample id']).any():
            df = design_data[['biological_sample id', 'subject id']].drop_duplicates(keep='first').reset_index(drop=True)
            df.columns = ['START_ID', 'END_ID']
            df['TYPE'] = 'BELONGS_TO_SUBJECT'

    return df


def extract_biological_sample_analytical_sample_rels(design_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'analytical_sample id' in design_data:
        if not pd.isna(design_data['analytical_sample external_id']).any():
            df = design_data[['biological_sample id', 'analytical_sample id']].drop_duplicates(keep='first').reset_index(drop=True)
            df.columns = ['START_ID', 'END_ID']
            df['TYPE'] = 'SPLITTED_INTO'

    return df


def extract_biological_samples_info(clinical_data):
    df = pd.DataFrame(columns=['ID'])
    if 'biological_sample external_id' in clinical_data:
        if not pd.isna(clinical_data['biological_sample external_id']).any():
            cols = [i for i in clinical_data.columns if str(i).startswith('biological_sample')]
            df = clinical_data[cols]
            df.columns = [col.replace('biological_sample ', '') for col in cols]
            df = df.rename(columns={'external_id': 'ID'})
            df = df.drop_duplicates(keep='first').reset_index(drop=True)

    return df


def extract_analytical_samples_info(data):
    df = pd.DataFrame(columns=['ID', 'group', 'secondary_group', 'batch'])
    if 'analytical_sample external_id' in data:
        if not pd.isna(data['analytical_sample external_id']).any():
            df = data.copy()
            df.columns = [col.replace('analytical_sample ', '') for col in df.columns]
            df = df.rename(columns={'external_id': 'ID', 'grouping1':'group', 'grouping2':'secondary_group'})
            if 'batch' not in df:
                df['batch'] = None

    return df


def extract_biosample_analytical_sample_relationship_attributes(clinical_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'quantity', 'quantity_units'])
    if 'analytical_sample external_id' in clinical_data:
        if not pd.isna(clinical_data['analytical_sample external_id']).any():
            cols = ['biological_sample external_id', 'analytical_sample external_id']
            edge_cols = ['START_ID', 'END_ID']
            if 'analytical_sample quantity' in clinical_data:
                cols.append('analytical_sample quantity')
                edge_cols.append('quantity')
            if 'analytical_sample quantity_units' in clinical_data:
                cols.append('analytical_sample quantity_units')
                edge_cols.append('quantity_units')
            df = clinical_data[cols].drop_duplicates(keep='first').reset_index(drop=True)
            df.columns = edge_cols

    return df


def extract_biological_sample_timepoint_rels(clinical_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE', 'timepoint_units', 'intervention'])
    if 'timepoint' in clinical_data:
        if not pd.isna(clinical_data['timepoint']).all():
            df = clinical_data[['biological_sample external_id', 'timepoint', 'timepoint units', 'intervention id']].drop_duplicates(keep='first').reset_index(drop=True)
            df['intervention id'] = df['intervention id'].replace(np.nan, 0).astype('int64').astype('str').replace('0', np.nan)
            df.columns = ['START_ID', 'END_ID', 'timepoint_units', 'intervention']
            df.insert(loc=2, column='TYPE', value='SAMPLE_AT_TIMEPOINT')

    return df


def extract_biological_sample_tissue_rels(clinical_data):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'tissue id' in clinical_data:
        if not pd.isna(clinical_data['tissue id']).all():
            df = clinical_data[['biological_sample external_id', 'tissue id']].drop_duplicates(keep='first').reset_index(drop=True)
            df.columns = ['START_ID', 'END_ID']
            df['TYPE'] = 'FROM_TISSUE'

    return df


def extract_subject_disease_rels(clinical_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE'])
    if 'disease id' in clinical_data:
        if not pd.isna(clinical_data['disease id']).all():
            clinical_data['disease id'] = clinical_data['disease id'].astype(str)
            df = pd.DataFrame(clinical_data['disease id'].str.split(separator).tolist(), index=clinical_data['subject external_id']).stack()
            df = df.reset_index([0, 'subject external_id']).replace('nan', np.nan).dropna().drop_duplicates(keep='first')
            df.columns = ['START_ID', 'END_ID']
            df['TYPE'] = 'HAS_DISEASE'

    return df


def extract_subject_intervention_rels(clinical_data, separator='|'):
    df = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE', 'in_combination', 'response'])
    if 'had_intervention' in clinical_data:
        if not pd.isna(clinical_data['had_intervention']).all():
            interventions = clinical_data.set_index('subject external_id')['had_intervention'].astype(str).str.split(separator, expand=True).stack().str.strip().reset_index(level=1, drop=True)
            types = clinical_data.set_index('subject external_id')['had_intervention_type'].astype(str).str.split(separator, expand=True).stack().str.strip().reset_index(level=1, drop=True)
            combi = clinical_data.set_index('subject external_id')['had_intervention_in_combination'].astype(str).str.split(separator, expand=True).stack().str.strip().reset_index(level=1, drop=True)
            response = clinical_data.set_index('subject external_id')['had_intervention_response'].astype(str).str.split(separator, expand=True).stack().str.strip().reset_index(level=1, drop=True)
            df = pd.concat([interventions, types, combi, response], axis=1,  join='inner')
            df = df.reset_index()
            df.columns = ['START_ID', 'END_ID', 'type', 'in_combination', 'response']
            df.insert(loc=2, column='TYPE', value='HAD_INTERVENTION')
            
    return df


def extract_biological_sample_clinical_variables_rels(clinical_data):
    df_quant = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE', 'value'])
    df_state = pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE', 'value'])
    if 'biological_sample external_id' in clinical_data:
        df = clinical_data.set_index('biological_sample external_id').copy()
        df.columns = [i.split()[-1] for i in df.columns]
        df.columns = df.columns.str.extract(r'.*\((.*)\).*')[0].tolist()
        df_quant = df._get_numeric_data()
        df_state = df.loc[:, ~df.columns.isin(df_quant.columns.tolist())]
        if not df_quant.empty:
            df_quant = df_quant.stack().reset_index().drop_duplicates(keep='first').dropna()
            df_quant.columns = ['START_ID', 'END_ID', 'value']
            df_quant = df_quant.drop_duplicates()
            df_quant.insert(loc=2, column='TYPE', value='HAS_QUANTIFIED_CLINICAL')
        if not df_state.empty:
            df_state = df_state.stack().reset_index().drop_duplicates(keep='first').dropna()
            df_state.columns = ['START_ID', 'END_ID', 'value']
            df_state = df_state.drop_duplicates()
            df_state.insert(loc=2, column='TYPE', value='HAS_CLINICAL_STATE')

    return df_state, df_quant
