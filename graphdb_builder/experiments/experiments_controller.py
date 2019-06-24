# from graphdb_builder.ontologies import ontologies_controller as oh
from graphdb_builder import builder_utils
import sys
import re
import os.path
import pandas as pd
from dask import dataframe as dd
import numpy as np
from collections import defaultdict
from report_manager.queries import query_utils
from graphdb_connector import connector
import config.ckg_config as ckg_config
import ckg_utils
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="experiments_controller")

driver = connector.getGraphDatabaseConnectionConfiguration()

try:
    config = builder_utils.setup_config('experiments')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

#########################
# General functionality #
#########################
def readDataset(uri):
    if uri.endswith('.xlsx'):
        data = readDataFromExcel(uri)
    elif uri.endswith(".csv") or uri.endswith(".tsv") or uri.endswith(".txt"):
        if uri.endswith(".tsv") or uri.endswith(".txt"):
            data = readDataFromTXT(uri)
        else:
            data = readDataFromCSV(uri)

    return data

def readDataFromCSV(uri):
    #Read the data from csv file
    data = pd.read_csv(uri, sep = ',', low_memory=False)

    return data

def readDataFromTXT(uri):
    #Read the data from tsv or txt file
    data = pd.read_csv(uri, sep = '\t', low_memory=False)

    return data

def readDataFromExcel(uri):
    #Read the data from Excel file
    data = pd.read_excel(uri, index_col=None, na_values=['NA'], convert_float = True)

    return data

def extractSubjectReplicates(data, regex):
    subjectDict = defaultdict(list)
    for r in regex:
        columns = data.filter(regex = r).columns
        for c in columns:
            fields  = c.split('_')
            value = " ".join(fields[0].split(' ')[0:-1])
            subject = fields[1]
            timepoint = ""
            if len(fields) > 2:
                timepoint = " " + fields[2]
            ident = value + " " + subject + timepoint
            subjectDict[ident].append(c)
    return subjectDict


def extractAttributes(data, attributes):
    auxAttr_col = pd.DataFrame(index = data.index)
    auxAttr_reg = pd.DataFrame(index = data.index)
    cCols = []
    regexCols = []
    for ctype in attributes:
        if ctype =="regex":
            for r in attributes[ctype]:
                regexCols.append(r.replace(' ','_'))
                auxAttr_reg = auxAttr_reg.join(data.filter(regex = r))
        else:
            auxAttr_col = auxAttr_col.join(data[attributes[ctype]])
            cCols = [c.replace(' ','_').replace('-','') for c in attributes[ctype]]
    
    reg_attr_index = auxAttr_reg.index.name
    col_attr_index = auxAttr_col.index.name
    auxAttr_reg = auxAttr_reg.reset_index().drop_duplicates().set_index(reg_attr_index)
    auxAttr_col = auxAttr_col.reset_index().drop_duplicates().set_index(col_attr_index)
    
    return (auxAttr_col,cCols), (auxAttr_reg,regexCols)

def mergeRegexAttributes(data, attributes, index):
    if not attributes.empty:
        attributes.columns = [c.split(" ")[1] for c in attributes.columns]
        attributes = attributes.stack()
        attributes = attributes.reset_index()
        attributes.columns = ["c"+str(i) for i in range(len(attributes.columns))]
        data = pd.merge(data, attributes, on = index)
        del(attributes)

    return data

def mergeColAttributes(data, attributes, index):
    if not attributes.empty:
        attributes = attributes.reset_index()
        data = pd.merge(data, attributes, on=index)
        del(attributes)
        data = data.reset_index()

    return data

def calculateMedianReplicates(data, log = "log2"):
    if log == "log2":
        data = data.applymap(lambda x:np.log2(x) if x > 0 else np.nan)
    elif log == "log10":
        data = data.applymap(lambda x:np.log10(x) if x > 0 else np.nan)
    median = data.median(axis=1).sort_values(axis=0, ascending= True, na_position = 'first').to_frame()
    median  = median.sort_index()[0]
    return median

def updateGroups(data, groups):
    #del groups.index.name
    data = data.join(groups.to_frame(), on='START_ID')

    return data

############################
#           Parsers        #
############################
########### Clinical Variables Datasets ############
def parseClinicalDataset(projectId, configuration, dataDir, key='project'):
    '''This function parses clinical data from subjects in the project
    Input: uri of the clinical data file. Format: Subjects as rows, clinical variables as columns
    Output: pandas DataFrame with the same input format but the clinical variables mapped to the
    right ontology (defined in config), i.e. type = -40 -> SNOMED CT'''

    if key == 'project':
        data_file = configuration['file_pro'].replace('PROJECTID', projectId)
    elif key == 'clinical':
        data_file = configuration['file_cli'].replace('PROJECTID', projectId)

    filepath = os.path.join(dataDir, data_file)
    data = None
    if os.path.isfile(filepath):
        data = readDataset(filepath)

    return data

########### Proteomics Datasets ############
def parseProteomicsDataset(projectId, configuration, dataDir):
    dataset = None
    dfile = configuration['file']
    filepath = os.path.join(dataDir, dfile)
    if os.path.isfile(filepath):
        data, regex = loadProteomicsDataset(filepath, configuration)
        data = data.sort_index()
        log = configuration['log']
        subjectDict = extractSubjectReplicates(data, regex)
        delCols = []
        for subject in subjectDict:
            delCols.extend(subjectDict[subject])
            aux = data[subjectDict[subject]]
            data[subject] = calculateMedianReplicates(aux, log)
        dataset = data.drop(delCols, 1)
    return dataset

########### Genomics Datasets ############
def parseWESDataset(projectId, configuration, dataDir):
    datasets = {}
    files = builder_utils.listDirectoryFiles(dataDir)
    for dfile in files:
        filepath = os.path.join(dataDir, dfile)
        if os.path.isfile(filepath):
            sample, data = loadWESDataset(filepath, configuration)
            datasets[sample] = data

    return datasets

###############################
#           Extractors        #
###############################
########### Clinical Variables Datasets ############
def extractProjectInfo(project_data):
    df = project_data.copy()
    df.columns = ['internal_id', 'name', 'acronym', 'description', 'subjects', 'datatypes', 'timepoints', 'disease', 'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date', 'status', 'external_id']
    return df

def extractResponsibleRelationships(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['responsible'][0]):
        return None
    else:
        df = pd.DataFrame(data.responsible.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
        df['END_ID'] = data['external_id'][0]
        df['TYPE'] = 'IS_RESPONSIBLE'
        return df

def extractParticipantRelationships(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['participant'][0]):
        return None
    else:
        df = pd.DataFrame(data.participant.str.split(separator).tolist()).T.rename(columns={0:'START_ID'})
        df['END_ID'] = data['external_id'][0]
        df['TYPE'] = 'PARTICIPATES_IN'
        return df
    
def extractProjectTissueRelationships(driver, project_data, separator='|'):
    data = project_data.copy()
    tissue_ids = []
    if pd.isna(data['tissue'][0]):
        return None
    else:
        for tissue in data['tissue'][0].split(separator):
            tissue_ids.append(query_utils.map_node_name_to_id(driver, 'Tissue', str(tissue)))
        df = pd.DataFrame(tissue_ids, columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=data['external_id'][0])
        df['TYPE'] = 'STUDIES_TISSUE'
        return df
    
def extractProjectDiseaseRelationships(driver, project_data, separator='|'):
    data = project_data.copy()
    disease_ids = []
    if pd.isna(data['disease'][0]):
        return None
    else:
        for disease in data['disease'][0].split(separator):
            disease_ids.append(query_utils.map_node_name_to_id(driver, 'Disease', str(disease)))
        df = pd.DataFrame(disease_ids, columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=data['external_id'][0])
        df['TYPE'] = 'STUDIES_DISEASE'
        return df

def extractProjectInterventionRelationships(project_data, separator='|'):
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

def extractTimepoints(project_data, separator='|'):
    data = project_data.copy()
    if pd.isna(data['timepoints'][0]):
        return pd.DataFrame(columns=['ID', 'units', 'type'])
    else:
        df = pd.DataFrame(data['timepoints'][0].replace(' ','').split(separator))
        df = df[0].str.extract(r'([\-\d]+)([a-zA-Z]+)', expand=True)
        df.columns = ['ID', 'units']
        df['type'] = 'Timepoint'
        return df

def extractProjectSubjectRelationships(project_data, clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['subject id']).any():
        return None
    else:
        df = pd.DataFrame(data['subject id'].dropna().unique(), columns=['END_ID'])
        df.insert(loc=0, column='START_ID', value=project_data['external_id'][0])
        df['TYPE'] = 'HAS_ENROLLED'
        return df

def extractSubjectIds(project_data, clinical_data):
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

def extractBiologicalSampleSubjectRelationships(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['biological_sample id']).any():
        return None
    else:
        df = data[['biological_sample id', 'subject id']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID']
        df['TYPE'] = 'BELONGS_TO_SUBJECT'
        return df

def extractBiologicalSamplesInfo(clinical_data):
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

def extractAnalyticalSamplesInfo(clinical_data):
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

def extractBiologicalSampleAnalyticalSampleRelationships(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['analytical_sample id']).any():
        return None
    else:
        df = data[['biological_sample id', 'analytical_sample id', 'analytical_sample quantity', 'analytical_sample quantity_units']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID', 'quantity', 'quantity_units']
        df.insert(loc=2, column='TYPE', value='SPLITTED_INTO')
        return df

def extractBiologicalSampleTimepointRelationships(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['timepoint']).all():
        return pd.DataFrame(columns=['START_ID', 'END_ID', 'TYPE','timepoint_units', 'intervention'])
    else:
        df = data[['biological_sample id', 'timepoint', 'timepoint units', 'intervention id']].drop_duplicates(keep='first').reset_index(drop=True)
        df['intervention id'] = df['intervention id'].replace(np.nan, 0).astype('int64').astype('str').replace('0', np.nan)
        df.columns = ['START_ID', 'END_ID', 'timepoint_units', 'intervention']
        df.insert(loc=2, column='TYPE', value='SAMPLE_AT_TIMEPOINT')
        return df

def extractBiologicalSampleTissueRelationships(clinical_data):
    data = clinical_data.copy()
    if pd.isna(data['tissue id']).all():
        return None
    else:
        df = data[['biological_sample id', 'tissue id']].drop_duplicates(keep='first').reset_index(drop=True)
        df.columns = ['START_ID', 'END_ID']
        df['TYPE'] = 'FROM_TISSUE'
        return df

def extractSubjectDiseaseRelationships(clinical_data, separator='|'):
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

def extractBiologicalSampleGroupRelationships(clinical_data):
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

def extractBiologicalSampleClinicalVariablesRelationships(clinical_data):
    data = clinical_data.set_index('biological_sample id').copy()
    df = data.loc[:,'grouping2':].drop('grouping2', axis=1)
    # df['intervention id'] = data['intervention id'].replace(np.nan, 0).astype('int64').astype('str').replace('0', np.nan)
    # df['biological_sample id'] = data['biological_sample id']
    # df = df.set_index('biological_sample id')
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

# def extractSubjectClinicalVariablesRelationships(data):
#     df = data.copy()
#     intervention = None
#     if 'intervention id' in df.columns and df['intervention id'].dropna().empty != True:
#         intervention = df['intervention id'].to_frame()
#         intervention = intervention.reset_index()
#         intervention.columns = ['START_ID', 'END_ID']
#         intervention['END_ID'] = intervention['END_ID'].astype('int64')
#         intervention['value'] = True
#     df = df.set_index('subject id')
#     df = df.drop([i for i in df.columns if str(i).endswith(' id')], axis=1)
#     df = df.stack()
#     df = df.reset_index()
#     df.columns = ['START_ID', 'END_ID', "value"]
#     if intervention is not None:
#         df = df.append(intervention, sort=True)
#     df['TYPE'] = "HAS_CLINICAL_STATE"
#     df = df[['START_ID', 'END_ID','TYPE', 'value']]
#     df['END_ID'] = df['END_ID'].apply(lambda x: int(x) if isinstance(x,float) else x)
#     df = df[df['value'].apply(lambda x: isinstance(x, str))]
#     df = df.drop_duplicates(keep='first').dropna()

#     return df

# def extractBiologicalSampleClinicalVariablesRelationships(clinical_data):
#     df = clinical_data.copy()
#     if 'biological_sample id' in df.columns:
#         df = df.set_index('biological_sample id')
#         df = df._get_numeric_data()
#         cols = [i for i in df.columns if str(i).endswith(' id')]
#         df = df.drop(cols, axis=1)
#         df = df.stack().reset_index()
#         df.columns = ['START_ID', 'END_ID', 'value']
#         df.insert(loc=2, column='TYPE', value='HAS_QUANTIFIED_CLINICAL')
#         df['END_ID'] = df['END_ID'].apply(lambda x: int(x) if isinstance(x,float) else x)
#         df = df.drop_duplicates(keep='first').dropna()
#         return df
#     else:
#         return None


########### Proteomics Datasets ############
############## ProteinModification entity ####################
def extractModificationProteinRelationships(data, configuration):
    modificationId = configuration["modId"]
    cols = configuration["positionCols"]
    aux = data[cols]
    aux = aux.reset_index()
    aux.columns = ["START_ID", "position", "residue"]
    aux["END_ID"] = modificationId
    aux['TYPE'] = "HAS_MODIFICATION"
    aux = aux[['START_ID', 'END_ID','TYPE', "position", "residue"]]
    aux['position'] = aux['position'].astype('int64')
    aux = aux.drop_duplicates()

    return aux

def extractProteinModificationSubjectRelationships(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    data = data.reset_index()
    data["END_ID"] = data[proteinCol].map(str) + "_" + data[positionCols[1]].map(str) + data[positionCols[0]].map(str) + '-' +configuration["mod_acronym"]
    data = data.set_index("END_ID")
    newIndexdf = data.copy()
    data = data.drop(cols, axis=1)
    data =  data.filter(regex = configuration["valueCol"].replace("\\\\","\\"))
    data.columns = [c.split(" ")[1] for c in data.columns]
    data = data.stack()
    data = data.reset_index()
    data.columns = ["c"+str(i) for i in range(len(data.columns))]
    columns = ['END_ID', 'START_ID',"value"]
    attributes = configuration["attributes"]
    (cAttributes,cCols), (rAttributes,regexCols) = extractAttributes(newIndexdf, attributes)
    if not rAttributes.empty:
        data = mergeRegexAttributes(data, rAttributes, ["c0","c1"])
        columns.extend(regexCols)
    if not cAttributes.empty:
        data = mergeColAttributes(data, cAttributes, "c0")
        columns.extend(cCols)

    data['TYPE'] = "HAS_QUANTIFIED_MODIFIED_PROTEIN"
    columns.append("TYPE")
    data.columns = columns
    data = data[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    data = data.drop_duplicates()

    return data

def extractProteinProteinModificationRelationships(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["END_ID"] =  aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.drop(positionCols, axis=1)
    aux = aux.set_index("END_ID")
    aux = aux.reset_index()
    aux.columns = ["END_ID", "START_ID"]
    aux['TYPE'] = "HAS_MODIFIED_SITE"
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    aux = aux.drop_duplicates()

    return aux

def extractPeptideProteinModificationRelationships(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [sequenceCol, proteinCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["END_ID"] =  aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.drop([proteinCol] + positionCols, axis=1)
    aux = aux.set_index("END_ID")
    aux = aux.reset_index()
    aux.columns = ["END_ID", "START_ID"]
    aux["START_ID"] = aux["START_ID"].str.upper()
    aux['TYPE'] = "HAS_MODIFIED_SITE"
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    aux = aux.drop_duplicates()

    return aux

def extractProteinModifications(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [proteinCol, sequenceCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["ID"] =  aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.set_index("ID")
    aux = aux.reset_index()
    aux[sequenceCol] = aux[sequenceCol].str.replace('_', '-')
    aux["source"] = "experimentally_identified"
    aux.columns = ["ID", "protein", "sequence_window", "position", "residue", "source"]
    aux = aux.drop_duplicates()

    return aux

def extractProteinModificationsModification(data, configuration):
    modID = configuration["modId"]
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [proteinCol, sequenceCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["START_ID"] =  aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux["END_ID"] = modID
    aux = aux[["START_ID", "END_ID"]]

    return aux

################# Peptide entity ####################
def extractPeptides(data, configuration):
    aux = data.copy()
    modid = configuration["type"]
    aux["type"] = modid
    aux = aux["type"]
    aux = aux.reset_index()
    aux = aux.groupby(aux.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    aux.columns = ["ID", "type", "count"]
    aux = aux.drop_duplicates()

    return aux

def extractPeptideSubjectRelationships(data, configuration):
    data = data[~data.index.duplicated(keep='first')]
    aux =  data.filter(regex = configuration["valueCol"].replace("\\\\","\\"))
    attributes = configuration["attributes"]
    aux.columns = [c.split(" ")[1] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["c"+str(i) for i in range(len(aux.columns))]
    columns = ['END_ID', 'START_ID',"value"]

    (cAttributes,cCols), (rAttributes,regexCols) = extractAttributes(data, attributes)
    if not rAttributes.empty:
        aux = mergeRegexAttributes(aux, rAttributes, ["c0","c1"])
        columns.extend(regexCols)
    if not cAttributes.empty:
        aux = mergeColAttributes(aux, cAttributes, "c0")
        columns.extend(cCols)

    aux['TYPE'] = "HAS_QUANTIFIED_PEPTIDE"
    columns.append("TYPE")
    aux.columns = columns
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    aux = aux.drop_duplicates()

    return aux

def extractPeptideProteinRelationships(data, configuration):
    cols = [configuration["proteinCol"]]
    cols.extend(configuration["positionCols"])
    aux =  data[cols]
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Protein", "Start", "End"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux['source'] = 'experimentally_identified'
    aux.columns = ['START_ID', 'END_ID', "start", "end", 'TYPE', 'source']
    aux = aux[['START_ID', 'END_ID', 'TYPE', 'source']]
    return aux

################# Protein entity #########################
def extractProteinSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["valueCol"].replace("\\\\","\\"))
    attributes = configuration["attributes"]
    aux.columns = [c.split(" ")[2] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["c"+str(i) for i in range(len(aux.columns))]
    columns = ['END_ID', 'START_ID',"value"]
    
    (cAttributes,cCols), (rAttributes,regexCols) = extractAttributes(data, attributes)
    
    if not rAttributes.empty:
        aux = mergeRegexAttributes(aux, rAttributes, ["c0","c1"])
        columns.extend(regexCols)
    if not cAttributes.empty:
        aux = mergeColAttributes(aux, cAttributes, "c0")
        columns.extend(cCols)
    
    aux['TYPE'] = "HAS_QUANTIFIED_PROTEIN"
    columns.append("TYPE")
    aux.columns = columns
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    
    return aux

############ Whole Exome Sequencing Datasets ##############
def extractWESRelationships(data, configuration):
    entityAux = data.copy()
    entityAux = entityAux.set_index("ID")

    variantAux = data.copy()
    variantAux = variantAux.rename(index=str, columns={"ID": "START_ID"})
    variantAux["END_ID"] = variantAux["START_ID"]
    variantAux = variantAux[["START_ID", "END_ID"]]
    variantAux["TYPE"] = "IS_KNOWN_VARIANT"
    variantAux = variantAux.drop_duplicates()
    variantAux = variantAux.dropna(how="any")
    variantAux = variantAux[["START_ID", "END_ID", "TYPE"]]

    sampleAux = data.copy()
    sampleAux = sampleAux.rename(index=str, columns={"ID": "END_ID", "sample": "START_ID"})
    sampleAux["TYPE"] = "HAS_MUTATION"
    sampleAux = sampleAux[["START_ID", "END_ID", "TYPE"]]

    geneAux = data.copy()
    geneAux = geneAux.rename(index=str, columns={"ID": "START_ID", "gene": "END_ID"})
    geneAux["TYPE"] = "VARIANT_FOUND_IN_GENE"
    geneAux = geneAux[["START_ID", "END_ID", "TYPE"]]
    s = geneAux["END_ID"].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del geneAux["END_ID"]
    aux = s.to_frame("END_ID")
    geneAux = geneAux.join(aux)

    chrAux = data.copy()
    chrAux = chrAux.rename(index=str, columns={"ID": "START_ID", "chr": "END_ID"})
    chrAux["END_ID"] = chrAux["END_ID"].str.replace("chr",'')
    chrAux["TYPE"] = "VARIANT_FOUND_IN_CHROMOSOME"
    chrAux = chrAux[["START_ID", "END_ID", "TYPE"]]

    return entityAux, variantAux, sampleAux, geneAux, chrAux

############################
#           Loaders        #
############################
########### Proteomics Datasets ############
def loadProteomicsDataset(uri, configuration):
    ''' This function gets the molecular data from a proteomics experiment.
        Input: uri of the processed file resulting from MQ
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    #Get the columns from config and divide them into simple or regex columns
    columns = configuration["columns"]
    regexCols = [c.replace("\\\\","\\") for c in columns if '+' in c]
    columns = set(columns).difference(regexCols)

    #Read the filters defined in config, i.e. reverse, contaminant, etc.
    filters = configuration["filters"]
    indexCol = configuration["indexCol"]
    
    #Read the data from file
    data = readDataset(uri)
    #Apply filters
    data = data[data[filters].isnull().all(1)]
    data = data.drop(filters, axis=1)
    data = expand_groups(data, configuration)
    columns = set(columns).difference(filters)
    columns.remove(indexCol)
    
    #Get columns using regex
    for regex in regexCols:
        r = re.compile(regex)
        columns.update(set(filter(r.match, data.columns)))
    #Add simple and regex columns into a single DataFrame
    data = data[list(columns)]
    
    return data, regexCols

def expand_groups(data, configuration):
    ddata = dd.from_pandas(data, 10)
    ddata = ddata.map_partitions(lambda df: df.drop(configuration["proteinCol"], axis=1).join(df[configuration["proteinCol"]].str.split(';', expand=True).stack().reset_index(drop=True, level=1).rename(configuration["proteinCol"])))
    if "multipositions" in configuration:
        ddata = ddata.map_partitions(lambda df: df.drop(configuration["multipositions"], axis=1).join(df[configuration["multipositions"]].str.split(';', expand=True).stack().reset_index(drop=True, level=1).rename(configuration["multipositions"])))
    data = ddata.compute()
    data["is_razor"] = ~ data[configuration["groupCol"]].duplicated()
    data = data.set_index(configuration["indexCol"])

    return data

############ Whole Exome Sequencing #############
def loadWESDataset(uri, configuration):
    ''' This function gets the molecular data from a Whole Exome Sequencing experiment.
        Input: uri of the processed file resulting from the WES analysis pipeline. The resulting
        Annovar annotated VCF file from Mutect (sampleID_mutect_annovar.vcf)
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    aux = uri.split("/")[-1].split("_")
    sample = aux[0]
    #Get the columns from config
    columns = configuration["columns"]
    #Read the data from file
    data = readDataset(uri)
    if configuration['filter'] in data.columns:
        data = data.loc[data[configuration['filter']], :]
    data = data[columns]
    data["sample"] = aux[0]
    data["variant_calling_method"] = aux[1]
    data["annotated_with"] = aux[2].split('.')[0]
    data["alternative_names"] = data[configuration["alt_names"]]
    data = data.drop(configuration["alt_names"], axis = 1)
    data = data.iloc[1:]
    data = data.replace('.', np.nan)
    data["ID"] = data[configuration["id_fields"]].apply(lambda x: str(x[0])+":g."+str(x[1])+str(x[2])+'>'+str(x[3]), axis=1)
    data.columns = configuration['new_columns']
    return sample, data

########################################
#          Generate graph files        #
########################################
def generateDatasetImports(projectId, dataType):
    stats = set()
    try:
        if dataType in config["dataTypes"]:
            if "directory" in config["dataTypes"][dataType]:
                dataDir = config["dataTypes"][dataType]["directory"].replace("PROJECTID", projectId)
                configuration = config["dataTypes"][dataType]
                if dataType == "clinicaly":
                    separator = configuration["separator"]
                    project_data = parseClinicalDataset(projectId, configuration, dataDir, key='project')
                    clinical_data = parseClinicalDataset(projectId, configuration, dataDir, key='clinical')
                    if project_data is not None and clinical_data is not None:
                        dataRows = extractProjectInfo(project_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'', projectId, stats, d = dataType)
                        dataRows = extractResponsibleRelationships(project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'responsibles', projectId, stats, d = dataType)
                        dataRows = extractParticipantRelationships(project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'participants', projectId, stats, d = dataType)
                        dataRows = extractProjectTissueRelationships(driver, project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'studies_tissue', projectId, stats, d = dataType)
                        dataRows = extractProjectDiseaseRelationships(driver, project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'studies_disease', projectId, stats, d = dataType)
                        dataRows = extractProjectInterventionRelationships(project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'studies_intervention', projectId, stats, d = dataType)
                        dataRows = extractTimepoints(project_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'timepoint', projectId, stats, d = dataType)
                        dataRows = extractProjectSubjectRelationships(project_data, clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'project', projectId, stats, d = dataType)
                        dataRows = extractSubjectIds(project_data, clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'subjects', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSampleSubjectRelationships(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'subject_biosample', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSamplesInfo(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'biological_samples', projectId, stats, d = dataType)
                        dataRows = extractAnalyticalSamplesInfo(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'analytical_samples', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSampleAnalyticalSampleRelationships(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'biosample_analytical', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSampleTimepointRelationships(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'biological_sample_at_timepoint', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSampleTissueRelationships(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'biosample_tissue', projectId, stats, d = dataType)
                        dataRows = extractSubjectDiseaseRelationships(clinical_data, separator=separator)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'disease', projectId, stats, d = dataType)
                        dataRows = extractBiologicalSampleGroupRelationships(clinical_data)
                        if dataRows is not None:
                            generateGraphFiles(dataRows,'groups', projectId, stats, d = dataType)
                        dataRows1, dataRows2 = extractBiologicalSampleClinicalVariablesRelationships(clinical_data)
                        if dataRows1 is not None and dataRows2 is not None:
                            generateGraphFiles(dataRows1,'clinical_state', projectId, stats, d = dataType)
                            generateGraphFiles(dataRows2,'clinical_quant', projectId, stats, d = dataType)                       
                elif dataType == "proteomics":
                    for dtype in configuration:
                        if dtype == "directory":
                            continue
                        datasetConfig = configuration[dtype]
                        df = parseProteomicsDataset(projectId, datasetConfig, dataDir)
                        if df is not None:
                            if dtype == "proteins":
                                dataRows = extractProteinSubjectRelationships(df, datasetConfig)
                                generateGraphFiles(dataRows,dtype, projectId, stats)
                            elif dtype == "peptides":
                                dataRows = extractPeptideSubjectRelationships(df, datasetConfig)
                                generateGraphFiles(dataRows, "subject_peptide", projectId, stats)
                                dataRows = extractPeptideProteinRelationships(df, datasetConfig)
                                generateGraphFiles(dataRows,"peptide_protein", projectId, stats)
                                dataRows = extractPeptides(df, datasetConfig)
                                generateGraphFiles(dataRows, dtype, projectId, stats)
                            else:
                                #dataRows = extractModificationProteinRelationships(data[dtype], configuration[dtype])
                                #generateGraphFiles(dataRows,"protein_modification", projectId, ot = 'a')
                                dataRows = extractProteinModificationSubjectRelationships(df, datasetConfig)
                                generateGraphFiles(dataRows, "modifiedprotein_subject", projectId, stats, ot = 'a')
                                dataRows = extractProteinProteinModificationRelationships(df,datasetConfig)
                                generateGraphFiles(dataRows, "modifiedprotein_protein", projectId, stats, ot = 'a')
                                dataRows = extractPeptideProteinModificationRelationships(df,datasetConfig)
                                generateGraphFiles(dataRows, "modifiedprotein_peptide", projectId, stats, ot = 'a')
                                dataRows = extractProteinModifications(df,datasetConfig)
                                generateGraphFiles(dataRows, "modifiedprotein", projectId, stats, ot = 'a')
                                dataRows = extractProteinModificationsModification(df, datasetConfig)
                                generateGraphFiles(dataRows, "modifiedprotein_modification", projectId, stats, ot = 'a')
                elif dataType == "wes":
                    data = parseWESDataset(projectId, configuration, dataDir)
                    if data is not None:
                        somatic_mutations = pd.DataFrame()
                        for sample in data:
                            entities, variantRows, sampleRows, geneRows, chrRows = extractWESRelationships(data[sample], configuration)
                            generateGraphFiles(variantRows, "somatic_mutation_known_variant", projectId, stats, d = dataType)
                            generateGraphFiles(sampleRows, "somatic_mutation_sample", projectId, stats, d = dataType)
                            generateGraphFiles(geneRows, "somatic_mutation_gene", projectId, stats, d = dataType)
                            generateGraphFiles(chrRows, "somatic_mutation_chromosome", projectId, stats, d = dataType)
                            if somatic_mutations.empty:
                                somatic_mutations = entities
                            else:
                                new = set(entities.index).difference(set(somatic_mutations.index))
                                somatic_mutations = somatic_mutations.append(entities.loc[new,:], ignore_index=False)
                        somatic_mutations = somatic_mutations.reset_index()
                        generateGraphFiles(somatic_mutations, "somatic_mutation", projectId, stats, d= dataType, ot = 'w')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Experiment {}: {} file: {}, line: {}".format(projectId, sys.exc_info(), fname, exc_tb.tb_lineno))
        raise Exception("Error when importing experiment {}.\n {}".format(projectId, err))

def generateGraphFiles(data, dataType, projectId, stats, ot = 'w', d = 'proteomics'):
    importDir = os.path.join(config["experimentsImportDirectory"], os.path.join(projectId,d))
    ckg_utils.checkDirectory(importDir)
    if dataType.lower() == '':
        outputfile = os.path.join(importDir, projectId+dataType.lower()+".tsv")
    else:
        outputfile = os.path.join(importDir, projectId+"_"+dataType.lower()+".tsv")
    
    with open(outputfile, ot) as f:
        data.to_csv(path_or_buf = f, sep='\t',
            header=True, index=False, quotechar='"',
            line_terminator='\n', escapechar='\\')
    
    logger.info("Experiment {} - Number of {} relationships: {}".format(projectId, dataType, data.shape[0]))
    stats.add(builder_utils.buildStats(data.shape[0], "relationships", dataType, "Experiment", outputfile))

if __name__ == "__main__":
    pass
