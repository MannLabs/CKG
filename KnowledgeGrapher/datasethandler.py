import pandas as pd
from collections import defaultdict
import numpy as np
import config
import sys
import ontologieshandler as oh
import os.path
import re

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
    data = pd.read_excel(open(uri), index_col=None, na_values=['NA'], convert_float = False)

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
            if fields > 2:
                timepoint = " " + fields[2]
            ident = value + " " + subject + timepoint
            subjectDict[ident].append(c)
    return subjectDict

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
def parseClinicalDataset(projectId, configuration, dataDir):
    '''This function parses clinical data from subjects in the project
    Input: uri of the clinical data file. Format: Subjects as rows, clinical variables as columns
    Output: pandas DataFrame with the same input format but the clinical variables mapped to the
    right ontology (defined in config), i.e. type = -40 -> SNOMED CT'''
    
    dfile = configuration['file']
    filepath = os.path.join(dataDir, os.path.join(projectId, dfile))
    data = readDataset(filepath)
    data['subject id'] = data['subject id'].astype('int64')
    data = data.set_index('subject id')
    
    return data

########### Proteomics Datasets ############
def parseProteomicsDataset(projectId, configuration, dataDir):
    datasets = {}
    for ftype in configuration:
        if ftype == "directory":
            continue
        datasetConfig = configuration[ftype]
        dfile = datasetConfig['file']
        filepath = os.path.join(dataDir, os.path.join(projectId,dfile))
        data, regex = loadProteomicsDataset(filepath, datasetConfig)
        data = data.sort_index()
        log = datasetConfig['log']
        subjectDict = extractSubjectReplicates(data, regex)
        delCols = []
        for subject in subjectDict:
            delCols.extend(subjectDict[subject])
            aux = data[subjectDict[subject]]
            data[subject] = calculateMedianReplicates(aux, log)
        datasets[ftype] = data.drop(delCols, 1)
    return datasets

###############################
#           Extractors        # 
###############################
########### Clinical Variables Datasets ############
def extractSubjectClinicalVariablesRelationships(data):
    cols = list(data.columns)
    #if "group" in data.columns:
    #    cols.remove("group")
    #    data = data[cols]
    data = data.stack()
    data = data.reset_index()
    data.columns = ['START_ID', 'END_ID', "value"]
    data['TYPE'] = "HAS_QUANTIFIED_CLINICAL"
    data = data[['START_ID', 'END_ID','TYPE', 'value']]

    return data

########### Proteomics Datasets ############
def extractPeptideSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["peptides"]["valueCol"])
    data = data.reset_index()
    
    aux.columns = [c.split(" ")[1] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Subject", "value"]
    aux['TYPE'] = "HAS_QUANTIFIED_PEPTIDE"
    aux.columns = ['END_ID', 'START_ID',"value", 'TYPE']
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"]]
    aux['START_ID'] = aux['START_ID'].astype('int64')
    
    return aux

def extractPeptideProteinRelationships(data, configuration):
    aux =  data[configuration["peptides"]["proteinCol"]]
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Protein"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux.columns = ['START_ID', 'END_ID', 'TYPE']
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    return aux

def extractProteinSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["proteins"]["valueCol"])
    aux.columns = [c.split(" ")[2] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux['TYPE'] = "HAS_QUANTIFIED_PROTEIN"
    aux.columns = ['END_ID', 'START_ID',"value", 'TYPE']
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"]]
    aux['START_ID'] = aux['START_ID'].astype('int64')
    
    return aux

def extractPTMSubjectRelationships(data, modification):
    code = config["modifications"][modification]["code"]
    aux = data.copy()
    aux["modId"] = aux["Protein"].str + "_" + aux["Amino acid"].str + aux["Positions"].str
    aux = aux.set_index("modId")
    aux =  aux.filter(regex = 'Intensity')
    aux.columns = [c.split(" ")[2] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux['TYPE'] = "HAD_QUANTIFIED_"+modification.upper()
    aux['code'] = code
    aux.columns = ['END_ID('+modification+')', 'START_ID(Sample)', "intensity", 'TYPE', 'code']
    aux = aux[['START_ID', 'END_ID', 'TYPE', "intensity", 'code']]
    
    return aux

#Not sure this is necessary -- Would correspond to the relationship "IS_MODIFIED_IN" 
#(MODIFIED needs to be substituted by the modification)
def extractPTMProteinRelationships(data, modification):
    code = config["modifications"][modification]["code"]
    aux = data.copy()
    aux["modId"] = aux["Protein"].str + "_" + aux["Amino acid"].str + aux["Positions"].str
    aux = aux.reset_index()
    aux = aux[["Protein", "modId"]]
    aux['TYPE'] = "IS_MODIFIED_AT"
    aux['code'] = code
    aux = aux[['START_ID', 'END_ID','TYPE', 'code']]
    
    return aux

def extractProjectSampleRelationships(data, projectId):
    samples = list(data['START_ID'])
    aux = pd.DataFrame({'START_ID': projectId , 'END_ID': samples, 'TYPE': "HAS_ENROLLED"})
    aux = aux[['START_ID', 'END_ID','TYPE']]
    aux['START_ID'] = aux['START_ID'].astype(str)
    
    return aux

############ Whole Exome Sequencing Datasets ##############
def extractWESSubjectRelationships(data):
    pass

############################
#           Loaders        # 
############################
########### Proteomics Datasets ############
def loadProteomicsDataset(uri, configuration):
    ''' This function gets the molecular data from a proteomics experiment.
        Input: uri of the processed file resulting from MQ
        Output: pandas DataFrame with the columns and filters defined in config.py '''

    aux = None
    #Get the columns from config and divide them into simple or regex columns
    columns = configuration["columns"]
    regexCols = [c for c in columns if '+' in c]
    columns = set(columns).difference(regexCols)
    
    #Read the filters defined in config, i.e. reverse, contaminant, etc.
    filters = configuration["filters"]
    proteinCol = configuration["proteinCol"]
    indexCol = configuration["indexCol"]
    if "geneCol" in configuration:
        geneCol = configuration["geneCol"]
    
    #Read the data from file
    data = readDataset(uri)

    #Apply filters
    data = data[data[filters].isnull().all(1)]
    data = data.drop(filters, axis=1)

    #Select all protein form protein groups, i.e. P01911;Q29830;Q9MXZ4
    # P01911
    # Q29830
    # Q9MXZ4
    s = data[proteinCol].str.split(';').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    del data[proteinCol]
    data = data.join(s.to_frame(proteinCol))
    #proteins = data[proteinCol].str.split(';').apply(pd.Series,1)[0]
    #data[proteinCol] = proteins
    data = data.set_index(indexCol)
    filters.append(indexCol)
    columns = set(columns).difference(filters)

    #Get columns using regex
    for regex in regexCols:
        r = re.compile(regex)
        columns.update(set(filter(r.match, data.columns)))
    #Add simple and regex columns into a single DataFrame
    data = data[list(columns)]
    
    return data, regexCols

############ Whole Exome Sequencing #############
def loadWESDataset(uri, configuration):
    ''' This function gets the molecular data from a Whole Exome Sequencing experiment.
        Input: uri of the processed file resulting from the WES analysis pipeline. The resulting
        annotated VCF file from Mutect
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    aux = None
    
    #Get the columns from config and divide them into simple or regex columns
    columns = configuration["columns"]
    regexCols = [c for c in columns if '+' in c]
    columns = set(columns).difference(regexCols)
    
########################################
#          Generate graph files        # 
########################################
def generateDatasetImports(projectId, dataType):
    dataDir = config.dataTypes[dataType]["directory"]
    configuration = config.dataTypes[dataType]
    if dataType == "clinicalData":
        data = parseClinicalDataset(projectId, configuration, dataDir)
        dataRows = extractSubjectClinicalVariablesRelationships(data)
        generateGraphFiles(dataRows,'clinical', projectId)
    elif dataType == "proteomicsData":
        data = parseProteomicsDataset(projectId, configuration, dataDir)
        for dtype in data:
            if dtype == "proteins":
                dataRows = extractProteinSubjectRelationships(data[dtype], configuration)
                proSamRows = extractProjectSampleRelationships(dataRows, projectId)
                generateGraphFiles(proSamRows,'project', projectId)
            elif dtype == "peptides":
                dataRows = extractPeptideSubjectRelationships(data[dtype], configuration) 
                ppDataRows = extractPeptideProteinRelationships(data[dtype], configuration)
                generateGraphFiles(ppDataRows,"PeptideProtein", projectId)
            generateGraphFiles(dataRows,dtype, projectId)
            
def generateGraphFiles(data, dataType, projectId):
    importDir = config.datasetsImportDirectory
    outputfile = os.path.join(importDir, projectId+"_"+dataType.lower()+".csv")
      
    data.to_csv(path_or_buf=outputfile, 
            header=True, index=False, quotechar='"', 
            line_terminator='\n', escapechar='\\')


if __name__ == "__main__":
    #uri = sys.argv[1]
    generateDatasetImports('000000000001', 'proteomicsData')
    generateDatasetImports('000000000001', 'clinicalData')
    
    
