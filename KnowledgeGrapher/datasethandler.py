import pandas as pd
from collections import defaultdict
import numpy as np
import config
import sys
import ontologieshandler as oh
import os.path
import re
import utils

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

########### Genomics Datasets ############
def parseWESDataset(projectId, configuration, dataDir):
    datasets = {}
    files = utils.listDirectoryFiles(os.path.join(dataDir,projectId))
    for dfile in files:
        filepath = os.path.join(dataDir, os.path.join(projectId,dfile))
        sample, data = loadWESDataset(filepath, configuration)
        datasets[sample] = data

    return datasets

###############################
#           Extractors        # 
###############################
########### Clinical Variables Datasets ############
def extractSubjectClinicalVariablesRelationships(data):
    cols = list(data.columns)
    if "group" in data.columns:
        cols.remove("group")
        data = data[cols]
    data = data.stack()
    data = data.reset_index()
    data.columns = ['START_ID', 'END_ID', "value"]
    data['TYPE'] = "HAS_QUANTIFIED_CLINICAL"
    data = data[['START_ID', 'END_ID','TYPE', 'value']]

    return data

def extractSubjectGroupRelationships(data):
    cols = list(data.columns)
    if "group" in data.columns:
        data = data["group"].to_frame()
    data = data.reset_index()
    data.columns = ['START_ID', 'END_ID']
    data['TYPE'] = "BELONGS_TO_GROUP"
    return data

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
    
    return aux

def extractProteinModificationSubjectRelationships(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    
    aux = data.copy()
    aux = aux.reset_index()
    aux["END_ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[0]].map(str) + aux[positionCols[1]].map(str)
    aux = aux.set_index("END_ID") 
    aux = aux.drop(cols, axis=1)
    aux =  aux.filter(regex = configuration["valueCol"])
    aux.columns = [c.split(" ")[1] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["END_ID", "START_ID", "value"]
    aux['TYPE'] = "HAS_QUANTIFIED_PROTEINMODIFICATION"
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"]]
    
    return aux

def extractProteinProteinModificationRelationships(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["START_ID"] =  aux[proteinCol].map(str) + "_" + aux[positionCols[0]].map(str) + aux[positionCols[1]].map(str)
    aux = aux.drop(positionCols, axis=1)
    aux = aux.set_index("START_ID")
    aux = aux.reset_index()
    aux.columns = ["START_ID", "END_ID"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    
    return aux

def extractProteinModifications(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [proteinCol, sequenceCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols] 
    aux["ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[0]].map(str) + aux[positionCols[1]].map(str)
    aux = aux.set_index("ID")
    aux = aux.reset_index()
    aux[sequenceCol] = aux[sequenceCol].str.replace('_', '-')
    aux.columns = ["ID", "protein", "sequence_window", "position", "Amino acid"]
    
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
    aux["START_ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[0]].map(str) + aux[positionCols[1]].map(str)
    aux["END_ID"] = modID
    aux = aux[["START_ID","END_ID"]]
    
    return aux

################# Peptide entity ####################
def extractPeptides(data, configuration):
    aux = data.copy()
    modid = configuration["type"]
    aux["type"] = modid
    aux = aux["type"]
    aux = aux.reset_index()
    aux.columns = ["ID", "type"]

    return aux

def extractPeptideSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["valueCol"])
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
    cols = [configuration["proteinCol"]]
    cols.extend(configuration["positionCols"])
    aux =  data[cols]
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Protein", "Start", "End"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux.columns = ['START_ID', 'END_ID', 'TYPE', "start", "end"]
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    return aux

################# Protein entity #########################
def extractProteinSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["valueCol"])
    aux.columns = [c.split(" ")[2] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux['TYPE'] = "HAS_QUANTIFIED_PROTEIN"
    aux.columns = ['END_ID', 'START_ID',"value", 'TYPE']
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"]]
    aux['START_ID'] = aux['START_ID'].astype('int64')
    
    return aux

def extractProjectSampleRelationships(data, projectId):
    samples = list(data['START_ID'])
    aux = pd.DataFrame({'START_ID': projectId , 'END_ID': samples, 'TYPE': "HAS_ENROLLED"})
    aux = aux[['START_ID', 'END_ID','TYPE']]
    aux['START_ID'] = aux['START_ID'].astype(str)
    
    return aux

############ Whole Exome Sequencing Datasets ##############
def extractWESRelationships(data, configuration):
    data.columns = configuration["new_columns"]
    entityAux = data.copy()
    entityAux = entityAux[configuration["somatic_mutation_attributes"]] 
    entityAux = entityAux.set_index("ID")

    variantAux = data.copy()
    variantAux = variantAux.rename(index=str, columns={"ID": "START_ID"})
    variantAux["END_ID"] = variantAux["alternative_names"]
    variantAux = variantAux[["START_ID", "END_ID"]]
    s = variantAux["END_ID"].str.split(',').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del variantAux["END_ID"]
    variants = s.to_frame("END_ID")
    variantAux = variantAux.join(variants)
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
    s = data[proteinCol].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del data[proteinCol]
    pdf = s.to_frame(proteinCol)
    if "multipositions" in configuration:
        s2 = data[configuration["multipositions"]].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
        del data[configuration["multipositions"]]
        pdf = pd.concat([s,s2], axis=1, keys=[proteinCol,configuration["multipositions"]])
    data = data.join(pdf)
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
        Annovar annotated VCF file from Mutect (sampleID_mutect_annovar.vcf)
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    regex = r"p\.(\w\d+\w)"
    aux = uri.split("/")[-1].split("_")
    sample = aux[0]
    #Get the columns from config 
    columns = configuration["columns"]
    #Read the data from file
    data = readDataset(uri)
    data = data[columns]
    data["sample"] = aux[0]
    data["variantCallingMethod"] = aux[1]
    data["annotated"] = aux[2].split('.')[0]
    data["alternative_names"] = data[configuration["alt_names"]].apply(lambda x: ','.join([match.group(1) for (matchNum,match) in enumerate(re.finditer(regex, x))]))
    data = data.drop(configuration["alt_names"], axis = 1)
    data = data.iloc[1:]
    data = data.replace('.', np.nan)
    data["ID"] = data[configuration["id_fields"]].apply(lambda x: x[0]+":g."+x[1]+x[2]+'>'+x[3], axis=1)
    
    return sample, data
    
########################################
#          Generate graph files        # 
########################################
def generateDatasetImports(projectId, dataType):
    dataDir = config.dataTypes[dataType]["directory"]
    configuration = config.dataTypes[dataType]
    if dataType == "clinical":
        data = parseClinicalDataset(projectId, configuration, dataDir)
        dataRows = extractSubjectClinicalVariablesRelationships(data)
        generateGraphFiles(dataRows,'clinical', projectId)
        dataRows = extractSubjectGroupRelationships(data)
        generateGraphFiles(dataRows,'groups', projectId)
    elif dataType == "proteomics":
        data = parseProteomicsDataset(projectId, configuration, dataDir)
        for dtype in data:
            if dtype == "proteins":
                dataRows = extractProteinSubjectRelationships(data[dtype], configuration[dtype])
                proSamRows = extractProjectSampleRelationships(dataRows, projectId)
                generateGraphFiles(proSamRows,'project', projectId)
                generateGraphFiles(dataRows,dtype, projectId)
            elif dtype == "peptides":
                dataRows = extractPeptideSubjectRelationships(data[dtype], configuration[dtype]) 
                generateGraphFiles(dataRows, "subject_peptide", projectId)
                dataRows = extractPeptideProteinRelationships(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows,"peptide_protein", projectId)
                dataRows = extractPeptides(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows, dtype, projectId)
            else:
                dataRows = extractModificationProteinRelationships(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows,"protein_modification", projectId, ot = 'a')
                dataRows = extractProteinModificationSubjectRelationships(data[dtype], configuration[dtype])                
                generateGraphFiles(dataRows, "modifiedprotein_subject", projectId, ot = 'a')
                dataRows = extractProteinProteinModificationRelationships(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows, "modifiedprotein_protein", projectId, ot = 'a')
                dataRows = extractProteinModifications(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows, "modifiedprotein", projectId, ot = 'a')
                dataRows = extractProteinModificationsModification(data[dtype], configuration[dtype])
                generateGraphFiles(dataRows, "modifiedprotein_modification", projectId, ot = 'a')
    elif dataType == "wes":
        data = parseWESDataset(projectId, configuration, dataDir)
        somatic_mutations = pd.DataFrame()
        for sample in data:
            entities, variantRows, sampleRows, geneRows, chrRows = extractWESRelationships(data[sample], configuration)
            generateGraphFiles(variantRows, "somatic_mutation_known_variant", projectId, d = dataType)
            generateGraphFiles(sampleRows, "somatic_mutation_sample", projectId, d = dataType)
            generateGraphFiles(geneRows, "somatic_mutation_gene", projectId, d = dataType)
            generateGraphFiles(chrRows, "somatic_mutation_chromosome", projectId, d = dataType)
            if somatic_mutations.empty:
                somatic_mutations = entities
            else:
                somatic_mutations.join(entities, on="ID")
        somatic_mutations = somatic_mutations.reset_index()
        generateGraphFiles(somatic_mutations, "somatic_mutation", projectId, dataType, d= dataType)
            
def generateGraphFiles(data, dataType, projectId, ot = 'w', d = 'proteomics'):
    importDir = os.path.join(config.datasetsImportDirectory, os.path.join(projectId,d))
    outputfile = os.path.join(importDir, projectId+"_"+dataType.lower()+".csv")
    with open(outputfile, ot) as f:  
        data.to_csv(path_or_buf = f, 
            header=True, index=False, quotechar='"', 
            line_terminator='\n', escapechar='\\')

if __name__ == "__main__":
    #uri = sys.argv[1]
    #generateDatasetImports('P0000001', 'proteomicsData')
    #generateDatasetImports('P0000001', 'clinicalData')
    generateDatasetImports('P0000002', 'wes') 
    
