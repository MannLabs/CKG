from KnowledgeGrapher.experiments import experiments_config as config
from KnowledgeGrapher.ontologies import ontologies_controller as oh
from KnowledgeGrapher import utils

import sys
import re
import os.path
import pandas as pd
import numpy as np
from collections import defaultdict


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
    data = pd.read_excel(uri, index_col=None, na_values=['NA'], convert_float = False)

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

    return (auxAttr_col,cCols), (auxAttr_reg,regexCols)

def mergeRegexAttributes(data, attributes, index):
    if not attributes.empty:
        attributes.columns = [c.split(" ")[-1] for c in attributes.columns]
        attributes = attributes.stack()
        attributes = attributes.reset_index()
        attributes.columns = ["c"+str(i) for i in range(len(attributes.columns))]
        data = pd.merge(data, attributes, on = index)

    return data

def mergeColAttributes(data, attributes, index):
    if not attributes.empty:
        data = data.set_index(index)
        data = data.join(attributes)
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
def parseClinicalDataset(projectId, configuration, dataDir):
    '''This function parses clinical data from subjects in the project
    Input: uri of the clinical data file. Format: Subjects as rows, clinical variables as columns
    Output: pandas DataFrame with the same input format but the clinical variables mapped to the
    right ontology (defined in config), i.e. type = -40 -> SNOMED CT'''
    
    dfile = configuration['file']
    filepath = os.path.join(dataDir, dfile)
    data = None
    if os.path.isfile(filepath):
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
        filepath = os.path.join(dataDir, dfile)
        if os.path.isfile(filepath):
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
    files = utils.listDirectoryFiles(dataDir)
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
    else:
        return None

def extractSubjectDiseaseRelationships(data):
    cols = list(data.columns)
    if "disease" in data.columns:
        data = data["disease"].to_frame()
        data = data.reset_index()
        data.columns = ['START_ID', 'END_ID']
        data['TYPE'] = "HAS_DISEASE"
        data = data.dropna()
        return data
    else:
        return None

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
    newIndexdf = aux.copy()
    aux = aux.drop(cols, axis=1)
    aux =  aux.filter(regex = configuration["valueCol"])
    aux.columns = [c.split(" ")[1] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["c"+str(i) for i in range(len(aux.columns))]
    columns = ['END_ID', 'START_ID',"value"]
    
    attributes = configuration["attributes"]
    (cAttributes,cCols), (rAttributes,regexCols) = extractAttributes(newIndexdf, attributes)
    if not rAttributes.empty:
        aux = mergeRegexAttributes(aux, rAttributes, ["c0","c1"])
        columns.extend(regexCols)
    if not cAttributes.empty:
        aux = mergeColAttributes(aux, cAttributes, "c0")
        columns.extend(cCols)
    

    aux['TYPE'] = "HAS_QUANTIFIED_PROTEINMODIFICATION"
    columns.append("TYPE")
    aux.columns = columns
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    
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
    aux = aux.groupby(aux.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    aux.columns = ["ID", "type", "count"]

    return aux

def extractPeptideSubjectRelationships(data, configuration):
    data = data[~data.index.duplicated(keep='first')]
    aux =  data.filter(regex = configuration["valueCol"])
    attributes = configuration["attributes"]
    aux.columns = [c.split(" ")[-1] for c in aux.columns]
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
    
    return aux

def extractPeptideProteinRelationships(data, configuration):
    cols = [configuration["proteinCol"]]
    cols.extend(configuration["positionCols"])
    aux =  data[cols]
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Protein", "Start", "End"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux.columns = ['START_ID', 'END_ID', "start", "end", 'TYPE']
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    return aux

################# Protein entity #########################
def extractProteinSubjectRelationships(data, configuration):
    aux =  data.filter(regex = configuration["valueCol"])
    attributes = configuration["attributes"]  
    aux.columns = [c.split(" ")[-1] for c in aux.columns]
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
    columns = set(columns).difference(filters)
    columns.remove(indexCol)

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
    data["ID"] = data[configuration["id_fields"]].apply(lambda x: str(x[0])+":g."+str(x[1])+str(x[2])+'>'+str(x[3]), axis=1)
    return sample, data
    
########################################
#          Generate graph files        # 
########################################
def generateDatasetImports(projectId, dataType):
    print(dataType)
    if dataType in config.dataTypes:
        if "directory" in config.dataTypes[dataType]:
            dataDir = config.dataTypes[dataType]["directory"].replace("PROJECTID", projectId)
            configuration = config.dataTypes[dataType]
            if dataType == "clinical":
                data = parseClinicalDataset(projectId, configuration, dataDir)
                if data is not None:
                    dataRows = extractSubjectClinicalVariablesRelationships(data)
                    generateGraphFiles(dataRows,'clinical', projectId, d = dataType)
                    dataRows = extractSubjectGroupRelationships(data)
                    if dataRows is not None:
                        generateGraphFiles(dataRows,'groups', projectId, d = dataType)
                    dataRows = extractSubjectDiseaseRelationships(data)
                    if dataRows is not None:
                        generateGraphFiles(dataRows,'disease', projectId, d = dataType)
            elif dataType == "proteomics":
                data = parseProteomicsDataset(projectId, configuration, dataDir)
                if data is not None:
                    for dtype in data:
                        if dtype == "proteins":
                            dataRows = extractProteinSubjectRelationships(data[dtype], configuration[dtype])
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
                if data is not None:
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
                            new = set(entities.index).difference(set(somatic_mutations.index))
                            somatic_mutations = somatic_mutations.append(entities.loc[new,:], ignore_index=False)
                    somatic_mutations = somatic_mutations.reset_index()
                    generateGraphFiles(somatic_mutations, "somatic_mutation", projectId, d= dataType, ot = 'w')
            
def generateGraphFiles(data, dataType, projectId, ot = 'w', d = 'proteomics'):
    importDir = os.path.join(config.experimentsImportDirectory, os.path.join(projectId,d))
    outputfile = os.path.join(importDir, projectId+"_"+dataType.lower()+".csv")
    with open(outputfile, ot) as f:  
        data.to_csv(path_or_buf = f, 
            header=True, index=False, quotechar='"', 
            line_terminator='\n', escapechar='\\')

if __name__ == "__main__":
    pass
