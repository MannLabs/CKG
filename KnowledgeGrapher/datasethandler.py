import pandas as pd
from collections import defaultdict
import numpy as np
import config
import sys
import ontologyhandler as oh

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
    data = pd.read_excel(open(uri), index_col=None, na_values=['NA'])

    return data

def extractSubjectReplicates(data, regex):
    subjectDict = defaultdict(list)
    for r in regex:
        columns = data.filter(regex = r).columns
        for c in columns:
            fields  = c.split('_')
            value = " ".join(fields[0].split(' ')[0:-1])
            subject = fields[1]
            ident = value + " " + subject 
            subjectDict[ident].append(c)

    return subjectDict

def calculateMedianReplicates(data, log = "log2"):
    if log == "log2":
        data = data.applymap(lambda x:np.log2(x) if x > 0 else np.nan)
    elif log == "log10":
        data = data.applymap(lambda x:np.log10(x) if x > 0 else np.nan)
    
    median = data.median(axis=1).sort_values(axis=0, ascending= True, na_position = 'first').to_frame()

    return median

############################
#           Parsers        # 
############################
########### Clinical Variables Datasets ############
def parseClinicalDataset(uri):
    '''This function parses clinical data from subjects in the project
    Input: uri of the clinical data file. Format: Subjects as rows, clinical variables as columns
    Output: pandas DataFrame with the same input format but the clinical variables mapped to the
    right ontology (defined in config), i.e. type = -40 -> SNOMED CT'''
    data = readDataset(uri)
    data = data.set_index("subject id")
    projectId = uri.split('/')[-2]
        
    return data, projectId

########### Proteomics Datasets ############
def parseProteomicsDataset(uri, qtype):
    configuration = config[qtype]
    data, regex = loadProteomicsDataset(uri, configuration)
    log = configuration[log]

    subjectDict = extractSubjectReplicates(data, regex)
    delCols = []
    for subject in subjectDict:
            delCols.extend(subjectDict[subject])
            aux = data[subjectDict[subject]]
            data[subject] = calculateMedianReplicates(aux, log)

    data = data.drop(delCols, 1)
    

    return data, projectId

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
    data.columns = [':START_ID(Subject)', ':END_ID(Clinical_variable)', "score"]
    data[':TYPE'] = "HAS_QUANTIFIED_CLINICAL"
    data = data[[':START_ID(Sample)', ':END_ID(Clinical_variable)',':TYPE', 'score']]

    return data

########### Proteomics Datasets ############
def extractProteinSubjectRelationships(data):
    aux =  data.filter(regex = 'LFQ intensity')
    aux.columns = [c.split(" ")[2] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux[':TYPE'] = "HAS_QUANTIFIED_PROTEIN"
    aux.columns = [':END_ID(Protein)', ':START_ID(Sample)',"LFQ intensity", ':TYPE']
    aux = aux[[':START_ID(Sample)', ':END_ID(Protein)', ':TYPE', "LFQ intensity"]]
    
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
    aux[':TYPE'] = "HAD_QUANTIFIED_"+modification.upper()
    aux['code'] = code
    aux.columns = [':END_ID('+modification+')', ':START_ID(Sample)', "Intensity", ':TYPE', 'code']
    aux = aux[[':START_ID(Sample)', ':END_ID('+modification+')', ':TYPE', "Intensity", 'code']]
    
    return aux

#Not sure this is necessary -- Would correspond to the relationship "IS_MODIFIED_IN" 
#(MODIFIED needs to be substituted by the modification)
def extractPTMProteinRelationships(data, modification):
    code = config["modifications"][modification]["code"]
    aux = data.copy()
    aux["modId"] = aux["Protein"].str + "_" + aux["Amino acid"].str + aux["Positions"].str
    aux = aux.reset_index()
    aux = aux[["Protein", "modId"]]
    aux[':TYPE'] = "IS_MODIFIED_AT"
    aux['code'] = code
    aux = aux[[':START_ID(Protein)', ':END_ID('+modification+')',':TYPE', 'code']]
    
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
    if "geneCol" in configuration:
        geneCol = configuration["geneCol"]
    
    #Read the data from file
    data = readDataset(uri)

    #Apply filters
    data = data[data[filters].isnull().all(1)]
    data = data.drop(filters, axis=1)

    #Select first protein form group, i.e. P01911;Q29830;Q9MXZ4 -> P01911
    #Set protein as index
    proteins = data[proteinCol].str.split(';').apply(pd.Series,1)[0]
    data[proteinCol] = proteins
    data = data.set_index(proteinCol)
    filters.append(proteinCol)
    columns = set(columns).difference(filters)

    #Get columns using regex
    for regex in regexCols:
        if aux is None:
            aux = data.filter(regex=regex)
        else:
            aux = aux.join(data.filter(regex=regex))
     
    #Add simple and regex columns into a single DataFrame
    data = data[list(columns)]
    data = data.join(aux)
    
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
def generateGraphFiles(importDirectory, data):
    outputfile = config.intact_file
    interactions = parseIntactDatabase(dataFile, proteins)
    interactions_outputfile = os.path.join(importDirectory, "intact_interacts_with.csv")
    
    interactionsDf = pd.DataFrame(interactions) 
    interactionsDf.columns = [':START_ID(Protein)', ':END_ID(Protein)',':TYPE', 'score', 'interaction_type', 'method', 'source', 'publications']
    
    interactionsDf.to_csv(path_or_buf=interactions_outputfile, 
            header=True, index=False, quotechar='"', 
            line_terminator='\n', escapechar='\\')



if __name__ == "__main__":
    uri = sys.argv[1]

    parseProteomics
