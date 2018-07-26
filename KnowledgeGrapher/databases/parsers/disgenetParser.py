import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import disgenetConfig as iconfig
from collections import defaultdict
from KnowledgeGrapher import utils

#########################
#       DisGeNet        # 
#########################
def parser(download = True):
    relationships = defaultdict(set)
    files = iconfig.disgenet_files
    url = iconfig.disgenet_url
    directory = os.path.join(dbconfig.databasesDir,"disgenet")
    header = iconfig.disgenet_header
    outputfileName = iconfig.outputfileName

    if download:
        for f in files:
            utils.downloadDB(url+files[f], "disgenet")

    proteinMapping = readDisGeNetProteinMapping() 
    diseaseMapping, diseaseSynonyms = readDisGeNetDiseaseMapping()
    for f in files:
        first = True
        associations = gzip.open(os.path.join(directory,files[f]), 'r')
        dtype, atype = f.split('_') 
        if dtype == 'gene':
            idType = "Protein"
            scorePos = 7
        if dtype == 'variant':
            idType = "Transcript"
            scorePos = 5
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            geneId = data[0]
            diseaseId = data[2]
            score = float(data[4])
            pmids = data[5]
            source = data[scorePos]
            if geneId in proteinMapping:
                for identifier in proteinMapping[geneId]:
                    if diseaseId in diseaseMapping:
                        for code in diseaseMapping[diseaseId]:
                            code = "DOID:"+code
                            relationships[idType].add((identifier, code,"ASSOCIATED_WITH", score, atype, "DisGeNet: "+source, pmids))
        associations.close()
    return (relationships,header,outputfileName)
    
def readDisGeNetProteinMapping():
    files = iconfig.disgenet_mapping_files
    directory = os.path.join(dbconfig.databasesDir,"disgenet")
    
    first = True
    mapping = defaultdict(set)
    if "protein_mapping" in files:
        mappingFile = files["protein_mapping"]
        f = gzip.open(os.path.join(directory,mappingFile), 'r')
        for line in f:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            identifier = data[0]
            intIdentifier = data[1]
            mapping[intIdentifier].add(identifier)
        f.close()
    return mapping

def readDisGeNetDiseaseMapping():
    files = iconfig.disgenet_mapping_files
    directory =  os.path.join(dbconfig.databasesDir,"disgenet")
    first = True
    mapping = defaultdict(set)
    synonyms = defaultdict(set)
    if "disease_mapping" in files:
        mappingFile = files["disease_mapping"]
        f = gzip.open(os.path.join(directory,mappingFile), 'r')
        for line in f:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            identifier = data[0]
            vocabulary = data[2]
            code = data[3]
            if vocabulary == onto_config.ontologies["Disease"]:
                mapping[identifier].add(code)
            else:
                synonyms[identifier].add(code)
        f.close()
    return mapping, synonyms
