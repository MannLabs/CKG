import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import siderConfig as iconfig
from KnowledgeGrapher import utils
from KnowledgeGrapher import mapping as mp
import re

#############################################
#              SIDER database               # 
#############################################
def parser(download=True):
    url = iconfig.SIDER_url
    header = iconfig.header
    outputfileName = iconfig.outputfileName
    
    drugsource = dbconfig.sources["Drug"]
    drugmapping = mp.getSTRINGMapping(iconfig.SIDER_mapping, source = drugsource, download = False, db = "STITCH")
    phenotypemapping = mp.getMappingFromOntology(ontology="Phenotype", source = iconfig.SIDER_source)
    
    relationships = set()
    directory = os.path.join(dbconfig.databasesDir,"SIDER")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, directory)
    associations = gzip.open(fileName, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        drug = re.sub(r'CID\d', 'CIDm', data[1])
        se = data[3]
        if se.lower() in phenotypemapping and drug in drugmapping:
            for d in drugmapping[drug]:
                p = phenotypemapping[se.lower()]
                relationships.add((d, p, "HAS_SIDE_EFFECT", "SIDER", se))
    associations.close()

    return (relationships, header, outputfileName, drugmapping, phenotypemapping)


def parserIndications(drugMapping, phenotypeMapping, download=True):
    url = iconfig.SIDER_indications
    header = iconfig.indications_header
    outputfileName = iconfig.indications_outputfileName

    relationships = set()
    directory = os.path.join(dbconfig.databasesDir,"SIDER")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, directory)
    associations = gzip.open(fileName, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        drug = re.sub(r'CID\d', 'CIDm', data[0])
        se = data[1]
        evidence = data[2]
        if se.lower() in phenotypeMapping and drug in drugMapping:
            for d in drugMapping[drug]:
                p = phenotypeMapping[se.lower()]
                relationships.add((d, p, "IS_INDICATED_FOR", evidence, "SIDER", se))
    
    associations.close()
    
    return (relationships, header, outputfileName)
