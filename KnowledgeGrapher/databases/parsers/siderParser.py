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
def parser(download = True):
    url = iconfig.SIDER_url
    header = iconfig.header
    outputfileName = iconfig.outputfileName
    
    drugsource = dbconfig.sources["Drug"]
    directory = os.path.join(dbconfig.databasesDir, drugsource)
    mappingFile = os.path.join(directory, "mapping.tsv")
    drugmapping = mp.getMappingFromDatabase(mappingFile)
    phenotypemapping = mp.getMappingFromOntology(ontology = "Phenotype", source = config.SIDER_source)
    
    relationships = set()
    directory = os.path.join(dbconfig.databasesDir,"SIDER")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "SIDER")
    associations = gzip.open(fileName, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        drug = re.sub(r'CID\d0+', '', data[1])
        se = data[3]
        if se in phenotypemapping and drug in drugmapping:
            do = phenotypemapping[se]
            drug = drugmapping[drug]            
            relationships.add((drug, do, "HAS_SIDE_EFFECT", "SIDER", se))
    associations.close()

    return (relationships, header, outputfileName)
