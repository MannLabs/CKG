import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import siderConfig as iconfig
from KnowledgeGrapher import utils
import re

#############################################
#              SIDER database               # 
#############################################
def parser(download = True):
    url = iconfig.SIDER_url
    
    drugsource = dbconfig.sources["Drug"]
    directory = os.path.join(dbconfig.databasesDir, drugsource)
    mappingFile = os.path.join(directory, "mapping.tsv")
    drugmapping = utils.getMappingFromDatabase(mappingFile)
    diseasemapping = utils.getMappingFromOntology(ontology = "Disease", source = config.SIDER_source)
    
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
        if se in diseasemapping and drug in drugmapping:
            do = diseasemapping[se]
            drug = drugmapping[drug]            
            relationships.add((drug, do, "HAS_SIDE_EFFECT", "SIDER", se))
    associations.close()

    return relationships
