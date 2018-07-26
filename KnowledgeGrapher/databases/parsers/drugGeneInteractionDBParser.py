import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import drugGeneInteractionDBConfig as iconfig
from KnowledgeGrapher import utils

############################################
#   The Drug Gene Interaction Database     # 
############################################
def parser(download = True):
    url = iconfig.DGIdb_url
    header = iconfig.header
    outputfileName = iconfig.outputfileName

    drugsource = dbconfig.sources["Drug"]
    directory = os.path.join(dbconfig.databasesDir, drugsource)
    mappingFile = os.path.join(directory, "mapping.tsv")
    drugmapping = utils.getMappingFromDatabase(mappingFile)

    relationships = set()
    directory = os.path.join(dbconfig.databasesDir,"DGIdb")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "DGIdb")
    with open(fileName, 'r') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            gene = data[0]
            source = data[3]
            interactionType = data[4]
            drug = data[8].lower()
            if drug == "":
                drug = data[7] 
                if drug == "" and data[6] != "":
                    drug = data[6]
                else:
                    continue
            if drug in drugmapping:
                drug = drugmapping[drug]
            relationships.add((drug, gene, "TARGETS", interactionType, "DGIdb: "+source))

    return (relationships, header, outputfileName)
