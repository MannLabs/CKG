import os.path
import zipfile
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import corumConfig as iconfig
from KnowledgeGrapher import mapping as mp
from collections import defaultdict
from KnowledgeGrapher import utils
import pandas as pd
import re

###################
#       CORUM     # 
###################
def parser(download=True):
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(dbconfig.databasesDir,"CORUM")
    utils.checkDirectory(directory)
    database_url = iconfig.database_url    
    entities_header = iconfig.entities_header
    relationships_headers = iconfig.relationships_headers
    zipped_fileName = os.path.join(directory, database_url.split('/')[-1])
    fileName = '.'.join(database_url.split('/')[-1].split('.')[0:2])
    if download:
        utils.downloadDB(database_url, directory)
    first = True
    with zipfile.ZipFile(zipped_fileName) as z:
        with z.open(fileName) as f:
            for line in f:
                if first:
                    first = False
                    continue
                data = line.decode("utf-8").rstrip("\r\n").split("\t")
                identifier = data[0]
                name = data[1]
                organism = data[2]
                synonyms = data[3] if data[3] != "None" else ""
                cell_lines = data[4]
                subunits = data[5].split(';')
                evidences = data[7]
                processes = data[8].split(';')
                pubmedid = data[14]
                
                if organism == "Human":
                    #ID name organism synonyms source
                    entities.add((identifier, name, "9606", synonyms, "CORUM"))
                    for subunit in subunits:
                        #START_ID END_ID type cell_lines evidences publication source
                        relationships[("Protein", "IS_SUBUNIT_OF")].add((subunit, identifier, "IS_SUBUNIT_OF", cell_lines, evidences, pubmedid, "CORUM"))
                    for process in processes:
                        #START_ID END_ID type evidence_type score source
                        relationships["Biological_process", "ASSOCIATED_WITH"].add((identifier, process, "ASSOCIATED_WITH", "CURATED", 5, "CORUM"))

    return entities, relationships, entities_header, relationships_headers
    
if __name__ == "__main__":
    pass
