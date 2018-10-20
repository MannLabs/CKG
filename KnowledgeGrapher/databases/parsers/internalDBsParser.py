import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import internalDBsConfig as iconfig
from KnowledgeGrapher import utils
from KnowledgeGrapher import mapping as mp
import pandas as pd

#############################################
#   Internal Databases (JensenLab.org)      # 
#############################################

def parser(download=True):
    result = {}
    string_url = iconfig.string_url
    string_mapping = mp.getSTRINGMapping(string_url, download=False)
    for qtype in iconfig.internal_db_types:
        relationships = parseInternalDatabasePairs(qtype, string_mapping)
        entity1, entity2 = iconfig.internal_db_types[qtype]
        outputfileName =  entity1+"_"+entity2+"_associated_with_integrated.csv"
        header = iconfig.header
        result[qtype] = (relationships, header, outputfileName)
    
    return result

def parserMentions(importDirectory,download=True):
    entities, header = parsePMClist()
    outputfileName = "Publications.csv"
    for qtype in iconfig.internal_db_mentions_types:
        parseInternalDatabaseMentions(qtype, importDirectory)

    return (entities, header, outputfileName)

def parseInternalDatabasePairs(qtype, mapping, download=True):
    url = iconfig.internal_db_url
    ifile = iconfig.internal_db_files[qtype]
    source = iconfig.internal_db_sources[qtype]
    relationships = set()
    directory = os.path.join(dbconfig.databasesDir, "InternalDatabases")
    utils.checkDirectory(directory)
    if download:
        utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory,"integration"))
    ifile = os.path.join(directory,os.path.join("integration",ifile))
    with open(ifile, 'r') as idbf:
        for line in idbf:
            data = line.rstrip("\r\n").split('\t')
            id1 = "9606."+data[0]
            id2 = data[2]
            score = float(data[4])

            if id1 in mapping:
                for ident in mapping[id1]:
                    relationships.add((ident, id2, "ASSOCIATED_WITH_INTEGRATED", source, score, "compiled"))
            else:
                continue
                
    return relationships

def parsePMClist(download = True):
    url = iconfig.PMC_db_url
    plinkout = iconfig.pubmed_linkout
    entities = set()
    directory = os.path.join(dbconfig.databasesDir, "InternalDatabases")
    utils.checkDirectory(directory)
    directory = os.path.join(directory,"textmining")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    
    if download:
        utils.downloadDB(url, directory)

    entities = pd.read_csv(fileName, sep = ',', dtype = str, compression = 'gzip', low_memory=False)
    entities = entities[iconfig.PMC_fields]
    entities = entities[entities.iloc[:,0].notnull()]
    entities = entities.set_index(list(entities.columns)[0])
    entities['linkout'] = [plinkout.replace("PUBMEDID", str(int(pubmedid))) for pubmedid in list(entities.index)]
    entities.index.names = ['ID']
    entities['TYPE'] = 'Publication'
    entities = entities.reset_index()
    header = [c.replace(' ','_').lower() if c not in ['ID', 'TYPE'] else c for c in list(entities.columns)]
    entities = list(entities.itertuples(index=False)) 
    entities = entities.replace('\\','')
    return entities, header

def parseInternalDatabaseMentions(qtype, importDirectory, download=True):
    url = iconfig.internal_db_url
    string_url = iconfig.string_url
    stitch_url = iconfig.stitch_url
    ifile = iconfig.internal_db_mentions_files[qtype]
    if qtype == "9606":
        mapping = mp.getSTRINGMapping(string_url, download=False)
    elif qtype == "-1":
        mapping = mp.getSTRINGMapping(stitch_url, source = iconfig.internal_db_sources["Drug"], download = False, db = "STITCH")
    filters = []
    if qtype in iconfig.internal_db_mentions_filters:
        filters = iconfig.internal_db_mentions_filters[qtype]
    entity1, entity2 = iconfig.internal_db_mentions_types[qtype]
    outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_mentioned_in_publication.csv")
    relationships = pd.DataFrame()
    directory = os.path.join(dbconfig.databasesDir, "InternalDatabases")
    if download:
        utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory,"textmining"))
    ifile = os.path.join(directory,os.path.join("textmining",ifile))
    with open(outputfile,'w') as f:
        f.write("START_ID,END_ID,TYPE\n")
        with open(ifile, 'r') as idbf:
            for line in idbf:
                data = line.rstrip("\r\n").split('\t')
                id1 = data[0]
                pubmedids = data[1].split(" ")
                
                if qtype == "9606":
                    id1 = "9606."+id1
                    if id1 in mapping:
                        ident = mapping[id1]
                    else:
                        continue
                elif qtype == "-1":
                    if id1 in mapping:
                        ident = mapping[id1]
                else:
                    ident = [id1]
                for i in ident:
                    if i not in filters:
                        aux = pd.DataFrame(data = {"Pubmedids":pubmedids})
                        aux["START_ID"] = i
                        aux["TYPE"] = "MENTIONED_IN_PUBLICATION"
                        aux.to_csv(path_or_buf=f, header=False, index=False, quotechar='"', line_terminator='\n', escapechar='\\')
