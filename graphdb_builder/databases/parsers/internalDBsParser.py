import os.path
import pandas as pd
import ckg_utils
from graphdb_builder import mapping as mp, builder_utils

#############################################
#   Internal Databases (JensenLab.org)      # 
#############################################

def parser(databases_directory, download=True):
    result = {}
    config = ckg_utils.get_configuration('../databases/config/internalDBsConfig.yml')
    string_url = config['string_url']
    string_mapping = mp.getSTRINGMapping(string_url, download=download)
    
    for qtype in config['internal_db_types']:
        relationships = parseInternalDatabasePairs(config, databases_directory, qtype, string_mapping)
        entity1, entity2 = config['internal_db_types'][qtype]
        outputfileName =  entity1+"_"+entity2+"_associated_with_integrated.csv"
        header = config['header']
        result[qtype] = (relationships, header, outputfileName)
    
    return result

def read_valid_pubs(organisms, organisms_file):
    pubs = set()
    with open(organisms_file, 'r') as idbf:
        for line in idbf:
            data = line.rstrip('\r\n').split('\t')
            if str(data[0]) in organisms:
                pubs.update(set(data[1]).split(" "))
    return pubs
    

def parserMentions(databases_directory, importDirectory, download=True):
    config = ckg_utils.get_configuration('../databases/config/internalDBsConfig.yml')
    entities, header = parsePMClist(config, databases_directory, download)
    outputfileName = "Publications.csv"
    url = config['internal_db_url']
    ifile = config['organisms_file']
    organisms = config['organisms']
    
    directory = os.path.join(databases_directory, "InternalDatabases")
    builder_utils.checkDirectory(directory)
    
    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory,"textmining"))
    
    ifile = os.path.join(directory,os.path.join("textmining",ifile))
    #valid_pubs = read_valid_pubs(organisms, ifile)
    #print(valid_pubs)
    for qtype in config['internal_db_mentions_types']:
        parseInternalDatabaseMentions(config, databases_directory, qtype, importDirectory, download)

    return (entities, header, outputfileName)

def parseInternalDatabasePairs(config, databases_directory, qtype, mapping, download=True):
    url = config['internal_db_url']
    ifile = config['internal_db_files'][qtype]
    source = config['internal_db_sources'][qtype]
    relationships = set()
    directory = os.path.join(databases_directory, "InternalDatabases")
    builder_utils.checkDirectory(directory)
    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory,"integration"))
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

def parsePMClist(config, databases_directory, download=True):
    url = config['PMC_db_url']
    plinkout = config['pubmed_linkout']
    entities = set()
    directory = os.path.join(databases_directory, "InternalDatabases")
    builder_utils.checkDirectory(directory)
    directory = os.path.join(directory,"textmining")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    
    if download:
        builder_utils.downloadDB(url, directory)
    entities = pd.read_csv(fileName, sep = ',', dtype = str, compression = 'gzip', low_memory=False)
    entities = entities[config['PMC_fields']]
    entities = entities[entities.iloc[:,0].notnull()]
    entities = entities.set_index(list(entities.columns)[0])
    entities['linkout'] = [plinkout.replace("PUBMEDID", str(int(pubmedid))) for pubmedid in list(entities.index)]
    entities.index.names = ['ID']
    entities['TYPE'] = 'Publication'
    entities = entities.reset_index()
    header = [c.replace(' ','_').lower() if c not in ['ID', 'TYPE'] else c for c in list(entities.columns)]
    entities = entities.replace('\\\\','', regex=True)
    entities = list(entities.itertuples(index=False)) 
    
    return entities, header

def parseInternalDatabaseMentions(config, databases_directory, qtype, importDirectory, download=True):
    url = config['internal_db_url']
    string_url = config['string_url']
    stitch_url = config['stitch_url']
    ifile = config['internal_db_mentions_files'][qtype]
    if qtype == "9606":
        mapping = mp.getSTRINGMapping(string_url, download=False)
    elif qtype == "-1":
        mapping = mp.getSTRINGMapping(stitch_url, source = config['internal_db_sources']["Drug"], download = False, db = "STITCH")
    filters = []
    if qtype in config['internal_db_mentions_filters']:
        filters = config['internal_db_mentions_filters'][qtype]
    entity1, entity2 = config['internal_db_mentions_types'][qtype]
    outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_mentioned_in_publication.csv")
    relationships = pd.DataFrame()
    directory = os.path.join(databases_directory, "InternalDatabases")
    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory,"textmining"))
    ifile = os.path.join(directory,os.path.join("textmining",ifile))
    with open(outputfile,'w') as f:
        f.write("START_ID,END_ID,TYPE\n")
        with open(ifile, 'r') as idbf:
            for line in idbf:
                data = line.rstrip("\r\n").split('\t')
                id1 = data[0]
                pubmedids = data[1].split(" ")
                #pubmedids = list(set(pubmedids).intersection(valid_pubs))
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

