#############################################
#   Internal Databases (JensenLab.org)      # 
#############################################
def parseInternalDatabasePairs(qtype, mapping, download = True):
    url = config.internal_db_url
    ifile = config.internal_db_files[qtype]
    source = config.internal_db_sources[qtype]
    relationships = set()
    directory = os.path.join(config.databasesDir, "InternalDatabases")
    if download:
        downloadDB(url.replace("FILE", ifile), os.path.join(directory,"integration"))
    ifile = os.path.join(directory,os.path.join("integration",ifile))
    with open(ifile, 'r') as idbf:
        for line in idbf:
            data = line.rstrip("\r\n").split('\t')
            id1 = "9606."+data[0]
            id2 = data[2]
            score = float(data[4])

            if id1 in mapping:
                for ident in mapping[id1]:
                    relationships.add((ident, id2, "ASSOCIATED_WITH_INTEGRATED", source, score))
            else:
                continue
                
    return relationships

def parsePMClist(download = True):
    url = config.PMC_db_url
    plinkout = config.pubmed_linkout
    entities = set()
    directory = os.path.join(config.databasesDir, "InternalDatabases")
    utils.checkDirectory(directory)
    directory = os.path.join(directory,"textmining")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    
    if download:
        downloadDB(url, directory)

    entities = pd.read_csv(fileName, sep = ',', dtype = str, compression = 'gzip', low_memory=False)
    entities = entities[config.PMC_fields]
    entities = entities[entities.iloc[:,0].notnull()]
    entities = entities.set_index(list(entities.columns)[0])
    entities['linkout'] = [plinkout.replace("PUBMEDID", str(int(pubmedid))) for pubmedid in list(entities.index)]
    entities.index = entities.index.rename('ID')
    entitites = entities.reset_index()
    header = list(entities.columns)
    entities = list(entities.itertuples(index=False)) 
    
    return entities, header

def parseInternalDatabaseMentions(qtype, mapping, importDirectory, download = True):
    url = config.internal_db_url
    ifile = config.internal_db_mentions_files[qtype]
    filters = []
    if qtype in config.internal_db_mentions_filters:
        filters = config.internal_db_mentions_filters[qtype]
    entity1, entity2 = config.internal_db_mentions_types[qtype]
    outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_mentioned_in_publication.csv")
    relationships = pd.DataFrame()
    directory = os.path.join(config.databasesDir, "InternalDatabases")
    if download:
        downloadDB(url.replace("FILE", ifile), os.path.join(directory,"textmining"))
    ifile = os.path.join(directory,os.path.join("textmining",ifile))
    with open(outputfile,'a') as f:
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
                else:
                    ident = [id1]
                for i in ident:
                    if i not in filters:
                        aux = pd.DataFrame(data = {"Pubmedids":pubmedids})
                        aux["START_ID"] = i
                        aux["TYPE"] = "MENTIONED_IN_PUBLICATION"
                        aux.to_csv(path_or_buf=f, header=False, index=False, quotechar='"', line_terminator='\n', escapechar='\\')
