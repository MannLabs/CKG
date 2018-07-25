############################################
#   The Drug Gene Interaction Database     # 
############################################
def parseDGIdb(download = True):
    url = config.DGIdb_url

    drugsource = config.sources["Drug"]
    directory = os.path.join(config.databasesDir, drugsource)
    mappingFile = os.path.join(directory, "mapping.tsv")
    drugmapping = utils.getMappingFromDatabase(mappingFile)

    relationships = set()
    directory = os.path.join(config.databasesDir,"DGIdb")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        downloadDB(url, "DGIdb")
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

    return relationships
