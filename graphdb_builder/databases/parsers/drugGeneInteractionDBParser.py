import os.path
import ckg_utils
from graphdb_builder import mapping as mp, builder_utils

############################################
#   The Drug Gene Interaction Database     # 
############################################
def parser(databases_directory, download = True):
    
    config = ckg_utils.get_configuration('../databases/config/drugGeneInteractionDBConfig.yml')

    url = config['DGIdb_url']
    header = config['header']
    output_file = "dgidb_targets.tsv"
    drugmapping = mp.getMappingForEntity("Drug")

    relationships = set()
    directory = os.path.join(databases_directory,"DGIdb")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
    with open(fileName, 'r') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            gene = data[0]
            source = data[3]
            interactionType = data[4] if data[4] != '' else 'unknown'
            drug = data[8].lower()
            if drug == "":
                drug = data[7] 
                if drug == "" and data[6] != "":
                    drug = data[6]
                else:
                    continue
            if gene != "":
                if drug in drugmapping:
                    drug = drugmapping[drug]
                    relationships.add((drug, gene, "TARGETS", "NA", "NA", "NA", interactionType, "DGIdb: "+source))

    return (relationships, header, output_file)
