import os.path
from ckg.graphdb_builder import mapping as mp, builder_utils

######################################
#   Databases at jensenLab.org)      #
######################################


def parser(databases_directory, download=True):
    result = {}
    config = builder_utils.get_config(config_name="jensenlabConfig.yml", data_type='databases')
    string_mapping = mp.getSTRINGMapping(download=download)

    for qtype in config['db_types']:
        relationships = parsePairs(config, databases_directory, qtype, string_mapping)
        entity1, entity2 = config['db_types'][qtype]
        outputfileName = entity1+"_"+entity2+"_associated_with_integrated.tsv"
        header = config['header']
        result[qtype] = (relationships, header, outputfileName)

    return result


def parsePairs(config, databases_directory, qtype, mapping, download=True):
    url = config['db_url']
    ifile = config['db_files'][qtype]
    source = config['db_sources'][qtype]
    relationships = set()

    directory = os.path.join(databases_directory, "Jensenlab")
    builder_utils.checkDirectory(os.path.join(directory, "integration"))

    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory, "integration"))
    ifile = os.path.join(directory,os.path.join("integration", ifile))

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
