import os.path
import gzip
from ckg.graphdb_builder import builder_utils

#########################
#   PathwayCommons      #
#########################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="pathwayCommonsConfig.yml", data_type='databases')
    url = config['pathwayCommons_pathways_url']
    entities = set()
    relationships = set()
    directory = os.path.join(databases_directory, "PathwayCommons")
    builder_utils.checkDirectory(directory)
    fileName = url.split('/')[-1]
    entities_header = config['pathways_header']
    relationships_header = config['relationships_header']

    if download:
        builder_utils.downloadDB(url, directory)
    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        linkout = data[0]
        code = data[0].split("/")[-1]
        ptw_dict = dict([item.split(": ")[0], ":".join(item.split(": ")[1:])] for item in data[1].split("; "))
        proteins = data[2:]
        if "organism" in ptw_dict and ptw_dict["organism"] == "9606":
            name = ptw_dict["name"]
            source = ptw_dict["datasource"]
        else:
            continue

        entities.add((code, "Pathway", name, name, ptw_dict["organism"], source, linkout))
        for protein in proteins:
            relationships.add((protein, code, "ANNOTATED_IN_PATHWAY", linkout, "PathwayCommons: "+source))

    associations.close()

    builder_utils.remove_directory(directory)

    return (entities, relationships, entities_header, relationships_header)
