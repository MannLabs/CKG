import os.path
import re
from collections import defaultdict
from ckg.graphdb_builder import mapping as mp, builder_utils

#########################
#   Reactome database   #
#########################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="reactomeConfig.yml", data_type='databases')
    urls = config['reactome_urls']
    entities = set()
    relationships = defaultdict(set)
    entities_header = config['pathway_header']
    relationships_headers = config['relationships_header']
    directory = os.path.join(databases_directory, "Reactome")
    builder_utils.checkDirectory(directory)
    metabolite_mapping = mp.getMappingForEntity("Metabolite")
    #drug_mapping = mp.getMappingForEntity("Drug")

    for dataset in urls:
        url = urls[dataset]
        file_name = url.split('/')[-1]
        if download:
            builder_utils.downloadDB(url, directory)
        f = os.path.join(directory, file_name)
        with open(f, 'r') as rf:
            if dataset == "pathway":
                entities = parsePathways(config, databases_directory, rf)
            elif dataset == "hierarchy":
                relationships[("pathway", "has_parent")] = parsePathwayHierarchy(rf)
            elif dataset == "protein":
                relationships[(dataset, "annotated_to_pathway")] = parsePathwayRelationships(config, rf)
            elif dataset == "metabolite":
                relationships[(dataset, "annotated_to_pathway")] = parsePathwayRelationships(config, rf, metabolite_mapping)
            #elif dataset == "drug":
                #relationships[(dataset, "annotated_to_pathway")] = set()

    builder_utils.remove_directory(directory)

    return entities, relationships, entities_header, relationships_headers


def parsePathways(config, databases_directory, fhandler):
    entities = set()
    organisms = config['organisms']
    url = config['linkout_url']
    directory = os.path.join(databases_directory, "Reactome")
    mapping_file = os.path.join(directory, "mapping.tsv")

    mp.reset_mapping(entity="Pathway")
    with open(mapping_file, 'w') as mf:
        for line in fhandler:
            data = line.rstrip("\r\n").split("\t")
            identifier = data[0]
            name = data[1]
            organism = data[2]
            linkout = url.replace("PATHWAY", identifier)
            if organism in organisms:
                organism = organisms[organism]
                entities.add((identifier, "Pathway", name, name, organism, linkout, "Reactome"))
                mf.write(identifier+"\t"+name+"\n")

    mp.mark_complete_mapping(entity="Pathway")

    return entities


def parsePathwayHierarchy(fhandler):
    relationships = set()
    for line in fhandler:
        data = line.rstrip("\r\n").split("\t")
        parent = data[0]
        child = data[1]
        relationships.add((child, parent, "HAS_PARENT", "Reactome"))

    return relationships


def parsePathwayRelationships(config, fhandler, mapping=None):
    relationships = set()
    regex = r"(.+)\s\[(.+)\]"
    organisms = config['organisms']
    for line in fhandler:
        data = line.rstrip("\r\n").split("\t")
        identifier = data[0]
        id_loc = data[2]
        pathway = data[3]
        evidence = data[6]
        organism = data[7]
        match = re.search(regex, id_loc)
        loc = "unspecified"
        if match:
            name = match.group(1)
            loc = match.group(2)
            if organism in organisms:
                organism = organisms[organism]
                if mapping is not None:
                    if identifier in mapping:
                        identifier = mapping[identifier]
                    elif name in mapping:
                        identifier = mapping[name]
                    else:
                        continue
                relationships.add((identifier, pathway, "ANNOTATED_TO_PATHWAY", evidence, organism, loc, "Reactome"))

    return relationships
