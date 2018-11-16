import os.path
from collections import defaultdict
from graphdb_builder.databases import databases_config as dbconfig
from graphdb_builder.databases.config import reactomeConfig as iconfig
from graphdb_builder import mapping as mp, utils
import re

#########################
#   Reactome database   #
#########################
def parser(download=True):
    urls = iconfig.reactome_urls
    entities = set()
    relationships = defaultdict(set)
    entities_header = iconfig.pathway_header
    relationships_headers = iconfig.relationships_header
    directory = os.path.join(dbconfig.databasesDir, "Reactome")
    utils.checkDirectory(directory)
    metabolite_mapping = mp.getMappingForEntity("Metabolite")
    drug_mapping = mp.getMappingForEntity("Drug")
    
    for dataset in urls:
        url = urls[dataset]
        file_name = url.split('/')[-1]
        if download:
            utils.downloadDB(url, directory)
        f = os.path.join(directory, file_name)
        with open(f, 'r') as rf:
            print(dataset)
            if dataset == "pathway":
                entities = parsePathways(rf)
            elif dataset == "hierarchy":
                relationships[("pathway", "has_parent")] = parsePathwayHierarchy(rf)
            elif dataset == "protein":
                relationships[(dataset, "annotated_to_pathway")] = parsePathwayRelationships(rf)
            elif dataset == "metabolite":
                relationships[(dataset, "annotated_to_pathway")] = parsePathwayRelationships(rf, metabolite_mapping)
            elif dataset == "drug":
                relationships[(dataset, "annotated_to_pathway")] = set()
    
    return entities, relationships, entities_header, relationships_headers
        
def parsePathways(fhandler):
    entities = set()
    organisms = iconfig.organisms
    url = iconfig.linkout_url
    directory = os.path.join(dbconfig.databasesDir, "Reactome")
    mapping_file = os.path.join(directory, "mapping.tsv")
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
                
    return entities

def parsePathwayHierarchy(fhandler):
    relationships = set()
    for line in fhandler:
        data = line.rstrip("\r\n").split("\t")
        parent = data[0]
        child = data[1]
        relationships.add((child, parent, "HAS_PARENT", "Reactome"))

    return relationships


def parsePathwayRelationships(fhandler, mapping=None):
    relationships = set()
    regex = r"(.+)\s\[(.+)\]" 
    organisms = iconfig.organisms
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
                    #print(name, identifier)
                    continue
            relationships.add((identifier, pathway, "ANNOTATED_TO_PATHWAY", evidence, organism, loc, "Reactome"))
    
    return relationships
