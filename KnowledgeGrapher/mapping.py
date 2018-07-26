from KnowledgeGrapher.ontologies import ontologies_config as config
import os.path

def generateMappingFromReflect():
    types = [-26,-25]
    ofiles = config.files
    odir = config.ontologiesDirectory
    mapping = {}
    for t in types:
        files = ofiles[t]
        entitiesFile, namesFile = files[0:2]
        entitiesFile = os.path.join(odir,entitiesFile)
        namesFile = os.path.join(odir,namesFile)
        entities = {}
        with open(entitiesFile, 'r') as ef:
            for line in ef:
                data = line.rstrip("\r\n").split("\t")
                internal = data[0]
                external = data[2]

                entities[internal] = external
	
        with open(namesFile, 'r') as nf:
            for line in nf:
                data = line.rstrip("\r\n").split("\t")
                internal = data[0]
                name = data[1]
                mapping[name.lower()] = entities[internal]
    return mapping


def getMappingFromOntology(ontology, source):
    mapping = {}
    ont = config.ontologies[ontology]
    dirFile = os.path.join(config.ontologiesDirectory,ont)
    dataFile = os.path.join(dirFile,"mapping.tsv")
    with open(dataFile, 'r') as f:
        for line in f:
            data = line.rstrip("\r\n").split("\t")
            if data[1] == source:
                mapping[data[2]] = data[0]

    return mapping
