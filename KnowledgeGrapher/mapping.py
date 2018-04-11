import ontologies_config as config
import os.path

def generateMappingFromReflect():
    types = [-26,-25,-1]
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
