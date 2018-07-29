from KnowledgeGrapher.ontologies import ontologies_config as oconfig
from KnowledgeGrapher.databases import databases_config as dbconfig
import os.path
from collections import defaultdict
import re

def generateMappingFromReflect():
    types = [-26,-25]
    ofiles = oconfig.files
    odir = oconfig.ontologiesDirectory
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


def getMappingFromOntology(ontology, source = None):
    mapping = {}
    ont = oconfig.ontologies[ontology]
    dirFile = os.path.join(oconfig.ontologiesDirectory,ont)
    dataFile = os.path.join(dirFile,"mapping.tsv")
    with open(dataFile, 'r') as f:
        for line in f:
            data = line.rstrip("\r\n").split("\t")
            if data[1] == source or source is None:
                mapping[data[2]] = data[0]

    return mapping

def getMappingFromDatabase(mappingFile):
    mapping = {}
    with open(mappingFile, 'r') as mf:
        for line in mf:
            data = line.rstrip("\r\n")
            ident = data[0]
            alias = data[1]
            mapping[alias] = ident

    return mapping

def getSTRINGMapping(url, source = "BLAST_UniProt_AC", download = True):
    mapping = collections.defaultdict(set)
    
    directory = os.path.join(dbconfig.databasesDir, "STRING")
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        downloadDB(url, "STRING")
    
    f = os.path.join(directory, fileName)
    mf = gzip.open(f, 'r')
    first = True
    for line in mf:
        if first:
            first = False
            continue
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        stringID = data[0]
        alias = data[1]
        sources = data[2].split(' ')
        if source in sources:
            mapping[stringID].add(alias)
        
    return mapping

def buildMappingFromOBO(oboFile, ontology):
    print(oboFile)
    outputDir = os.path.join(oconfig.ontologiesDirectory, ontology)
    outputFile = os.path.join(outputDir, "mapping.tsv")
    identifiers = defaultdict(list)
    re_synonyms = r'\"(.+)\"'
    with open(oboFile, 'r') as f:
        for line in f:
            if line.startswith("id:"):
                ident = ":".join(line.rstrip("\r\n").split(":")[1:])
            if line.startswith("xref:"):
                source_ref = line.rstrip("\r\n").split(":")[1:]
                if len(source_ref) == 2:
                    identifiers[ident.strip()].append((source_ref[0].strip(), source_ref[1]))
            if line.startswith("synonym:"):
                synonym_type = "".join(line.rstrip("\r\n").split(":")[1:])
                matches = re.search(re_synonyms, synonym_type)
                if matches:
                     identifiers[ident.strip()].append(("SYN",matches.group(1)))
    with open(outputFile, 'w') as out:
        for ident in identifiers:
            for source, ref in identifiers[ident]:
                out.write(ident+"\t"+source+"\t"+ref+"\n")
