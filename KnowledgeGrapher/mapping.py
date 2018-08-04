from KnowledgeGrapher.ontologies import ontologies_config as oconfig
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher import utils
import os.path
import time
from collections import defaultdict
import re
import gzip


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
    while not os.path.isfile(mappingFile):
        time.sleep(5)
    with open(mappingFile, 'r') as mf:
        for line in mf:
            data = line.rstrip("\r\n").split("\t")
            ident = data[0]
            alias = data[1]
            mapping[alias] = ident

    return mapping

def getSTRINGMapping(url, source = "BLAST_UniProt_AC", download = True, db = "STRING"):
    mapping = defaultdict(set)
    
    directory = os.path.join(dbconfig.databasesDir, db)
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        utils.downloadDB(url, db)
    
    f = os.path.join(directory, fileName)
    mf = gzip.open(f, 'r')
    first = True
    for line in mf:
        if first:
            first = False
            continue
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        if db == "STRING":
            stringID = data[0]
            alias = data[1]
            sources = data[2].split(' ')
        else:
            stringID = data[0]
            alias = data[2]
            sources = data[3].split(' ')
            if not alias.startswith('DB'):
                continue
        
        if source in sources:
            mapping[stringID].add(alias)
        
    return mapping

def updateMappingFileWithSTRING(mappingFile, mapping, db = "STRING"):
    directory = os.path.join(dbconfig.databasesDir, db)
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        utils.downloadDB(url, db)
    
    f = os.path.join(directory, fileName)
    mf = gzip.open(f, 'r')
    first = True
    with open(mappingFile, 'a') as mf:
        for line in mf:
            if first:
                first = False
                continue
            data = line.decode('utf-8').rstrip("\r\n").split("\t")
            if db == "STRING":
                stringID = data[0]
                alias = data[1]
                sources = data[2].split(' ')
            else:
                stringID = data[0]
                alias = data[2]
                sources = data[3].split(' ')
                if not alias.startswith('DB'):
                    continue
            
            if stringID in mapping:
                for ident in mapping[stringID]:
                    mf.write(ident+"\t"+stringID)
                    for alias in allAlias[stringID]:
                        mf.write(ident+"\t"+alias)

def buildMappingFromOBO(oboFile, ontology):
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

