from graphdb_builder import builder_utils
import config.ckg_config as ckg_config
import ckg_utils
import os.path
import time
from collections import defaultdict
import re
import gzip

try:
    oconfig = builder_utils.setup_config('ontologies')
    dbconfig = builder_utils.setup_config('databases')
except Exception as err:
    raise Exception("mapping - Reading configuration > {}.".format(err))


def reset_mapping(entity):
    if entity in dbconfig["sources"]:
        directory = os.path.join(dbconfig["databasesDir"], dbconfig["sources"][entity])
        mapping_file = os.path.join(directory,"complete_mapping.tsv")
        if os.path.exists(mapping_file):
            os.remove(mapping_file) 

def mark_complete_mapping(entity):
    if entity in dbconfig["sources"]:
        directory = os.path.join(dbconfig["databasesDir"], dbconfig["sources"][entity])
        mapping_file = os.path.join(directory,"mapping.tsv")
        new_mapping_file = os.path.join(directory,"complete_mapping.tsv")
        if os.path.exists(mapping_file):
            os.rename(mapping_file, new_mapping_file)

def getMappingFromOntology(ontology, source = None):
    mapping = {}
    ont = oconfig["ontologies"][ontology]
    dirFile = os.path.join(oconfig["ontologies_directory"],ont)
    mapping_file = os.path.join(dirFile,"complete_mapping.tsv")
    max_wait = 0
    while not os.path.isfile(mapping_file) and max_wait < 5000:
        time.sleep(5)
        max_wait += 1

    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                data = line.rstrip("\r\n").split("\t")
                if data[1] == source or source is None:
                    mapping[data[2].lower()] = data[0]
    except:
        raise Exception("mapping - No mapping file {} for entity {}".format(mapping_file, ontology))


    return mapping

def getMappingForEntity(entity):
    mapping = {}
    if entity in dbconfig["sources"]:
        mapping_file = os.path.join(dbconfig["databasesDir"], os.path.join(dbconfig["sources"][entity],"complete_mapping.tsv"))
        max_wait = 0
        while not os.path.isfile(mapping_file) and max_wait < 5000:
            time.sleep(5)
            max_wait += 1
        try:
            with open(mapping_file, 'r') as mf:
                for line in mf:
                    data = line.rstrip("\r\n").split("\t")
                    if len(data) > 1:
                        ident = data[0]
                        alias = data[1]
                        mapping[alias] = ident
        except:
            raise Exception("mapping - No mapping file {} for entity {}".format(mapping_file, entity))

    return mapping

def getMultipleMappingForEntity(entity):
    mapping = defaultdict(set)
    if entity in dbconfig["sources"]:
        mapping_file = os.path.join(dbconfig["databasesDir"], os.path.join(dbconfig["sources"][entity],"complete_mapping.tsv"))
        max_wait = 0
        while not os.path.isfile(mapping_file) and max_wait < 5000:
            time.sleep(5)
            max_wait += 1
        
        try:
            with open(mapping_file, 'r') as mf:
                for line in mf:
                    data = line.rstrip("\r\n").split("\t")
                    if len(data) > 1:
                        ident = data[0]
                        alias = data[1]
                        mapping[alias].add(ident)
        except:
            raise Exception("mapping - No mapping file {} for entity {}".format(mapping, entity))
            
    return mapping

def getSTRINGMapping(url, source = "BLAST_UniProt_AC", download = True, db = "STRING"):
    mapping = defaultdict(set)
    
    directory = os.path.join(dbconfig["databasesDir"], db)
    file_name = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, directory)
    
    f = os.path.join(directory, file_name)
    first = True
    with gzip.open(f, 'rb') as mf:
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
    directory = os.path.join(dbconfig["databasesDir"], db)
    file_name = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, db)
    
    f = os.path.join(directory, file_name)
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
    outputDir = os.path.join(oconfig["ontologies_directory"], ontology)
    cmapping_file = os.path.join(outputDir, "complete_mapping.tsv")
    mapping_file = os.path.join(outputDir, "mapping.tsv")
    identifiers = defaultdict(list)
    re_synonyms = r'\"(.+)\"'
    
    if os.path.exists(cmapping_file):
            os.remove(cmapping_file)

    with open(oboFile, 'r') as f:
        for line in f:
            if line.startswith("id:"):
                ident = ":".join(line.rstrip("\r\n").split(":")[1:])
            elif line.startswith("name:"):
                name = "".join(line.rstrip("\r\n").split(':')[1:])
                identifiers[ident.strip()].append(("NAME", name.lstrip()))
            elif line.startswith("xref:"):
                source_ref = line.rstrip("\r\n").split(":")[1:]
                if len(source_ref) == 2:
                    identifiers[ident.strip()].append((source_ref[0].strip(), source_ref[1]))
            elif line.startswith("synonym:"):
                synonym_type = "".join(line.rstrip("\r\n").split(":")[1:])
                matches = re.search(re_synonyms, synonym_type)
                if matches:
                     identifiers[ident.strip()].append(("SYN",matches.group(1).lstrip()))
    with open(mapping_file, 'w') as out:
        for ident in identifiers:
            for source, ref in identifiers[ident]:
                out.write(ident+"\t"+source+"\t"+ref+"\n")

    os.rename(mapping_file, cmapping_file)


if __name__ == "__main__":
    pass
    #buildMappingFromOBO(oboFile = '../../data/ontologies/DO/do.obo', ontology='DO')
