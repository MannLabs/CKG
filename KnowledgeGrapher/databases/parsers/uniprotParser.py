import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import uniprotConfig as iconfig
from collections import defaultdict
from KnowledgeGrapher import utils
import pandas as pd
import re

#########################
#       UniProt         # 
#########################
def parseUniProtDatabase(dataFile):
    proteins = {}
    relationships = defaultdict(set)

    fields = iconfig.uniprot_ids
    synonymFields = config.uniprot_synonyms
    protein_relationships = iconfig.uniprot_protein_relationships
    identifier = None
    with open(dataFile, 'r') as uf:
        for line in uf:
            data = line.rstrip("\r\n").split("\t")
            iid = data[0]
            field = data[1]
            alias = data[2]
            
            if iid not in proteins:
                if identifier is not None:
                    prot_info["synonyms"] = synonyms
                    proteins[identifier] = prot_info
                identifier = iid
                proteins[identifier] = {}
                prot_info = {}
                synonyms = []
            if field in fields:
                if field in synonymFields:
                    prot_info[field] = alias
                    synonyms.append(alias)
                if field in protein_relationships:
                    relationships[protein_relationships[field]].add((iid, alias, protein_relationships[field][1], "UniProt"))
    
    return proteins, relationships

def addUniProtTexts(textsFile, proteins):
    with open(textsFile, 'r') as tf:
        for line in tf:
            data = line.rstrip("\r\n").split("\t")
            protein = data[0]
            name = data[1]
            function = data[3]
            
            if protein in proteins:
                proteins[protein].update({"description":function})

def parseUniProtVariants(download = True):
    data = defaultdict()
    url = iconfig.uniprot_variant_file
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "UniProt")
    with gzip.open(fileName, 'r') as f:
        din = False
        i = 0
        for line in f:
            line = line.decode('utf-8')
            if not line.startswith('#') and not din:
                continue
            elif i<2:
                din = True
                i += 1
                continue
            data = line.rstrip("\r\n").split("\t")
            gene = data[0]
            protein = data[1]
            ident = re.sub('[a-z|\.]','', data[2])
            altName = [data[3]]
            altName.append(data[5])
            consequence = data[4]
            mutIdent = re.sub('NC_\d+\.', 'chr', data[9])
            altName.append(mutIdent)
            chromosome = 'chr'+data[9].split('.')[1].split(':')[0]

            entities.add((ident, "Known_variant", ",".join(altName)))
            relationships['known_variant_found_in_chromosome'].add((ident, chromosome, "VARIANT_FOUND_IN_CHROMOSOME"))
            relationships['known_variant_found_in_gene'].add((ident, gene, "VARIANT_FOUND_IN_GENE"))
            relationships['known_variant_found_in_protein'].add((ident, protein, "VARIANT_FOUND_IN_PROTEIN"))

    return entities, relationships


def parseUniProtUniquePeptides(download=True):
    url = iconfig.uniprot_unique_peptides_file
    entities = set()
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    checkDirectory
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "UniProt")

    with open(fileName, 'r') as f:
        for line in f:
            data = line.rstrip("\r\n").split("\t")
