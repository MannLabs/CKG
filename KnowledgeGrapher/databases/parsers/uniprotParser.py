import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import uniprotConfig as iconfig
from collections import defaultdict
from KnowledgeGrapher import utils
import pandas as pd
import re
import time

#########################
#       UniProt         # 
#########################
def parser(download=True):
    result = {"Protein":None, "Known_variant":None, "Peptide":None}
    uniprot_texts_file = iconfig.uniprot_text_file    
    relationships_header = iconfig.relationships_header
    proteins, proteins_relationships = parseUniProtDatabase(download=download)
    addUniProtTexts(uniprot_texts_file, proteins)
    proteins_outputfileName = "Protein.csv"
    proteins_header = iconfig.proteins_header
    protein_entities = set()
    for protein in proteins:
        accession = ""
        name = ""
        synonyms = []
        taxid = 9606
        description = ""
        if "UniProtKB-ID" in proteins[protein]:
            accession  = proteins[protein]["UniProtKB-ID"]
        if "Gene_Name" in proteins[protein]:
            name = proteins[protein]["Gene_Name"]
        if "synonyms" in proteins[protein]:
            synonyms = proteins[protein]["synonyms"]
        if "NCBI_TaxID" in proteins[protein]:
            taxid = int(proteins[protein]["NCBI_TaxID"])
        if "description" in proteins[protein]:
            description = proteins[protein]["description"]
        protein_entities.add((protein, "Protein", accession , name, ",".join(synonyms), description, taxid))

    result["Protein"] = (protein_entities, proteins_relationships, proteins_header, relationships_header)
    
    #Peptides
    peptides, peptides_relationships = parseUniProtPeptides()
    peptides_outputfile = "Peptides.csv"
    peptides_header = iconfig.peptides_header
    result["Peptide"] = (peptides, peptides_relationships, peptides_header, relationships_header)

    #Variants
    variants, variants_relationships = parseUniProtVariants()
    variants_outputfile = "Known_variant.csv"
    variants_header = iconfig.variants_header
    result["Known_variant"] = (variants, variants_relationships, variants_header, relationships_header)
    
    #Gene ontology annotation
    go_annotations = parseUniProtAnnotations()
    go_annotations_header = iconfig.go_header
    result["go"] = (None, go_annotations, None, go_annotations_header)

    return result


def parseUniProtDatabase(download=True):
    proteins = {}
    relationships = defaultdict(set)

    url = iconfig.uniprot_id_url
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, directory)

    fields = iconfig.uniprot_ids
    synonymFields = iconfig.uniprot_synonyms
    protein_relationships = iconfig.uniprot_protein_relationships
    identifier = None
    with gzip.open(file_name, 'r') as uf:
        for line in uf:
            data = line.decode('utf-8').rstrip("\r\n").split("\t")
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

def parseUniProtVariants(download = False):
    data = defaultdict()
    url = iconfig.uniprot_variant_file
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, directory)
    with gzip.open(fileName, 'r') as f:
        din = False
        i = 0
        for line in f:
            line = line.decode('utf-8')
            if not line.startswith('#') and not din:
                continue
            elif i<=2:
                din = True
                i += 1
                continue
            data = line.rstrip("\r\n").split("\t")
            if len(data) > 9:
                gene = data[0]
                protein = data[1]
                ident = re.sub('[a-z|\.]','', data[2])
                altName = [data[3]]
                altName.append(data[5])
                consequence = data[4]
                mutIdent = re.sub('NC_\d+\.', 'chr', data[9])
                altName.append(mutIdent)
                if len(data[9].split('.')) >1:
                    chromosome = data[9].split('.')[1].split(':')[0]
                else:
                    chromosome = data[9]
                entities.add((ident, "Known_variant", ",".join(altName)))
                if chromosome != '-':
                    relationships[('Chromosome','known_variant_found_in_chromosome')].add((ident, chromosome, "VARIANT_FOUND_IN_CHROMOSOME","UniProt"))
                if gene != "":
                    relationships[('Gene','known_variant_found_in_gene')].add((ident, gene, "VARIANT_FOUND_IN_GENE", "UniProt"))
                if protein !="":
                    relationships[('Protein','known_variant_found_in_protein')].add((ident, protein, "VARIANT_FOUND_IN_PROTEIN", "UniProt"))

    return entities, relationships

def parseUniProtAnnotations(download=False):
    roots = {'F':'Molecular_function', 'C':'Cellular_component', 'P':'Biological_process'}
    url = iconfig.uniprot_go_annotations
    relationships = defaultdict(set)
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, directory)
    with gzip.open(fileName, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            if line.startswith('!'):
                continue
            data = line.rstrip("\r\n").split("\t")
            identifier = data[1]
            go = data[4]
            evidence = data[6]
            root = data[8]
            if root in roots:
                root = roots[root]
                relationships[(root,'associated_with')].add((identifier, go, "ASSOCIATED_WITH", evidence, 5, "UniProt"))

    return relationships

def parseUniProtPeptides(download=True):
    file_urls = iconfig.uniprot_peptides_files
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(dbconfig.databasesDir,"UniProt")
    utils.checkDirectory(directory)
    for url in file_urls:
        fileName = os.path.join(directory, url.split('/')[-1])
        if download:
            utils.downloadDB(url, directory)
        first = True
        with open(fileName, 'r') as f:
            for line in f:
                if first:
                    first = False
                    continue
    
                data = line.rstrip("\r\n").split("\t")
                peptide = data[0]
                groups = data[3].split(';')
                is_unique = True
                if len(groups) > 1:
                    is_unique = False
                entities.add((peptide, "Peptide", "tryptic peptide", is_unique))
                for accs in groups:
                    for protein in accs.split(','):
                        relationships[("Peptide", 'belongs_to_protein')].add((peptide, protein, "BELONGS_TO_PROTEIN", "UniProt"))
    return entities, relationships
    
if __name__ == "__main__":
    pass
