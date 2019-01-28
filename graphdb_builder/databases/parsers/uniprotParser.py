import os.path
import gzip
from collections import defaultdict
import pandas as pd
import re
import time
import ckg_utils
from graphdb_builder import builder_utils

#########################
#       UniProt         # 
#########################
def parser(databases_directory, download=True):
    result = {"Protein":None, "Known_variant":None, "Peptide":None}
    config = ckg_utils.get_configuration('../databases/config/uniprotConfig.yml')
    uniprot_texts_file = config['uniprot_text_file']
    relationships_header = config['relationships_header']
    proteins, proteins_relationships = parseUniProtDatabase(config, databases_directory, download=download)
    addUniProtTexts(uniprot_texts_file, proteins)
    proteins_outputfileName = "Protein.csv"
    proteins_header = config['proteins_header']
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
    peptides, peptides_relationships = parseUniProtPeptides(config, databases_directory, download)
    peptides_outputfile = "Peptides.csv"
    peptides_header = config['peptides_header']
    result["Peptide"] = (peptides, peptides_relationships, peptides_header, relationships_header)

    #Variants
    variants, variants_relationships = parseUniProtVariants(config, databases_directory, download)
    variants_outputfile = "Known_variant.csv"
    variants_header = config['variants_header']
    result["Known_variant"] = (variants, variants_relationships, variants_header, relationships_header)
    
    #Gene ontology annotation
    go_annotations = parseUniProtAnnotations(config, databases_directory, download)
    go_annotations_header = config['go_header']
    result["go"] = (None, go_annotations, None, go_annotations_header)

    return result


def parseUniProtDatabase(config, databases_directory, download=True):
    proteins = {}
    relationships = defaultdict(set)

    url = config['uniprot_id_url']
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    mapping_file = os.path.join(directory, 'mapping.tsv')
    if download:
        builder_utils.downloadDB(url, directory)

    fields = config['uniprot_ids']
    synonymFields = config['uniprot_synonyms']
    protein_relationships = config['uniprot_protein_relationships']
    identifier = None
    with open(mapping_file, 'w') as mf:
        with gzip.open(file_name, 'r') as uf:
            for line in uf:
                data = line.decode('utf-8').rstrip("\r\n").split("\t")
                iid = data[0]
                field = data[1]
                alias = data[2]

                mf.write(iid+"\t"+alias+"\n")
                
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
                        relationships[tuple(protein_relationships[field])].add((iid, alias, protein_relationships[field][1], "UniProt"))
    
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

def parseUniProtVariants(config, databases_directory, download=True):
    data = defaultdict()
    variant_regex = r"(g\.\w+>\w)"
    chromosome_regex = r"(\w+)p"
    url = config['uniprot_variant_file']
    aa = config['amino_acids']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
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
                pvariant = data[2]
                externalID = data[3]
                impact = data[4]
                clin_relevance = data[5]
                disease = data[6]
                chromosome_coord = data[8]
                original_source = data[13]
                ref = pvariant[2:5]
                pos = pvariant[5:-3]
                alt = pvariant[-3:]
                var_matches = re.search(variant_regex, data[9])
                chr_matches = re.search(chromosome_regex, chromosome_coord)
                if var_matches and chr_matches:
                    chromosome = 'chr'+chr_matches.group(1)
                    ident = chromosome+":".join(var_matches.groups())
                    consequence = data[4]
                    altName = [externalID, data[5], pvariant, chromosome_coord]
                    if ref in aa and alt in aa:
                        altName.append(aa[ref]+pos+aa[alt])
                    pvariant = protein+"_"+pvariant
                    entities.add((ident, "Known_variant", pvariant, ",".join(altName), impact, clin_relevance, disease, original_source, "UniProt"))
                    if chromosome != 'chr-':
                        relationships[('Chromosome','known_variant_found_in_chromosome')].add((ident, chromosome, "VARIANT_FOUND_IN_CHROMOSOME","UniProt"))
                    if gene != "":
                        relationships[('Gene','known_variant_found_in_gene')].add((ident, gene, "VARIANT_FOUND_IN_GENE", "UniProt"))
                    if protein !="":
                        relationships[('Protein','known_variant_found_in_protein')].add((ident, protein, "VARIANT_FOUND_IN_PROTEIN", "UniProt"))

    return entities, relationships

def parseUniProtAnnotations(config, databases_directory, download=True):
    roots = {'F':'Molecular_function', 'C':'Cellular_component', 'P':'Biological_process'}
    url = config['uniprot_go_annotations']
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
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

def parseUniProtPeptides(config, databases_directory, download=True):
    file_urls = config['uniprot_peptides_files']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    for url in file_urls:
        fileName = os.path.join(directory, url.split('/')[-1])
        if download:
            builder_utils.downloadDB(url, directory)
        first = True
        with open(fileName, 'r') as f:
            for line in f:
                if first:
                    first = False
                    continue
    
                data = line.rstrip("\r\n").split("\t")
                peptide = data[0]
                accs = data[5].split(",")
                is_unique = True
                if len(accs) > 1:
                    is_unique = False
                entities.add((peptide, "Peptide", "tryptic peptide", is_unique))
                for protein in accs:
                    relationships[("Peptide", 'belongs_to_protein')].add((peptide, protein, "BELONGS_TO_PROTEIN", "UniProt"))
    return entities, relationships
    
if __name__ == "__main__":
    pass
