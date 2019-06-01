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
def parser(databases_directory, import_directory, download=True):
    result = {"Protein":None, "Known_variant":None, "Peptide":None}
    config = ckg_utils.get_configuration('../databases/config/uniprotConfig.yml')
    uniprot_texts_file = config['uniprot_text_file']
    relationships_header = config['relationships_header']
    proteins, relationships = parseUniProtDatabase(config, databases_directory, download=download)
    addUniProtTexts(uniprot_texts_file, proteins)
    entities_header = config['proteins_header']
    entities = set()
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
        entities.add((protein, "Protein", accession , name, ",".join(synonyms), description, taxid))
    
    stats = print_out_file(entities, entities_header, relationships, relationships_header, "Protein", import_directory)
    
    #Peptides
    entities, relationships = parseUniProtPeptides(config, databases_directory, download)
    entities_header = config['peptides_header']
    stats.update(print_out_file(entities, entities_header, relationships, relationships_header, "Pepetide", import_directory))

    #Variants
    entities, relationships = parseUniProtVariants(config, databases_directory, download)
    entities_header = config['variants_header']
    stats.update(print_out_file(entities, entities_header, relationships, relationships_header, "Known_variant", import_directory))
    
    #Gene ontology annotation
    entities = None
    entities_header = None
    relationships = parseUniProtAnnotations(config, databases_directory, download)
    relationships_header = config['go_header']
    stats.update(print_out_file(entities, entities_header, relationships, relationships_header, "go", import_directory))

    return stats 

def parse_id_mapping_file(config, databases_directory, import_directory, download=False):
    taxids = config['species']
    proteins = {}

    url = config['uniprot_id_url']
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    mapping_file = os.path.join(directory, 'mapping.tsv')
    if download:
        print("Downloading")
        builder_utils.downloadDB(url, directory)

    fields = config['uniprot_ids']
    synonymFields = config['uniprot_synonyms']
    identifier = None
    transcripts = set()
    skip = set()
    print("Reading file")
    is_first = True
    uf = builder_utils.read_gzipped_file(file_name)
    aux = {}
    with open(mapping_file, 'w') as mf:
        for line in uf:
            data = line.decode('utf-8').rstrip("\r\n").split("\t")
            iid = data[0]
            field = data[1]
            alias = data[2]
            
            if iid not in skip:
                skip = set()
                if '-' in iid:
                    transcripts.add(iid)
                if iid not in aux and iid.split('-')[0] not in aux:
                    if identifier is not None:
                        prot_info["synonyms"] = synonyms
                        aux[identifier].update(prot_info)
                        if "UniProtKB-ID" in aux[identifier] and "NCBI_TaxID" in aux[identifier]:
                            proteins[identifier] = aux[identifier]
                            for t in transcripts:
                                proteins[t] = aux[identifier]
                            transcripts = set()
                            aux.pop(identifier, None)
                            if len(proteins) >= 1000:
                                entities, relationships, pdb_ent, pdb_rel = format_output(proteins)
                                print_files(entities, relationships, pdb_entities, is_first)
                                is_first = False
                                proteins = {}
                    identifier = iid
                    transcripts = set()
                    aux[identifier] = {}
                    prot_info = {}
                    synonyms = []
                if field in fields:
                    if field == 'NCBI_TaxID':
                        if int(alias) not in taxids:
                            skip.add(identifier)
                            aux.pop(identifier, None)
                            identifier = None
                    if field in synonymFields:
                        synonyms.append(alias)
                        mf.write(identifier+"\t"+alias+"\n")
                    prot_info.setdefault(field, [])
                    prot_info[field].append(alias)

    uf.close()
    print("Done reading")

    
    if len(proteins)>0:
        entities, relationships, pdb_ent, pdb_rel = format_output(proteins)
        print_files(entities, relationships, pdb_ent, pdb_rel, is_first)
    
    print("Done")


def format_output(proteins):
    entities = set()
    relationships = defaultdict(set())
    pdb_entities = set()
    for protein in proteins:
        accession = []
        name = []
        synonyms = []
        description = []
        taxid = []
        pdb = []
        if "UniProtKB-ID" in proteins[protein]:
            accession  = proteins[protein]["UniProtKB-ID"]
        if "Gene_Name" in proteins[protein]:
            name = proteins[protein]["Gene_Name"]
        if "synonyms" in proteins[protein]:
            synonyms = proteins[protein]["synonyms"]
        if "NCBI_TaxID" in proteins[protein]:
            taxid = proteins[protein]["NCBI_TaxID"]
            relationships[('Protein','BELONGS_TO_TAXONOMY')].add((protein, int(",".join(taxid)), "BELONGS_TO_TAXONOMY", 'UniProt'))
        if 'PDB' in proteins[protein]:
            pdb = proteins[protein]['PDB']
            for i in pdb:
                pdb_ent.add((i, 'Protein_structure', 'Uniprot', 'http://www.rcsb.org/structure/{}'.format(i)))
                relationships[('Protein','HAS_STRUCTURE')].add((protein, i, "HAS_STRUCTURE", 'UniProt'))
        if "description" in proteins[protein]:
            description = proteins[protein]["description"]
        entities.add((protein, "Protein", ",".join(accession), ",".join(name), ",".join(synonyms), ",".join(description), int(",".join(taxid))))
            
    return entities, relationships, pdb_entities

def print_files(entities, relationships, pdb_ent, pdb_rel, is_first):
    entities_header = ['ID', ':LABEL', 'accession', 'name', 'synonyms', 'description', 'taxid'] 
    rel_header = ['START_ID', 'END_ID', 'TYPE', 'source']
    pdb_ent_header = ['ID', ':LABEL', 'source', 'link']
    pdb_rel_header = ['START_ID', 'END_ID', 'TYPE', 'source']
    
    entities_outputfile = '../../data/imports/Proteins.tsv'
    rel_outputfile = '../../data/imports/Protein_belongs_to_taxonomy.tsv'
    pdb_ent_outputfile = '../../data/imports/Protein_structures.tsv'
    pdb_rel_outputfile = '../../data/imports/Protein_has_structure.tsv'
    
    edf = pd.DataFrame(list(entities), columns=entities_header)
    rdf = pd.DataFrame(list(relationships), columns=rel_header)
    pedf = pd.DataFrame(list(pdb_ent), columns=pdb_ent_header)
    prdf = pd.DataFrame(list(pdb_rel), columns=pdb_rel_header)
    
    header = is_first
    with open(entities_outputfile, 'a') as ef:
        edf.to_csv(path_or_buf=ef, sep='\t',
                header=is_first, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')
    with open(rel_outputfile, 'a') as rf:
        rdf.to_csv(path_or_buf=rf, sep='\t',
                header=is_first, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')
    with open(pdb_ent_outputfile, 'a') as pef:
        pedf.to_csv(path_or_buf=pef, sep='\t',
                header=is_first, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')
    with open(pdb_rel_outputfile, 'a') as prf:
        prdf.to_csv(path_or_buf=prf, sep='\t',
                header=is_first, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')


def print_out_file(entities, entities_header, relationships, relationship_header, dataset, import_directory):
    stats = set()
    if entities is not None:
        outputfile = os.path.join(import_directory, dataset+".csv")
        builder_utils.write_entities(entities, entities_header, outputfile)
        #logger.info("Database {} - Number of {} entities: {}".format(database, dataset, len(entities)))
        stats.add(builder_utils.buildStats(len(entities), "entity", dataset, "UniProt", outputfile))
        for entity, rel in relationships:
            outputfile = os.path.join(import_directory, "uniprot_"+entity.lower()+"_"+rel.lower()+".csv")
            builder_utils.write_relationships(relationships[(entity,rel)], relationship_header, outputfile)
            #logger.info("Database {} - Number of {} relationships: {}".format(database, rel, len(relationships[(entity,rel)])))
            stats.add(builder_utils.buildStats(len(relationships[(entity,rel)]), "relationships", rel, "UniProt", outputfile))

    return stats

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
    chromosome_regex = r"(\w+)[p|q]"
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
                    ident = chromosome+":"+var_matches.group(1)
                    consequence = data[4]
                    altName = [externalID, data[5], pvariant, chromosome_coord]
                    if ref in aa and alt in aa:
                        altName.append(aa[ref]+pos+aa[alt])
                    pvariant = protein+"_"+pvariant
                    entities.add((ident, "Known_variant", pvariant, ",".join(altName), impact, clin_relevance, disease, original_source, "UniProt"))
                    if chromosome != 'chr-':
                        relationships[('Chromosome','known_variant_found_in_chromosome')].add((ident, chromosome.replace('chr',''), "VARIANT_FOUND_IN_CHROMOSOME","UniProt"))
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
