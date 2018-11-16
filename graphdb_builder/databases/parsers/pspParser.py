import os.path
import gzip
from graphdb_builder.databases import databases_config as dbconfig
from graphdb_builder.databases.config import pspConfig as iconfig
from graphdb_builder import mapping as mp, utils
from collections import defaultdict
import pandas as pd
import re

#############################
#       PhosphoSitePlus     # 
#############################
def parser():
    directory = os.path.join(dbconfig.databasesDir,"PhosphoSitePlus")
    utils.checkDirectory(directory)
    modifications = iconfig.modifications
    annotation_files = iconfig.annotation_files
    entities_header = iconfig.entities_header
    relationships_headers = iconfig.rel_headers
    entities = set()
    relationships = defaultdict(set)
    for site_file in iconfig.site_files:
        file_name = os.path.join(directory, site_file)
        with gzip.open(file_name, 'r') as f:
            sites, site_relationships = parseSites(f, modifications)
            entities.update(sites)
            for r in site_relationships:
                relationships[r].update(site_relationships[r])
    for entity, relationship_type in annotation_files:
        file_name = os.path.join(directory, annotation_files[(entity,relationship_type)])
        with gzip.open(file_name, 'r') as f:
            if entity == "disease":
                mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)
                relationships[(entity,relationship_type)].update(parseDiseaseAnnotations(f, modifications, mapping))
            elif entity == "biological_process":
                mapping = mp.getMappingFromOntology(ontology = "Gene_ontology", source = None)
                relationships[(entity,relationship_type)].update(parseRegulationAnnotations(f, modifications, mapping))
            elif entity == "substrate":
                relationships[(entity,relationship_type)] = parseKinaseSubstrates(f, modifications)
    
    return entities, relationships, entities_header, relationships_headers
    
def parseSites(fhandler, modifications):
    entities = set()
    relationships = defaultdict(set)
    i = 0
    for line in fhandler:
        if i < 4:
            i += 1
            continue
        data = line.decode("utf-8").rstrip("\r\n").split("\t")
        protein = data[2]
        residue_mod = data[4].split('-')
        modified_protein_id = protein+'_'+data[4]
        organism = data[6]
        seq_window = data[9]
        if len(residue_mod) > 1:
            modification = modifications[residue_mod[1]]
            position = residue_mod[0][0]
            residue = ''.join(residue_mod[0][1:])
            if organism == "human":
                #"sequence_window", "position", "Amino acid"
                entities.add((modified_protein_id, "Modified_protein", protein, seq_window, position, residue))
                relationships[("Protein", "has_modified_site")].add((protein, modified_protein_id, "HAS_MODIFIED_SITE", "PhosphositePlus"))
                relationships[("Modified_protein", "has_modification")].add((modified_protein_id, modification, "HAS_MODIFICATION", "PhosphositePlus"))
    
    return entities, relationships

def parseKinaseSubstrates(fhandler, modifications):
    relationships = set()
    i = 0
    for line in fhandler:
        if i < 4:
            i += 1
            continue
        data = line.decode("utf-8").rstrip("\r\n").split("\t")
        kinase = data[2]
        organism = data[3]
        substrate = data[6]
        residue_mod = data[9].split('-')
        modified_protein_id = substrate+'_'+data[9]+'-p'
        if organism == "human":
            relationships.add((modified_protein_id,kinase,"IS_SUBSTRATE_OF", "CURATED", 5, "PhosphoSitePlus"))
    return relationships
    
def parseRegulationAnnotations(fhandler, modifications, mapping):
    relationships = set()
    i = 0
    for line in fhandler:
        if i < 4:
            i += 1
            continue
        data = line.decode("utf-8").rstrip("\r\n").split("\t")
        protein = data[3]
        organism = data[6]
        residue_mod = data[7].split('-')
        modified_protein_id = protein+'_'+data[7]
        functions  = data[11].split('; ')
        processes = data[12].split('; ')
        pmid = data[15]
        if organism == "human":
            for process in processes:
                if process.lower() in mapping:
                    process_code = mapping[process.lower()]
                    relationships.add((modified_protein_id,process_code,"ASSOCIATED_WITH", "CURATED", 5, "PhosphoSitePlus", pmid,"unspecified"))
                elif process.lower().split(',')[0] in mapping:
                    process_code = mapping[process.lower().split(',')[0]]
                    relationships.add((modified_protein_id,process_code,"ASSOCIATED_WITH", "CURATED", 5, "PhosphoSitePlus", pmid,process.lower().split(',')[1]))
                else:
                    pass
    return relationships

def parseDiseaseAnnotations(fhandler, modifications, mapping):
    relationships = set()
    i = 0
    for line in fhandler:
        if i < 4:
            i += 1
            continue
        data = line.decode("utf-8").rstrip("\r\n").split("\t")
        if len(data) > 13:
            diseases = data[0].split('; ')
            alteration = data[1]
            protein = data[4]
            organism = data[8]
            internalid = data[9]
            residue_mod = data[10].split('-')
            modified_protein_id = protein+'_'+data[10]
            pmid = data[13]
            if organism == "human":
                for disease_name in diseases:
                    if disease_name.lower() in mapping:
                        disease_code = mapping[disease_name.lower()]
                        relationships.add((modified_protein_id,disease_code,"ASSOCIATED_WITH", "CURATED", 5, "PhosphoSitePlus", pmid))
    return relationships

if __name__ == "__main__":
    pass
