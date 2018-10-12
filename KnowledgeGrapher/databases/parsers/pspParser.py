import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import pspConfig as iconfig
from KnowledgeGrapher import mapping as mp
from collections import defaultdict
from KnowledgeGrapher import utils
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
    headers = iconfig.headers
    relationships = defaultdict()
    for entity, relationship_type in annotation_files:
        file_name = os.path.join(directory, annotation_files[(entity,relationship_type)])
        with gzip.open(file_name, 'r') as f:
            if entity == "disease":
                mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)
                relationships[(entity,relationship_type)] = parseDiseaseAnnotations(f, modifications, mapping)
            elif entity == "biological_process":
                mapping = mp.getMappingFromOntology(ontology = "Gene_ontology", source = None)
                relationships[(entity,relationship_type)] = parseRegulationAnnotations(f, modifications, mapping)
            elif entity == "modified_protein":
                relationships[(entity,relationship_type)] = parseKinaseSubstrates(f, modifications)
    return headers, relationships
    
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
        modified_protein_id = substrate+'_'+data[7]
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
                    relationships.add((modified_protein_id,process_code,"ASSOCIATED_WITH", "CURATED", 5, "PhosphoSitePlus", pmid,""))
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
