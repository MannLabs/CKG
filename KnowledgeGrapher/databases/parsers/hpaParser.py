import os.path
import zipfile
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import hpaConfig as iconfig
from KnowledgeGrapher import utils
from collections import defaultdict
from KnowledgeGrapher import mapping as mp
import numpy as np
import pandas as pd

##########################################
#   Human Protein Atlas (pathology)      # 
##########################################
def parser(download = True):
    url = iconfig.hpa_pathology_url
    disease_mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)
    protein_mapping = mp.getMappingForEntity("Protein")
    directory = os.path.join(dbconfig.databasesDir, "HPA")
    utils.checkDirectory(directory)
    compressed_fileName = os.path.join(directory, url.split('/')[-1])
    file_name = '.'.join(url.split('/')[-1].split('.')[0:2])
    relationships_headers = iconfig.relationships_headers

    if download:
        utils.downloadDB(url, directory)
    
    with zipfile.ZipFile(compressed_fileName) as z:
        if file_name == "pathology.tsv":
            pathology = parsePathologyFile(z, file_name, protein_mapping, disease_mapping)
    
    return (pathology, relationships_headers)

def parsePathologyFile(fhandler, file_name, protein_mapping, disease_mapping):
    url = iconfig.linkout_url
    pathology = defaultdict(set)
    first = True
    with fhandler.open(file_name) as f:
        df = pd.read_csv(f, sep='\t', header=None, error_bad_lines=False, low_memory=False)
        df = df.fillna(0) 
        first = True
        for index, row in df.iterrows():
            if first:
                first = False
                continue
            identifier = row[0]
            name = row[1]
            disease_name = row[2]
            hexpr = row[3]
            mexpr = row[4]
            lexpr =  row[5]
            ndetected = row[6]
            if isinstance(row[7],str):
                row[7] = float(row[7].replace('e','E'))
            if isinstance(row[8],str):
                row[8] = float(row[8].replace('e','E'))
            if isinstance(row[9],str):
                row[9] = float(row[9].replace('e','E'))
            if isinstance(row[10],str):
                row[10] = float(row[10].replace('e','E'))
            
            uprog_pos = row[8]
            prog_pos = row[7] if row[7] != 0 else uprog_pos 
            uprog_neg = row[10]
            prog_neg = row[9] if row[9] != 0 else uprog_neg 
            
            linkout = url.replace("GENECODE",identifier).replace("GENENAME",name).replace("DISEASE", disease_name.replace(' ', '+'))
            
            if identifier in protein_mapping or name in protein_mapping:
                if identifier in protein_mapping:
                    identifier = protein_mapping[identifier]
                else:
                    identifier = protein_mapping[name]
                if disease_name in disease_mapping:
                    disease_id = disease_mapping[disease_name]
                    pathology[("protein", "detected_in_pathology_sample")].add((identifier, disease_id, "DETECTED_IN_PATHOLOGY_SAMPLE", hexpr, mexpr, lexpr, ndetected, prog_pos, prog_neg, linkout, "Human Protein Atlas pathology"))
                    
    return pathology

