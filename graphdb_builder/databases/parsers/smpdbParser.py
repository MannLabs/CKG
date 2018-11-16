import os.path
import zipfile
from collections import defaultdict
from graphdb_builder.databases import databases_config as dbconfig
from graphdb_builder.databases.config import smpdbConfig as iconfig
from graphdb_builder import mapping as mp, utils
import pandas as pd

#########################
#     SMPDB database    #
#########################
def parser(download=True):
    urls = iconfig.smpdb_urls
    entities = set()
    relationships = defaultdict(set)
    entities_header = iconfig.pathway_header
    relationships_headers = iconfig.relationships_header
    directory = os.path.join(dbconfig.databasesDir, "SMPDB")
    utils.checkDirectory(directory)
    
    for dataset in urls:
        url = urls[dataset]
        file_name = url.split('/')[-1]
        if download:
            utils.downloadDB(url, directory)
        zipped_file = os.path.join(directory, file_name)
        with zipfile.ZipFile(zipped_file) as rf:
            if dataset == "pathway":
                entities = parsePathways(rf)
            elif dataset == "protein":
                relationships.update(parsePathwayProteinRelationships(rf))
            elif dataset == "metabolite":
                relationships.update(parsePathwayMetaboliteDrugRelationships(rf))
    
    return entities, relationships, entities_header, relationships_headers
        
def parsePathways(fhandler):
    entities = set()
    url = iconfig.linkout_url
    organism = 9606
    for filename in fhandler.namelist():
        if not os.path.isdir(filename):
            with fhandler.open(filename) as f:
                df = pd.read_csv(f, sep=',', error_bad_lines=False, low_memory=False)
                for index, row in df.iterrows():
                    identifier = row[0]
                    name = row[1]
                    description = row[3]
                    linkout = url.replace("PATHWAY", identifier)
                    entities.add((identifier, "Pathway", name, description, organism, linkout, "SMPDB"))
                
    return entities

def parsePathwayProteinRelationships(fhandler):
    relationships = defaultdict(set)
    loc = "unspecified"
    evidence = "unspecified"
    organism = 9606
    for filename in fhandler.namelist():
        if not os.path.isdir(filename):
            with fhandler.open(filename) as f:
                df = pd.read_csv(f, sep=',', error_bad_lines=False, low_memory=False)
                for index, row in df.iterrows():
                    identifier = row[0]
                    protein = row[3] 
                    if protein != '':
                        relationships[("protein", "annotated_to_pathway")].add((protein, identifier, "ANNOTATED_TO_PATHWAY", evidence, organism, loc, "SMPDB"))
    
    return relationships

def parsePathwayMetaboliteDrugRelationships(fhandler):
    relationships = defaultdict(set)
    loc = "unspecified"
    evidence = "unspecified"
    organism = 9606
    for filename in fhandler.namelist():
        if not os.path.isdir(filename):
            with fhandler.open(filename) as f:
                df = pd.read_csv(f, sep=',', error_bad_lines=False, low_memory=False)
                for index, row in df.iterrows():
                    identifier = row[0]
                    metabolite = row[5] 
                    drug = row[8]
                    if metabolite != '':
                        relationships[("metabolite", "annotated_to_pathway")].add((metabolite, identifier, "ANNOTATED_TO_PATHWAY", evidence, organism, loc, "SMPDB"))
                    if drug != "":
                        relationships[("drug", "annotated_to_pathway")].add((drug, identifier, "ANNOTATED_TO_PATHWAY", evidence, organism, loc, "SMPDB"))
    
    return relationships

