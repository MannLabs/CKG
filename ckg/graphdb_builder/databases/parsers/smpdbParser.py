import os.path
import zipfile
import pandas as pd
from collections import defaultdict
from ckg.graphdb_builder import builder_utils

#########################
#     SMPDB database    #
#########################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="smpdbConfig.yml", data_type='databases')
    urls = config['smpdb_urls']
    entities = set()
    relationships = defaultdict(set)
    entities_header = config['pathway_header']
    relationships_headers = config['relationships_header']
    directory = os.path.join(databases_directory, "SMPDB")
    builder_utils.checkDirectory(directory)

    for dataset in urls:
        url = urls[dataset]
        file_name = url.split('/')[-1]
        if download:
            builder_utils.downloadDB(url, directory)
        zipped_file = os.path.join(directory, file_name)
        with zipfile.ZipFile(zipped_file) as rf:
            if dataset == "pathway":
                entities = parsePathways(config, rf)
            elif dataset == "protein":
                relationships.update(parsePathwayProteinRelationships(rf))
            elif dataset == "metabolite":
                relationships.update(parsePathwayMetaboliteDrugRelationships(rf))

    builder_utils.remove_directory(directory)

    return entities, relationships, entities_header, relationships_headers


def parsePathways(config, fhandler):
    entities = set()
    url = config['linkout_url']
    organism = 9606
    for filename in fhandler.namelist():
        if not os.path.isdir(filename):
            with fhandler.open(filename) as f:
                df = pd.read_csv(f, sep=',', error_bad_lines=False, low_memory=False)
                for index, row in df.iterrows():
                    identifier = row[0]
                    name = row[2]
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
