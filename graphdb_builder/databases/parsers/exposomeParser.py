import os.path
import zipfile
from graphdb_builder.databases.config import exposomeConfig as iconfig
from graphdb_builder import mapping as mp, builder_utils
from collections import defaultdict
import pandas as pd
import re

###############################
#       Exposome Explorer     # 
###############################
def parser(databases_directory, download=True):
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"ExposomeExplorer")
    builder_utils.checkDirectory(directory)
    database_urls = iconfig.database_urls 
    relationships_header = iconfig.relationships_header
    mapping = mp.getMappingForEntity("Food")
    correlations = {}
    for url in database_urls:
        zipped_fileName = os.path.join(directory, url.split('/')[-1])
        file_name = '.'.join(url.split('/')[-1].split('.')[0:2])
        if download:
            builder_utils.downloadDB(url, directory)

        with zipfile.ZipFile(zipped_fileName) as z:
            if file_name == "biomarkers.csv":
                biomarkers = parseBiomarkersFile(z, file_name)
            elif file_name == "correlation_values.csv":
                correlations = parseCorrelationsFile(z, file_name, biomarkers, mapping)

    return correlations, relationships_header
        

def parseBiomarkersFile(fhandler, file_name):
    biomarkers = {}
    first = True
    with fhandler.open(file_name) as f:
        df = pd.read_csv(f, sep=',', header=None, error_bad_lines=False, low_memory=False)
        first = True
        for index, row in df.iterrows():
            if first:
                first = False
                continue
            identifier = row[0]
            metabolite = row[9]
            if metabolite != '':
                biomarkers[identifier] = metabolite

    return biomarkers


def parseCorrelationsFile(fhandler, file_name, biomarkers, mapping):
    correlations = defaultdict(set)
    first = True
    with fhandler.open(file_name) as f:
        df = pd.read_csv(f, sep=',', header=None, error_bad_lines=False, low_memory=False)
        first = True
        for index, row in df.iterrows():
            if first:
                first = False
                continue
            biomarker = row[0]
            food_name = row[10]
            intake_median = row[15]
            intake_units = row[16]
            biosample = row[19]
            method = row[20]
            corr_method = row[29]
            corr = float(row[30])
            ci_low = row[31]
            ci_high = row[32]
            pvalue = row[33]
            significant = row[34]
            publication = row[38]

            if significant in ["Yes","yes","YES", "Y"]:
                if food_name in mapping:
                    food_id = mapping[food_name]
                    if biomarker in biomarkers:
                        biomarker_id = biomarkers[biomarker]
                        correlations[("food", "correlated_with_metabolite")].add((food_id, biomarker_id, "CORRELATED_WITH_METABOLITE",intake_median, intake_units, biosample, method, corr, ci_low, ci_high, pvalue, significant, publication, "Exposome Explorer" ))
    
    
    return correlations
    
if __name__ == "__main__":
    pass

