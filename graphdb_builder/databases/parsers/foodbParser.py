import os.path
import tarfile
from collections import defaultdict
import ckg_utils
from graphdb_builder import mapping as mp, builder_utils
import pandas as pd
import numpy as np

###################
#       FooDB     # 
###################
def parser(databases_directory, download=True):
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"FooDB")
    builder_utils.checkDirectory(directory)
    cwd = os.path.abspath(os.path.dirname(__file__)) 
    config = ckg_utils.get_configuration(os.path.join(cwd, '../config/foodbConfig.yml'))

    database_url = config['database_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    tar_fileName = os.path.join(directory, database_url.split('/')[-1])
    tar_dir = database_url.split('/')[-1].split('.')[0]
    if download:
        builder_utils.downloadDB(database_url, directory)
    contents = {}
    food = set()
    compounds = {}
    tf = tarfile.open(tar_fileName, 'r:gz')
    tf.extractall(path=directory)
    tf.close()
    for file_name in config['files']:
        path = os.path.join(directory,os.path.join(tar_dir, file_name))
        with open(path, 'r', encoding="utf-8", errors='replace') as f:
            if file_name == "contents.csv":
                contents = parseContents(f)
            elif file_name == "foods.csv":
                food, mapping = parseFood(f)
            elif file_name == "compounds.csv":
                compounds = parseCompounds(f)
    for food_id, compound_id in contents:
        if compound_id in compounds:
            compound_code = compounds[compound_id].replace("HMDB","HMDB00")
            relationships[("food", "has_content")].add((food_id, compound_code, "HAS_CONTENT") + contents[(food_id, compound_id)])
    mp.reset_mapping(entity="Food")
    with open(os.path.join(directory, "mapping.tsv"), 'w') as out:
        for food_id in mapping:
            for alias in mapping[food_id]:
                out.write(str(food_id)+"\t"+str(alias)+"\n")
    
    mp.mark_complete_mapping(entity="Food")
    
    return food, relationships, entities_header, relationships_headers

def parseContents(fhandler):
    contents = {}
    first = True
    for line in fhandler:
        if first:
            first = False
            continue
        data = line.rstrip("\r\n").split(",")
        if len(data) == 24:
            compound_id = data[0]
            food_id = int(data[3])
            min_cont = float(data[11]) if data[11] != 'NULL' else 0 
            max_cont = float(data[12]) if data[12] != 'NULL' else 0
            units = data[13].replace('"', '')
            average = float(data[23]) if data[23] != 'NULL' else 0
            contents[(food_id, compound_id)] = (min_cont, max_cont, average, units, "FooDB")
    return contents


def parseFood(fhandler):
    food = set()
    mapping = defaultdict(set)
    df = pd.read_csv(fhandler, sep=',', header=None, error_bad_lines=False, low_memory=False)
    first = True
    for index, row in df.iterrows():
        if first:
            first = False
            continue
        food_id = int(row[0])
        name= row[1]
        sci_name = row[2]
        description = row[3]
        group = row[11]
        subgroup = row[12]
        food.add((food_id, name, sci_name, description, group, subgroup, "FooDB"))
        mapping[food_id].add(name)
        mapping[food_id].add(sci_name)

    return food, mapping


def parseCompounds(fhandler):
    compounds = {}
    first = True
    df = pd.read_csv(fhandler, sep=',', header=None, error_bad_lines=False, low_memory=False)
    first = True
    for index, row in df.iterrows():
        if first:
            first = False
            continue
        compound_id = row[0]
        mapped_code = row[44]
        if str(mapped_code) != 'nan':
            compounds[compound_id] = mapped_code
    return compounds
            
if __name__ == "__main__":
    pass

