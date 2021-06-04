import os.path
import tarfile
from collections import defaultdict
from ckg.graphdb_builder import mapping as mp, builder_utils
import pandas as pd

###################
#       FooDB     #
###################


def parser(databases_directory, download=True):
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory, "FooDB")
    builder_utils.checkDirectory(directory)
    config = builder_utils.get_config(config_name="foodbConfig.yml", data_type='databases')

    database_url = config['database_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    tar_fileName = os.path.join(directory, database_url.split('/')[-1])
    if download:
        builder_utils.downloadDB(database_url, directory)

    contents = {}
    food = set()
    compounds = {}
    try:
        tf = tarfile.open(tar_fileName, 'r')
        file_content = tf.getnames()
        tar_dir = file_content[1]
        tf.extractall(path=directory)
        tf.close()
        for file_name in config['files']:
            path = os.path.join(directory, os.path.join(tar_dir, file_name))
            with open(path, 'r', encoding="utf-8", errors='replace') as f:
                if file_name == "Content.csv":
                    contents = parseContents(f)
                elif file_name == "Food.csv":
                    food, mapping = parseFood(f)
                elif file_name == "Compound.csv":
                    compounds = parseCompounds(f)
        for food_id, compound_id in contents:
            if compound_id in compounds:
                compound_code = compounds[compound_id].replace("HMDB", "HMDB00")
                relationships[("food", "has_content")].add((food_id, compound_code, "HAS_CONTENT") + contents[(food_id, compound_id)])
        mp.reset_mapping(entity="Food")
        with open(os.path.join(directory, "mapping.tsv"), 'w', encoding='utf-8') as out:
            for food_id in mapping:
                for alias in mapping[food_id]:
                    out.write(str(food_id)+"\t"+str(alias)+"\n")

        mp.mark_complete_mapping(entity="Food")
    except tarfile.ReadError as err:
        raise Exception("Error importing database FooDB.\n {}".format(err))

    builder_utils.remove_directory(directory)

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
    df = pd.read_csv(fhandler, sep=',', header=None, error_bad_lines=False, low_memory=False, encoding="utf-8")
    first = True
    for index, row in df.iterrows():
        if first:
            first = False
            continue
        food_id = row[22]
        name = row[1]
        sci_name = row[2]
        description = str(row[3]).replace('"', '')
        group = row[11]
        subgroup = row[12]
        food.add((food_id, name, sci_name, description, group, subgroup, "FooDB"))
        mapping[food_id].add(name)
        mapping[food_id].add(sci_name)

    return food, mapping


def parseCompounds(fhandler):
    compounds = {}
    first = True
    df = pd.read_csv(fhandler, sep=',', header=None, error_bad_lines=False, low_memory=False, encoding="utf-8")
    first = True
    for index, row in df.iterrows():
        if first:
            first = False
            continue
        print(row)
        print(row.shape)
        compound_id = row[0]
        mapped_code = row[44]
        if str(mapped_code) != 'nan':
            compounds[compound_id] = mapped_code
    return compounds


if __name__ == "__main__":
    pass
