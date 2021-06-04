import os.path
import pandas as pd
import zipfile
from collections import defaultdict
from ckg.graphdb_builder import mapping as mp, builder_utils

##########################################
#   Human Protein Atlas (pathology)      #
##########################################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="hpaConfig.yml", data_type='databases')
    url = config['hpa_pathology_url']
    disease_mapping = mp.getMappingFromOntology(ontology="Disease", source=None)
    protein_mapping = mp.getMultipleMappingForEntity("Protein")
    directory = os.path.join(databases_directory, "HPA")
    builder_utils.checkDirectory(directory)
    compressed_fileName = os.path.join(directory, url.split('/')[-1])
    file_name = '.'.join(url.split('/')[-1].split('.')[0:2])
    relationships_headers = config['relationships_headers']

    if download:
        builder_utils.downloadDB(url, directory)

    with zipfile.ZipFile(compressed_fileName) as z:
        if file_name == "pathology.tsv":
            pathology = parsePathologyFile(config, z, file_name, protein_mapping, disease_mapping)

    builder_utils.remove_directory(directory)

    return (pathology, relationships_headers)


def parsePathologyFile(config, fhandler, file_name, protein_mapping, disease_mapping):
    url = config['linkout_url']
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
                    identifiers = protein_mapping[identifier]
                else:
                    identifiers = protein_mapping[name]
                for identifier in identifiers:
                    if disease_name in disease_mapping:
                        disease_id = disease_mapping[disease_name]
                        pathology[("protein", "detected_in_pathology_sample")].add((identifier, disease_id, "DETECTED_IN_PATHOLOGY_SAMPLE", hexpr, mexpr, lexpr, ndetected, prog_pos, prog_neg, linkout, "Human Protein Atlas pathology"))

    return pathology
