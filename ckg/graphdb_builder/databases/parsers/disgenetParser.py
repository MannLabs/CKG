import os.path
import gzip
from collections import defaultdict
from ckg.graphdb_builder import builder_utils

#########################
#       DisGeNet        #
#########################
def parser(databases_directory, download=True):
    relationships = defaultdict(set)
    config = builder_utils.get_config(config_name="disgenetConfig.yml", data_type='databases')

    files = config['disgenet_files']
    mapping_files = config['disgenet_mapping_files']
    url = config['disgenet_url']
    directory = os.path.join(databases_directory, "disgenet")
    builder_utils.checkDirectory(directory)
    header = config['disgenet_header']
    output_file = 'disgenet_associated_with.tsv'

    if download:
        for f in files:
            builder_utils.downloadDB(url+files[f], directory)
        for f in mapping_files:
            builder_utils.downloadDB(url+mapping_files[f], directory)

    proteinMapping = readDisGeNetProteinMapping(config, directory)
    diseaseMapping = readDisGeNetDiseaseMapping(config, directory)
    for f in files:
        first = True
        associations = gzip.open(os.path.join(directory, files[f]), 'r')
        dtype, atype = f.split('_')
        if dtype == 'gene':
            idType = "Protein"
            scorePos = 9
        if dtype == 'variant':
            idType = "Transcript"
            scorePos = 5
        for line in associations:
            if first:
                first = False
                continue
            try:
                data = line.decode('utf-8').rstrip("\r\n").split("\t")
                geneId = str(int(data[0]))
                #disease_specificity_index =  data[2]
                #disease_pleiotropy_index = data[3]
                diseaseId = data[4]
                score = float(data[scorePos])
                pmids = data[13]
                source = data[-1]
                if geneId in proteinMapping:
                    for identifier in proteinMapping[geneId]:
                        if diseaseId in diseaseMapping:
                            for code in diseaseMapping[diseaseId]:
                                code = "DOID:"+code
                                relationships[idType].add((identifier, code, "ASSOCIATED_WITH", score, atype, "DisGeNet: "+source, pmids))
            except UnicodeDecodeError:
                continue
        associations.close()

    builder_utils.remove_directory(directory)

    return (relationships, header, output_file)


def readDisGeNetProteinMapping(config, directory):
    files = config['disgenet_mapping_files']
    first = True
    mapping = defaultdict(set)
    if "protein_mapping" in files:
        mappingFile = files["protein_mapping"]
        with gzip.open(os.path.join(directory, mappingFile), 'r') as f:
            for line in f:
                if first:
                    first = False
                    continue
                data = line.decode('utf-8').rstrip("\r\n").split("\t")
                identifier = data[0]
                intIdentifier = data[1]
                mapping[intIdentifier].add(identifier)
    return mapping


def readDisGeNetDiseaseMapping(config, directory):
    files = config['disgenet_mapping_files']
    first = True
    mapping = defaultdict(set)
    if "disease_mapping" in files:
        mappingFile = files["disease_mapping"]
        with gzip.open(os.path.join(directory, mappingFile), 'r') as f:
            for line in f:
                if first:
                    first = False
                    continue
                data = line.decode('utf-8').rstrip("\r\n").split("\t")
                identifier = data[0]
                vocabulary = data[2]
                code = data[3]
                if vocabulary == "DO":
                    mapping[identifier].add(code)
    return mapping
