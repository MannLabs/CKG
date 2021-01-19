import os.path
import gzip
import re
from ckg.graphdb_builder import mapping as mp, builder_utils


def parser(databases_directory, drug_source, download=True):
    config = builder_utils.get_config(config_name="siderConfig.yml", data_type='databases')
    url = config['SIDER_url']
    header = config['header']

    output_file = 'sider_has_side_effect.tsv'

    drugmapping = mp.getSTRINGMapping(source=drug_source, download=download, db="STITCH")
    phenotypemapping = mp.getMappingFromOntology(ontology="Phenotype", source=config['SIDER_source'])

    relationships = set()
    directory = os.path.join(databases_directory, "SIDER")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
    associations = gzip.open(fileName, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        drug = re.sub(r'CID\d', 'CIDm', data[0])
        se = data[2]
        evidence_from = str(data[3])
        #freq = data[4]
        #lower_bound = data[5]
        #upper_bound = data[6]
        if se.lower() in phenotypemapping and drug in drugmapping:
            for d in drugmapping[drug]:
                p = phenotypemapping[se.lower()]
                relationships.add((d, p, "HAS_SIDE_EFFECT", "SIDER", se, evidence_from))
    associations.close()

    return (relationships, header, output_file, drugmapping, phenotypemapping)


def parserIndications(databases_directory, drugMapping, phenotypeMapping, download=True):
    config = builder_utils.get_config(config_name="siderConfig.yml", data_type='databases')
    url = config['SIDER_indications']
    header = config['indications_header']
    output_file = 'sider_is_indicated_for.tsv'

    relationships = set()
    directory = os.path.join(databases_directory, "SIDER")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
    associations = gzip.open(fileName, 'r')
    for line in associations:
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        drug = re.sub(r'CID\d', 'CIDm', data[0])
        se = data[1]
        evidence = data[2]
        if se.lower() in phenotypeMapping and drug in drugMapping:
            for d in drugMapping[drug]:
                p = phenotypeMapping[se.lower()]
                relationships.add((d, p, "IS_INDICATED_FOR", evidence, "SIDER", se))

    associations.close()

    builder_utils.remove_directory(directory)

    return (relationships, header, output_file)
