import os.path
import gzip
import csv
from ckg.graphdb_builder import mapping as mp, builder_utils


def parser(databases_directory, importDirectory, drug_source=None, download=True, db="STRING"):
    config = builder_utils.get_config(config_name="stringConfig.yml", data_type='databases')
    mapping = mp.getSTRINGMapping(download=download)
    stored = set()
    relationship = None
    cutoff = config['STRING_cutoff']
    header = config['header']
    drugmapping = {}
    if db == "STITCH":
        evidences = ["experimental", "prediction", "database", "textmining", "score"]
        relationship = "COMPILED_INTERACTS_WITH"
        url = config['STITCH_url']
        outputfile = os.path.join(importDirectory, "stitch_associated_with.tsv")

        drugmapping = mp.getSTRINGMapping(source=drug_source, download=download, db=db)

    elif db == "STRING":
        evidences = ["Neighborhood in the Genome", "Gene fusions", "Co-ocurrence across genomes", "Co-expression", "Experimental/biochemical data", "Association in curated databases", "Text-mining"]
        relationship = "COMPILED_TARGETS"
        outputfile = os.path.join(importDirectory, "string_interacts_with.tsv")
        url = config['STRING_url']
    directory = os.path.join(databases_directory, db)
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, directory)

    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    first = True
    with open(outputfile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for line in associations:
            if first:
                first = False
                continue
            data = line.decode('utf-8').rstrip("\r\n").split()
            intA = data[0]
            intB = data[1]
            scores = data[2:]
            fscores = [str(float(score)/1000) for score in scores]
            if db == "STRING":
                if intA in mapping and intB in mapping and float(fscores[-1]) >= cutoff:
                    for aliasA in mapping[intA]:
                        for aliasB in mapping[intB]:
                            if (aliasA, aliasB) not in stored:
                                row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                                stored.add((aliasA, aliasB))
                                stored.add((aliasB, aliasB))
                                writer.writerow(row)
            elif db == "STITCH":
                if intA in drugmapping and intB in mapping and float(fscores[-1]) >= cutoff:
                    for aliasA in drugmapping[intA]:
                        for aliasB in mapping[intB]:
                            if (aliasA, aliasB) not in stored:
                                row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                                stored.add((aliasA, aliasB))
                                stored.add((aliasB, aliasB))
                                writer.writerow(row)
    associations.close()

    return mapping, drugmapping


def parseActions(databases_directory, importDirectory, proteinMapping, drugMapping=None, download=True, db="STRING"):
    config = builder_utils.get_config(config_name="stringConfig.yml", data_type='databases')
    url = None
    bool_dict = {'t': True, 'T': True, 'True': True, 'TRUE': True, 'f': False, 'F': False, 'False': False, 'FALSE': False}
    header = config['header_actions']
    relationship = "COMPILED_ACTS_ON"
    stored = set()
    if db == "STRING":
        url = config['STRING_actions_url']
        outputfile = os.path.join(importDirectory, "string_protein_acts_on_protein.tsv")
    elif db == "STITCH":
        url = config['STITCH_actions_url']
        outputfile = os.path.join(importDirectory, "stitch_drug_acts_on_protein.tsv")

    directory = os.path.join(databases_directory, db)
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    first = True
    with open(outputfile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for line in associations:
            if first:
                first = False
                continue
            data = line.decode('utf-8').rstrip("\r\n").split()
            intA = data[0]
            intB = data[1]
            action = data[2]
            score = float(data[-1])/1000
            directionality = bool_dict[data[-3]] if db == "STRING" else True

            if intB in proteinMapping:
                aliasesA = []
                if intA in drugMapping:
                    aliasesA = drugMapping[intA]
                elif intA in proteinMapping:
                    aliasesA = proteinMapping[intA]
                for aliasA in aliasesA:
                    for aliasB in proteinMapping[intB]:
                        if (aliasA, aliasB, action) not in stored:
                            row = (aliasA, aliasB, relationship, action, directionality, score, db)
                            writer.writerow(row)
                            stored.add((aliasA, aliasB, action))
                            stored.add((aliasB, aliasA, action))
    associations.close()
