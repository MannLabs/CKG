import os.path
import gzip
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import stringConfig as iconfig
from collections import defaultdict
from KnowledgeGrapher import utils
from KnowledgeGrapher import mapping as mp
import csv
import re

#########################
#   STRING like DBs     #
#########################
def parser(importDirectory, download = True, db="STRING"):
    mapping_url = iconfig.STRING_mapping_url
    mapping = mp.getSTRINGMapping(mapping_url, download = False)
    relationship = None
    cutoff = iconfig.STRING_cutoff
    header = iconfig.header
    drugmapping = {}
    if db == "STITCH":
        evidences = ["experimental", "prediction", "database","textmining", "score"]
        relationship = "COMPILED_INTERACTS_WITH"
        url = iconfig.STITCH_url
        outputfile = os.path.join(importDirectory, "STITCH_associated_with.csv")

        drugsource = dbconfig.sources["Drug"]
        drugmapping_url = iconfig.STITCH_mapping_url
        drugmapping = mp.getSTRINGMapping(drugmapping_url, source = drugsource, download = False, db = db)
    elif db == "STRING":
        evidences = ["experimental", "prediction", "database","textmining", "score"]
        relationship = "COMPILED_TARGETS"
        outputfile = os.path.join(importDirectory, "STRING_interacts_with.csv")
        url = iconfig.STRING_url
    directory = os.path.join(dbconfig.databasesDir, db)
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        utils.downloadDB(url, db)
    
    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    first = True
    with open(outputfile, 'w') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for line in associations:
            if first:
                first = False
                continue
            data = line.decode('utf-8').rstrip("\r\n").split()
            intA = data[0]
            intB = data[1]
            print(intA, intB)
            scores = data[2:]
            fscores = [str(float(score)/1000) for score in scores]
            if db == "STRING":
                if intA in mapping and intB in mapping and float(fscores[-1])>=cutoff:
                    for aliasA in mapping[intA]:
                        for aliasB in mapping[intB]:
                            row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                            writer.writerow(row)
            elif db == "STITCH":
                print(intA in drugmapping, intB in mapping, float(fscores[-1])>=cutoff)
                if intA in drugmapping and intB in mapping and float(fscores[-1])>=cutoff:
                    for aliasA in drugmapping[intA]:
                        print(aliasA)
                        for aliasB in mapping[intB]:
                            row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                            writer.writerow(row)
    associations.close()



def parseActions(db="STRING"):
    url = None
    actions = defaultdict(set)

    if db == "STRING":
        url = iconfig.STRING_actions_url
    if db == "STITCH":
        url = iconfig.STITCH_actions_url

    directory = os.path.join(dbconfig.databasesDir, db)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, db)
    
    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    first = True
    for line in associations:
        if first:
            first = False
            continue
        data = line.decode('utf-8').rstrip("\r\n").split()
        actions[(data[0],data[1])].add((data[2],data[3]))
