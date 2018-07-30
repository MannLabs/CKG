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
    utils.checkDirectory(directory)
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
            scores = data[2:]
            fscores = [str(float(score)/1000) for score in scores]
            if db == "STRING":
                if intA in mapping and intB in mapping and float(fscores[-1])>=cutoff:
                    for aliasA in mapping[intA]:
                        for aliasB in mapping[intB]:
                            row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                            writer.writerow(row)
            elif db == "STITCH":
                if intA in drugmapping and intB in mapping and float(fscores[-1])>=cutoff:
                    for aliasA in drugmapping[intA]:
                        for aliasB in mapping[intB]:
                            row = (aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1])
                            writer.writerow(row)
    associations.close()

    return mapping, drugmapping



def parseActions(importDirectory, proteinMapping, drugMapping = None, download = True, db="STRING"):
    url = None
    bool_dict = {'t':True, 'T':True, 'True':True, 'TRUE': True, 'f':False, 'F':False, 'False': False, 'FALSE':False}
    header = iconfig.header_actions
    relationship = "COMPILED_ACTS_ON"

    if db == "STRING":
        url = iconfig.STRING_actions_url
        outputfile = os.path.join(importDirectory, "STRING_protein_acts_on_protein.csv")
    elif db == "STITCH":
        url = iconfig.STITCH_actions_url
        outputfile = os.path.join(importDirectory, "STITCH_drug_acts_on_protein.csv")
    
    directory = os.path.join(dbconfig.databasesDir, db)
    utils.checkDirectory(directory)
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
                        row = (aliasA, aliasB, relationship, action, directionality, score, db)
                        writer.writerow(row)
    associations.close()
