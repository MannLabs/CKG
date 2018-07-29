import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import intactConfig as iconfig
from collections import defaultdict
from KnowledgeGrapher import utils
import re

#########################
#          IntAct       # 
#########################
def parser(download = False):
    intact_dictionary = defaultdict()
    relationships = set()
    header = iconfig.header
    outputfileName = "INTACT_interacts_with.csv"
    regex = r"\((.*)\)"
    url = iconfig.intact_psimitab_url
    directory = os.path.join(dbconfig.databasesDir,"Intact")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "Intact")

    with open(fileName, 'r') as idf:
        first = True
        for line in idf:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            intA = data[0].split(":")[1]
            intB = data[1].split(':')
            if len(intB)> 1:
                intB = intB[1]
            else:
                continue
            print("1!!!")
            methodMatch = re.search(regex, data[6])
            method = methodMatch.group(1) if methodMatch else "unknown"
            publications = data[8]
            taxidA = data[9]
            taxidB = data[10]
            itypeMatch = re.search(regex, data[11])
            itype = itypeMatch.group(1) if itypeMatch else "unknown"
            sourceMatch = re.search(regex, data[12])
            source = sourceMatch.group(1) if sourceMatch else "unknown"
            score = data[14].split(":")[1]
            if utils.is_number(score):
                score = float(score)
            else:
                continue
            print("2!!!!!!!!!")
            if taxidA == "9606" and taxidB == "9606":
                print("3!!!!!!")
                if (intA, intB) in intact_dictionary:
                    intact_dictionary[(intA,intB)]['methods'].add(method)
                    intact_dictionary[(intA,intB)]['sources'].add(source)
                    intact_dictionary[(intA,intB)]['publications'].add(publications.replace('|',','))
                    intact_dictionary[(intA,intB)]['itype'].add(itype)
                else:
                    intact_dictionary[(intA,intB)]= {'methods': set([method]),'sources':set([source]),'publications':set([publications]), 'itype':set([itype]), 'score':score}
    for (intA, intB) in intact_dictionary:
        relationships.add((intA,intB,"CURATED_INTERACTS_WITH",intact_dictionary[(intA, intB)]['score'], ",".join(intact_dictionary[(intA, intB)]['itype']), ",".join(intact_dictionary[(intA, intB)]['methods']), ",".join(intact_dictionary[(intA, intB)]['sources']), ",".join(intact_dictionary[(intA, intB)]['publications'])))
    print(relationships)    
    return (relationships, header, outputfileName)
