import os.path
import re
from collections import defaultdict
from ckg.graphdb_builder import builder_utils

#########################
#          IntAct       #
#########################
def parser(databases_directory, download=True):
    intact_dictionary = defaultdict()
    stored = set()
    relationships = set()
    config = builder_utils.get_config(config_name="intactConfig.yml", data_type='databases')
    header = config['header']
    outputfileName = "intact_interacts_with.tsv"
    regex = r"\((.*)\)"
    taxid_regex = r"\:(\d+)"
    url = config['intact_psimitab_url']
    directory = os.path.join(databases_directory, "Intact")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    with open(fileName, 'r', encoding="utf-8") as idf:
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
            methodMatch = re.search(regex, data[6])
            method = methodMatch.group(1) if methodMatch else "unknown"
            publications = data[8]
            tAmatch = re.search(taxid_regex, data[9])
            tBmatch = re.search(taxid_regex, data[10])
            taxidA = ""
            taxidB = ""
            if tAmatch and tBmatch:
                taxidA = tAmatch.group(1)
                taxidB = tBmatch.group(1)
            itypeMatch = re.search(regex, data[11])
            itype = itypeMatch.group(1) if itypeMatch else "unknown"
            sourceMatch = re.search(regex, data[12])
            source = sourceMatch.group(1) if sourceMatch else "unknown"
            score = data[14].split(":")[1]
            if builder_utils.is_number(score):
                score = float(score)
            else:
                continue
            if taxidA == "9606" and taxidB == "9606":
                if (intA, intB) in intact_dictionary:
                    intact_dictionary[(intA, intB)]['methods'].add(method)
                    intact_dictionary[(intA, intB)]['sources'].add(source)
                    intact_dictionary[(intA, intB)]['publications'].add(publications.replace('|', ','))
                    intact_dictionary[(intA, intB)]['itype'].add(itype)
                else:
                    intact_dictionary[(intA, intB)] = {'methods': set([method]), 'sources': set([source]), 'publications': set([publications]), 'itype': set([itype]), 'score': score}
    for (intA, intB) in intact_dictionary:
        if (intA, intB, intact_dictionary[(intA, intB)]["score"]) not in stored:
            relationships.add((intA, intB, "CURATED_INTERACTS_WITH", intact_dictionary[(intA, intB)]['score'], ",".join(intact_dictionary[(intA, intB)]['itype']), ",".join(intact_dictionary[(intA, intB)]['methods']), ",".join(intact_dictionary[(intA, intB)]['sources']), ",".join(intact_dictionary[(intA, intB)]['publications'])))
            stored.add((intA, intB, intact_dictionary[(intA, intB)]["score"]))

    builder_utils.remove_directory(directory)

    return (relationships, header, outputfileName)
