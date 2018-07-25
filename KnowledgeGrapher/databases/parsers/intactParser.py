#########################
#          IntAct       # 
#########################
def parseIntactDatabase(dataFile, proteins):
    intact_dictionary = defaultdict()
    intact_interactions = set()
    regex = r"\((.*)\)"
    with open(dataFile, 'r') as idf:
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
            taxidA = data[9]
            taxidB = data[10]
            itypeMatch = re.search(regex, data[11])
            itype = itypeMatch.group(1) if itypeMatch else "unknown"
            sourceMatch = re.search(regex, data[12])
            source = sourceMatch.group(1) if sourceMatch else "unknown"
            score = data[14].split(":")[1]
            if is_number(score):
                score = float(score)
            else:
                continue
            if intA in proteins and intB in proteins:
                if (intA, intB) in intact_dictionary:
                    intact_dictionary[(intA,intB)]['methods'].add(method)
                    intact_dictionary[(intA,intB)]['sources'].add(source)
                    intact_dictionary[(intA,intB)]['publications'].add(publications.replace('|',','))
                    intact_dictionary[(intA,intB)]['itype'].add(itype)
                else:
                    intact_dictionary[(intA,intB)]= {'methods': set([method]),'sources':set([source]),'publications':set([publications]), 'itype':set([itype]), 'score':score}
    for (intA, intB) in intact_dictionary:
        intact_interactions.add((intA,intB,"CURATED_INTERACTS_WITH",intact_dictionary[(intA, intB)]['score'], ",".join(intact_dictionary[(intA, intB)]['itype']), ",".join(intact_dictionary[(intA, intB)]['methods']), ",".join(intact_dictionary[(intA, intB)]['sources']), ",".join(intact_dictionary[(intA, intB)]['publications'])))

    return intact_interactions
