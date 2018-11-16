from collections import defaultdict

#################################
# Clinical_variable - SNOMED-CT # 
#################################
def parser(files, filters):
    terms = {"SNOMED-CT":defaultdict(list)}
    relationships = defaultdict(set)
    definitions = defaultdict()
    for f in files:
        first = True
        with open(f, 'r') as fh:
            if "Description" in f:
                for line in fh:
                    if first:
                        first = False
                        continue
                    data = line.rstrip("\r\n").split("\t")
                    if int(data[2]) == 1:
                        conceptID = data[4]
                        term = data[7]
                        terms["SNOMED-CT"][conceptID].append(term)
                        definitions[conceptID] = term
            elif "Relationship" in f:
                for line in fh:
                    if first:
                        first = False
                        continue
                    data = line.rstrip("\r\n").split("\t")
                    if int(data[2]) == 1:
                        sourceID = data[4] #child
                        destinationID = data[5] #parent
                        relationships["SNOMED-CT"].add((sourceID, destinationID, "HAS_PARENT"))
            elif "Definition" in f:
                for line in fh:
                    if first:
                        first = False
                        continue
                    data = line.rstrip("\r\n").split("\t")
                    if int(data[2]) == 1:
                        conceptID = data[4]
                        definition = data[7].replace('\n', ' ').replace('"', '').replace('\\', '')

                        definitions[conceptID] = definition
    #for f in filters:
    #    relationships[f].add(f)

    #relationships, toRemove = trimSNOMEDTree(relationships, filters)
    
    #entries_to_remove(toRemove, terms)
    
    return terms, relationships, definitions
