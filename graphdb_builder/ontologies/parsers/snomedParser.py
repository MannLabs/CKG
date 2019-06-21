from collections import defaultdict

#################################
# Clinical_variable - SNOMED-CT # 
#################################
def parser(files, filters):
    terms = {"SNOMED-CT":defaultdict(list)}
    relationships = defaultdict(set)
    definitions = defaultdict()
    inactive_terms = read_concept_file(files[0])
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
                        if conceptID not in inactive_terms:
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
                        if sourceID not in inactive_terms and destinationID not in inactive_terms:
                            relationships["SNOMED-CT"].add((sourceID, destinationID, "HAS_PARENT"))
            elif "Definition" in f:
                for line in fh:
                    if first:
                        first = False
                        continue
                    data = line.rstrip("\r\n").split("\t")
                    if int(data[2]) == 1:
                        conceptID = data[4]
                        if conceptID not in inactive_terms:
                            definition = data[7].replace('\n', ' ').replace('"', '').replace('\\', '')
                            definitions[conceptID] = definition
    
    return terms, relationships, definitions

def read_concept_file(concept_file):
    inactive_terms = set()
    first = True
    with open(concept_file, 'r') as cf:
        for line in cf:
            if first:
                first = False
                continue
            data = line.rstrip('\r\n').split('\t')
            concept = data[0]
            is_active = data[2]

            if not is_active:
                invalid_concepts.add(concept)

    return inactive_terms
