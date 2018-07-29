############################################
# REFLECT ontologies - DO, BTO, STITCH, GO # 
############################################
def parser(files, filters, qtype = None):
    entity = {}
    terms = defaultdict(list)
    relationships = set()
    definitions = defaultdict()
    for f in files:
        with open(f, 'r') as fh:
            if "entities" in f:
                for line in fh:
                    data = line.rstrip("\r\n").split("\t")
                    if data[1] == str(qtype) or qtype is None:
                        entity[data[0]] = data[2]
            if "names" in f:
                for line in fh:
                    data = line.rstrip("\r\n").split("\t")
                    if data[0] in entity:
                        code = entity[data[0]]
                        term = data[1]
                        if len(data)<=2 or int(data[2]) == 1:
                            terms[code].insert(0,term)
                        elif int(data[2]) != 2:
                            terms[code].append(term)
                        definitions[code] = term
            if "groups" in f:
                for line in fh:
                    data = line.rstrip("\r\n").split("\t")
                    if data[0] in entity and data[1] in entity:
                        sourceID = entity[data[0]] #child
                        destinationID = entity[data[1]] #parent
                        relationships.add((sourceID, destinationID, "HAS_PARENT"))
            if "texts" in f:
                for line in fh:
                    data = line.rstrip("\r\n").split("\t")
                    if data[0] in entity:
                        code = entity[data[0]]
                        definition = data[1]
                        definitions[code] = definition.replace('\n', ' ').replace('"', '').replace('\\', '')

    return terms, relationships, definitions
