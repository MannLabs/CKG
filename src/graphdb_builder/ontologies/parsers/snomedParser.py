from collections import defaultdict

#################################
# Clinical_variable - SNOMED-CT # 
#################################
def parser(files, filters):
    """
    Parses and extracts relevant data from SNOMED CT database files.

    :param list files: list of files downloaded from SNOMED CT and used to generate nodes and relationships in the graph database.
    :param list filters: list of SNOMED CT Identifiers to be ignored.
    :return: Three nested dictionaries: terms, relationships between terms, and definitions of the terms.

        - terms: Dictionary where each key is a SNOMED CT Identifier (*str*) and the values are lists of names and synonyms (*list[str]*).
        - relationships: Dictionary of tuples (*str*). Each tuple contains two SNOMED CT Identifiers (source and target) and \
                        the relationship type between them.
        - definitions: Dictionary with SNOMED CT Identifiers as keys (*str*), and definition of the terms as values (*str*).
    """
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
    """
    
    :param concept_file:
    :return:
    """
    inactive_terms = set()
    first = True
    with open(concept_file, 'r') as cf:
        for line in cf:
            if first:
                first = False
                continue
            data = line.rstrip('\r\n').split('\t')
            concept = data[0]
            is_active = bool(data[2])

            if not is_active:
                inactive_terms.add(concept)

    return inactive_terms
