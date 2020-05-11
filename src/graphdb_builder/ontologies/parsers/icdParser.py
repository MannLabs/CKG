from collections import defaultdict

#########################
# Diagnose entity - ICD # 
#########################
def parser(ICDfile):
    """
    Parses and extracts relevant data from ICD-10 files (Classification of Diseases).

    :param ICDfile: list of files downloaded from the ontology database and used to generate nodes and relationships to the graph database.
    :return: Three nested dictionaries: terms, relationships between terms, and definitions of the terms.

        - terms: Dictionary where each key is an ontology identifier (*str*) and the values are lists of names and synonyms (*list[str]*).
        - relationships: Dictionary of tuples (*str*). Each tuple contains two ontology identifiers (source and target) and \
                        the relationship type between them.
        - definitions: Dictionary with ontology identifiers as keys (*str*), and definition of the terms as values (*str*).
    """
    terms = defaultdict(set)
    relationships = set()
    definitions = defaultdict()
    ICDfile = ICDfile[0]
    #version = ICDfile.split('/')[1].split('_')[1]
    first = True
    with open(ICDfile, 'r') as fh:
        for line in fh:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            icdCode = data[0]
            icdTerm = data[1]
            chapter = data[2]
            chapId = data[3]
            block = data[4]
            blockId = data[5]

            terms[icdCode].add(icdTerm)
            definitions[icdCode] = "term"
            terms[chapId].add(chapter)
            definitions[chapId] = "chapter"
            terms[blockId].add(block)
            definitions[blockId] = "block"

            if len(icdCode) > 3:
                order = len(icdCode) - 1
                i = 3
                while i <= order:
                    if icdCode[0:i] in terms:
                        relationships.add((icdCode, icdCode[0:i], "HAS_PARENT"))
                        i += 1

            relationships.add((icdCode, chapId, "HAS_PARENT"))
            relationships.add((icdCode, blockId, "HAS_PARENT"))
            relationships.add((blockId, chapId, "HAS_PARENT"))

    return terms, relationships, definitions
