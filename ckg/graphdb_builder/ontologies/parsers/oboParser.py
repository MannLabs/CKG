from collections import defaultdict
from ckg.graphdb_builder import builder_utils
import re


def parser(ontology, files):
    """
    Multiple ontology database parser.
    This function parses and extracts relevant data from: Disease Ontology, Tissues, Human Phenotype Ontology, \
    HUPO-PSI and Gene Ontology databases.

    :param str ontology: name of the ontology to be imported ('Disease', 'Tissue', 'Phenotype', 'Experiment', \
                        'Modification', 'Molecular_interactions', 'Gene_ontology')
    :param list files: list of files downloaded from an ontology and used to generate nodes and relationships in the graph database.
    :return: Three nested dictionaries: terms, relationships between terms, and definitions of the terms.

        - terms: Dictionary where each key is an ontology identifier (*str*) and the values are lists of names and synonyms (*list[str]*).
        - relationships: Dictionary of tuples (*str*). Each tuple contains two ontology identifiers (source and target) and \
                        the relationship type between them.
        - definitions: Dictionary with ontology identifiers as keys (*str*), and definition of the terms as values (*str*).
    """
    terms = defaultdict(list)
    relationships = defaultdict(set)
    definitions = defaultdict()
    for obo_file in files:
        oboGraph = builder_utils.convertOBOtoNet(obo_file)
        namespace = ontology
        for term, attr in oboGraph.nodes(data=True):
            if "namespace" in attr:
                namespace = attr["namespace"]
            if namespace not in terms:
                terms[namespace] = defaultdict(list)
            if "name" in attr:
                terms[namespace][term].append(attr["name"])
            else:
                terms[namespace][term].append(term)
            if "synonym" in attr:
                for s in attr["synonym"]:
                    terms[namespace][term].append(re.match(r'\"(.+?)\"', s).group(1))
            if "xref" in attr:
                for s in attr["xref"]:
                    terms[namespace][term].append(s)
            if "def" in attr:
                definitions[term] = attr["def"].replace('"', '')
            else:
                definitions[term] = terms[namespace][term][0]
            if "is_a" in attr:
                for isa in attr["is_a"]:
                    relationships[namespace].add((term, isa, "HAS_PARENT"))
    return terms, relationships, definitions
