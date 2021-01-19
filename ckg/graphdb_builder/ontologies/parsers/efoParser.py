from ckg.graphdb_builder import builder_utils
from collections import defaultdict
import re


def parser(ontology_files):
    terms = defaultdict(list)
    relationships = defaultdict(set)
    mappings = defaultdict(set)
    definitions = defaultdict()
    for obo_file in ontology_files:
        with open(obo_file, encoding="utf-8") as f:
            oboGraph = builder_utils.convertOBOtoNet(f)
            namespace = "EFO"
            for term, attr in oboGraph.nodes(data=True):
                if term.startswith("EFO:"):
                    if "namespace" in attr:
                        namespace = attr["namespace"]
                    if namespace not in terms:
                        terms[namespace] = defaultdict(list)
                    if "name" in attr:
                        terms[namespace][term].append(attr["name"].replace('"', ''))
                    else:
                        terms[namespace][term].append(term)
                    if "synonym" in attr:
                        for s in attr["synonym"]:
                            terms[namespace][term].append(re.match(r'\"(.+?)\"', s).group(1))
                    if "xref" in attr:
                        for s in attr["xref"]:
                            xref = None
                            if s.startswith('DOID:'):
                                xref = 'Disease'
                            elif s.startswith('SNOMEDCT:'):
                                xref = 'Clinical_variable'
                                s = s.split(':')[1]
                            elif s.startswith('HP:'):
                                xref = 'Phenotype'
                            if xref is not None:
                                mappings["Experimental_factor_maps_to_"+xref].add((term, s, "MAPS_TO"))
                    if "def" in attr:
                        definitions[term] = attr["def"].replace('"', '')
                    else:
                        definitions[term] = terms[namespace][term][0]
                    if "is_a" in attr:
                        for isa in attr["is_a"]:
                            relationships[namespace].add((term, isa, "HAS_PARENT"))

    return (terms, relationships, definitions), mappings
