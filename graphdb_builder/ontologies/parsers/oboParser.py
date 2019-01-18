from collections import defaultdict
from graphdb_builder import builder_utils
import re
##################
# OBO ontologies # 
##################
def parser(ontology, files):
    entity = {}
    terms = defaultdict(list)
    relationships = defaultdict(set)
    definitions = defaultdict()
    for obo_file in files:
        oboGraph = builder_utils.convertOBOtoNet(obo_file)
        namespace = ontology
        for term,attr in oboGraph.nodes(data = True):
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
                    terms[namespace][term].append(re.match(r'\"(.+?)\"',s).group(1))
            if "xref" in attr:
                for s in attr["xref"]:
                    terms[namespace][term].append(s)
            if "def" in attr:
                definitions[term] = attr["def"].replace('"','')
            else:
                definitions[term] = terms[namespace][term][0]
            if "is_a" in attr:
                for isa in attr["is_a"]:
                    relationships[namespace].add((term, isa, "HAS_PARENT"))
    return terms, relationships, definitions

