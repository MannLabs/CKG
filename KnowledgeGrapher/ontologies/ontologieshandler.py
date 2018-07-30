import urllib
import ontologies_config as config
from KnowledgeGrapher import mapping as mp
from parsers import *
import os.path
from collections import defaultdict
import pandas as pd
import csv
import obonet
import re

#########################
# General functionality # 
#########################

def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]

def downloadOntology(ontologyURL):
    pass

def getOntology(ontologyType):
    pass


def trimSNOMEDTree(relationships, filters):
    filteredRelationships = set()
    toRemove = set()
    for term in relationships:
        intersect = relationships[term].intersection(filters)
        if len(intersect) > 1:
            for destinationID in relationships[term]:
                if term not in toRemove:
                    filteredRelationships.add((term,destinationID, "HAS_PARENT"))
        else:
            toRemove.add(term)
    
    return filteredRelationships, toRemove



############################
# Calling the right parser # 
############################
def parseOntology(ontology):
    ontologyDirectory = config.ontologiesDirectory
    ontologyFiles = []
    if ontology in config.ontology_types:
        otype = config.ontology_types[ontology]
        if otype in config.files:
            ofiles = config.files[otype]
            ###Check SNOMED-CT files exist
            for f in ofiles:
                if os.path.isfile(os.path.join(ontologyDirectory, f)):
                    ontologyFiles.append(os.path.join(ontologyDirectory, f))
        filters = None
        if otype in config.parser_filters:
            filters = config.parser_filters[otype]
    print(ontology)
    if ontology == "SNOMED-CT":
        ontologyData = snomedParser.parser(ontologyFiles, filters)
    if ontology == "ICD":
        ontologyData = icdParser.parser(ontologyFiles)
    if ontology in ["DO","BTO","PSI-MOD", "HPO", "GO","PSI-MS"]:
        ontologyData = oboParser.parser(ontology, ontologyFiles)
        mp.buildMappingFromOBO(ontologyFiles.pop(), ontology)
   
    return ontologyData
    
#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory, ontologies=None):
    entities = config.ontologies
    if ontologies is not None:
        for ontology in ontologies:
            entities = {ontology:ontologies[ontology]}
    for entity in entities:
        ontology = config.ontologies[entity]
        if ontology in config.ontology_types:
            ontologyType = config.ontology_types[ontology]
        
        relationships_outputfile = os.path.join(importDirectory, entity.capitalize()+"_has_parent.csv")
        terms, relationships, definitions = parseOntology(ontology)
        for namespace in terms:
            entity_outputfile = os.path.join(importDirectory, namespace.capitalize()+".csv")
            with open(entity_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'name', 'description', 'type', 'synonyms'])
                for term in terms[namespace]:
                    writer.writerow([term, entity, list(terms[namespace][term])[0], definitions[term], ontologyType, ",".join(terms[namespace][term])])
        relationshipsDf = pd.DataFrame(list(relationships))
        print(relationshipsDf.head())
        relationshipsDf.columns = ['START_ID', 'END_ID', 'TYPE']

        relationshipsDf.to_csv(path_or_buf=relationships_outputfile, 
                                header=True, index=False, quotechar='"', 
                                quoting=csv.QUOTE_ALL,
                                line_terminator='\n', escapechar='\\')

if __name__ == "__main__":
    pass
