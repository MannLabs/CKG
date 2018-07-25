import urllib
import ontologies_config as config
import os.path
from collections import defaultdict
import pandas as pd
import csv
import obonet
import re

#########################
# General functionality # 
#########################
def convertOBOtoNet(ontologyFile):
    graph = obonet.read_obo(ontologyFile)
    return graph

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

def buildMappingFromOBO(oboFile, ontology):
    outputDir = os.path.join(config.ontologiesDirectory, config.ontologies[ontology])
    outputFile = os.path.join(outputDir, "mapping.tsv")
    identifiers = defaultdict(list)
    with open(oboFile, 'r') as f:
        for line in f:
            if line.startswith("id:"):
                ident = ":".join(line.rstrip("\r\n").split(":")[1:])
            if line.startswith("xref:"):
                source_ref = line.rstrip("\r\n").split(":")[1:]
                if len(source_ref) == 2:
                    identifiers[ident.strip()].append((source_ref[0].strip(), source_ref[1]))
    with open(outputFile, 'w') as out:
        for ident in identifiers:
            for source, ref in identifiers[ident]:
                out.write(ident+"\t"+source+"\t"+ref+"\n")

#################################
# Clinical_variable - SNOMED-CT # 
#################################
def parseSNOMED(files, filters):
    terms = defaultdict(list)
    relationships = set()
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
                        terms[conceptID].append(term)
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
                        relationships.add((sourceID, destinationID, "HAS_PARENT"))
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

#########################
# Diagnose entity - ICD # 
#########################
def parseICD(ICDfile):
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
                while i<=order:
                    if icdCode[0:i] in terms:
                        relationships.add((icdCode, icdCode[0:i], "HAS_PARENT"))
                        i += 1
    
            relationships.add((icdCode,chapId, "HAS_PARENT"))
            relationships.add((icdCode,blockId, "HAS_PARENT"))
            relationships.add((blockId, chapId, "HAS_PARENT"))

    return terms, relationships, definitions

############################################
# REFLECT ontologies - DO, BTO, STITCH, GO # 
############################################
def parseReflectFiles(files, filters, qtype = None):
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

##################
# OBO ontologies # 
##################
def parseOBOFiles(files):
    entity = {}
    terms = defaultdict(list)
    relationships = set()
    definitions = defaultdict()
    for f in files:
        oboGraph = convertOBOtoNet(f)
        for term,attr in oboGraph.nodes(data = True):
            if "name" in attr:
                terms[term].append(attr["name"])
            else:
                terms[term].append(term)
            if "synonym" in attr:
                for s in attr["synonym"]:
                    terms[term].append(re.match(r'\"(.+?)\"',s).group(1))
            if "def" in attr:
                definitions[term] = attr["def"].replace('"','')
            else:
                definitions[term] = terms[term][0]
            if "is_a" in attr:
                for isa in attr["is_a"]:
                    relationships.add((term, isa, "HAS_PARENT"))
    return terms, relationships, definitions

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
        ontologyData = parseSNOMED(ontologyFiles, filters)
    if ontology == "ICD":
        ontologyData = parseICD(ontologyFiles)
    if ontology in ["DO", "BTO", "GOBP", "GOMF", "GOCC", "STITCH"]:
        ontologyData = parseReflectFiles(ontologyFiles, filters, otype)
    if ontology in ["PSI-MOD"]:
        ontologyData = parseOBOFiles(ontologyFiles)
   
    return ontologyData
    
#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory):
    for entity in config.ontologies:
        ontology = config.ontologies[entity]
        if ontology in config.ontology_types:
            ontologyType = config.ontology_types[ontology]

        entity_outputfile = os.path.join(importDirectory, entity+".csv")
        relationships_outputfile = os.path.join(importDirectory, entity+"_has_parent.csv")
        
        terms, relationships, definitions = parseOntology(ontology)
        with open(entity_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'name', 'description', 'type', 'synonyms'])
                for term in terms:
                    writer.writerow([term, entity, list(terms[term])[0], definitions[term], ontologyType, ",".join(terms[term])])
        relationshipsDf = pd.DataFrame(list(relationships))
        print(relationshipsDf.head())
        relationshipsDf.columns = ['START_ID', 'END_ID', 'TYPE']

        relationshipsDf.to_csv(path_or_buf=relationships_outputfile, 
                                header=True, index=False, quotechar='"', 
                                quoting=csv.QUOTE_ALL,
                                line_terminator='\n', escapechar='\\')

if __name__ == "__main__":
    pass
