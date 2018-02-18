import os.path
import gzip
import databases_config as config
import ontologies_config as onto_config
from collections import defaultdict
import mapping as mp
import csv
import pandas as pd
import re

#########################
# General functionality # 
#########################

def getMapping():
    mapping = mp.generateMappingFromReflect()

    return mapping

def downloadDB(databaseURL, extraFolder =""):
    import urllib
    directory = os.path.join(config.databasesDir,extraFolder)
    fileName = databaseURL.split('/')[-1]

    requestedFile = urllib.URLopener()
    requestedFile.retrieve(databaseURL, os.path.join(directory, fileName))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
#########################
#   PathwayCommons      # 
#########################
def parsePathwayCommons(download = True):
    url = config.pathwayCommons_pathways_url
    entities = set()
    relationships = set()
    directory = os.path.join(config.databasesDir, "PathwayCommons")
    fileName = url.split('/')[-1]
    if download:
        downloadDB(url, "PathwayCommons")
    
    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    for line in associations:
        data = line.rstrip("\r\n").split("\t")
        linkout = data[0]
        code = data[0].split("/")[-1]
        ptw_dict = dict([item.split(": ")[0],":".join(item.split(": ")[1:])] for item in data[1].split("; "))
        proteins = data[2:]
        if "organism" in ptw_dict and ptw_dict["organism"] == "9606":
            name = ptw_dict["name"]
            source = ptw_dict["datasource"]
        else:
            continue
        
        entities.add((code, "Pathway", name, source, linkout))
        for protein in proteins:
            relationships.add((protein, code, "MEMBER_OF_PATHWAY", "PathwayCommons: "+source))

    associations.close()
    
    return entities, relationships

############################################
#   The Drug Gene Interaction Database     # 
############################################
def parseDGIdb(download = True, mapping = {}):
    url = config.DGIdb_url
    relationships = set()
    directory = os.path.join(config.databasesDir,"DGIdb")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        downloadDB(url, "DGIdb")
    with open(fileName, 'r') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            gene = data[0]
            source = data[3]
            interactionType = data[4]
            drug = data[8]
            if drug == "":
                drug = data[7] 
                if drug == "" and data[6] != "":
                    drug = data[6]
                else:
                    continue
            if drug in mapping:
                drug = mapping[drug]
            relationships.add((drug, gene, "TARGETS", interactionType, "DGIdb: "+source))

    return relationships

#########################
#   OncoKB database     #
#########################
def parseOncoKB(download = False, mapping = {}):
    url = config.OncoKB_url
    relationships = set()
    directory = os.path.join(config.databasesDir,"OncoKB")
    fileName = os.path.join(directory,url.split('/')[-1])
    if download:
        downloadDB(url, "OncoKB")
    
    with open(fileName, 'r') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            isoform = data[1]
            gene = data[3]
            variant = gene+"_"+data[4]
            disease = data[5]
            level = data[6]
            drugs = data[7].split(',')
            pubmed_ids = data[8].split(',')
            
            for drug in drugs:
                if drug in mapping:
                    drug = mapping[drug]
                if disease in mapping:
                    disease = mapping[disease]
                relationships.add((drug, gene, "TARGETS", "", "OncoKB"))
                #relationships[drug].add((drug, variant, "TARGETS", "", "OncoKB"))
                #relationships[variant].add((variant, disease, "ASSOCIATED_WITH", "OncoKB"))   
    return relationships

#########################
#   PhosphositePlus     # 
#########################
def parsePhosphoSitePlus(download = True):
    pass

#########################
#       DisGeNet        # 
#########################
def parseDisGeNetDatabase(download = True):
    relationships = defaultdict(set)
    files = config.disgenet_files
    url = config.disgenet_url
    directory = os.path.join(config.databasesDir,"disgenet")

    if download:
        for f in files:
            downloadDB(url+files[f], "disgenet")

    proteinMapping = readDisGeNetProteinMapping() 
    diseaseMapping, diseaseSynonyms = readDisGeNetDiseaseMapping()
    for f in files:
        first = True
        associations = gzip.open(os.path.join(directory,files[f]), 'r')
        dtype, atype = f.split('_') 
        if dtype == 'gene':
            idType = "Protein"
            scorePos = 7
        if dtype == 'variant':
            idType = "Transcript"
            scorePos = 5
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            geneId = data[0]
            diseaseId = data[2]
            score = float(data[4])
            pmids = data[5]
            source = data[scorePos]
            if geneId in proteinMapping:
                for identifier in proteinMapping[geneId]:
                    if diseaseId in diseaseMapping:
                        for code in diseaseMapping[diseaseId]:
                            code = "DOID:"+code
                            relationships[idType].add((identifier, code,"ASSOCIATED_WITH", score, atype, "DisGeNet: "+source, pmids))
        associations.close()
    return relationships
    
def readDisGeNetProteinMapping():
    files = config.disgenet_mapping_files
    directory = os.path.join(config.databasesDir,"disgenet")
    
    first = True
    mapping = defaultdict(set)
    if "protein_mapping" in files:
        mappingFile = files["protein_mapping"]
        f = gzip.open(os.path.join(directory,mappingFile), 'r')
        for line in f:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            identifier = data[0]
            intIdentifier = data[1]
            mapping[intIdentifier].add(identifier)
        f.close()
    return mapping

def readDisGeNetDiseaseMapping():
    files = config.disgenet_mapping_files
    directory =  os.path.join(config.databasesDir,"disgenet")
    first = True
    mapping = defaultdict(set)
    synonyms = defaultdict(set)
    if "disease_mapping" in files:
        mappingFile = files["disease_mapping"]
        f = gzip.open(os.path.join(directory,mappingFile), 'r')
        for line in f:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            identifier = data[0]
            vocabulary = data[2]
            code = data[3]
            if vocabulary == onto_config.ontologies["Disease"]:
                mapping[identifier].add(code)
            else:
                synonyms[identifier].add(code)
        f.close()
    return mapping, synonyms

#########################
#       UniProt         # 
#########################
def parseUniProtDatabase(dataFile):
    proteins = {}
    relationships = defaultdict(set)

    fields = config.uniprot_ids
    synonymFields = config.uniprot_synonyms
    protein_relationships = config.uniprot_protein_relationships
    identifier = None
    with open(dataFile, 'r') as uf:
        for line in uf:
            data = line.rstrip("\r\n").split("\t")
            iid = data[0]
            field = data[1]
            alias = data[2]
            
            if iid not in proteins:
                if identifier is not None:
                    prot_info["synonyms"] = synonyms
                    proteins[identifier] = prot_info
                identifier = iid
                proteins[identifier] = {}
                prot_info = {}
                synonyms = []
            if field in fields:
                if field in synonymFields:
                    prot_info[field] = alias
                    synonyms.append(alias)
                if field in protein_relationships:
                    relationships[protein_relationships[field]].add((iid, alias, protein_relationships[field][1], "UniProt"))
    
    return proteins, relationships

def addUniProtTexts(textsFile, proteins):
    with open(textsFile, 'r') as tf:
        for line in tf:
            data = line.rstrip("\r\n").split("\t")
            protein = data[0]
            name = data[1]
            function = data[3]
            
            if protein in proteins:
                proteins[protein].update({"description":function})

#########################################
#          HUGO Gene Nomenclature       # 
#########################################
def parseHGNCDatabase(download = True):
    url = config.hgnc_url
    entities = set()
    relationships = set()
    directory = os.path.join(config.databasesDir,"HGNC")
    fileName = os.path.join(directory, url.split('/')[-1])
    
    if download:
        downloadDB(url, "HGNC")
    
    with open(fileName, 'r') as df:
        first = True
        for line in df:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            geneSymbol = data[1]
            geneName = data[2]
            status = data[5]
            geneFamily = data[12]
            synonyms = data[18:23]
            transcript = data[23]
            if status != "Approved":
                continue

            entities.add((geneSymbol, geneName, geneFamily, ",".join(synonyms)))
            relationships.add((geneSymbol, transcript, "TRANSCRIBED_INTO"))

    return entities, relationships

#########################
#          RefSeq       # 
#########################
def parseRefSeqDatabase(download = False):
    url = config.refseq_url
    entities = defaultdict(set)
    relationships = defaultdict(set)
    directory = os.path.join(config.databasesDir,"RefSeq")
    fileName = os.path.join(directory, url.split('/')[-1])
    
    if download:
        downloadDB(url, "RefSeq")

    df = gzip.open(fileName, 'r')
    first = True
    for line in df:
        if first:
            first = False
            continue
        data = line.rstrip("\r\n").split("\t")
        tclass = data[1]
        assembly = data[2]
        chrom = data[5]
        geneAcc = data[6]
        start = data[7]
        end = data[8]
        strand = data[9]
        protAcc = data[10]
        name = data[13]
        symbol = data[14]
        
        if protAcc != "":
            entities["Transcript"].add((protAcc, "Transcript", name, tclass, assembly))
            if chrom != "":
                entities["Chromosome"].add((chrom, "Chromosome", chrom))
                relationships["LOCATED_IN"].add((protAcc, chrom, "LOCATED_IN", start, end, strand, "RefSeq"))
            if symbol != "":
                relationships["TRANSCRIBED_INTO"].add((symbol, protAcc, "TRANSCRIBED_INTO", "RefSeq"))
        elif geneAcc != "":
            entities["Transcript"].add((geneAcc, "Transcript", name, tclass, assembly))
            if chrom != "":
                entities["Chromosome"].add((chrom, "Chromosome", chrom))
                relationships["LOCATED_IN"].add((protAcc, chrom, "LOCATED_IN", start, end, strand, "RefSeq"))
    df.close()

    return entities, relationships

#########################
#          IntAct       # 
#########################
def parseIntactDatabase(dataFile, proteins):
    intact_interactions = set()
    regex = r"\((.*)\)"
    with open(dataFile, 'r') as idf:
        first = True
        for line in idf:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            intA = data[0].split(":")[1]
            intB = data[1].split(':')
            if len(intB)> 1:
                intB = intB[1]
            else:
                continue
            methodMatch = re.search(regex, data[6])
            method = methodMatch.group(1) if methodMatch else "unknown"
            publications = data[8]
            taxidA = data[9]
            taxidB = data[10]
            itypeMatch = re.search(regex, data[11])
            itype = itypeMatch.group(1) if itypeMatch else "unknown"
            sourceMatch = re.search(regex, data[12])
            source = sourceMatch.group(1) if sourceMatch else "unknown"
            score = data[14].split(":")[1]
            if is_number(score):
                score = float(score)
            else:
                continue

            if intA in proteins and intB in proteins:
                intact_interactions.add((intA,intB,"INTACT_INTERACTS_WITH",score, itype, method, source, publications))
    
    return intact_interactions


#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory):
    mapping = getMapping()
    databases = config.databases
    for database in databases:
        print database
        if database.lower() == "hgnc":
            #HGNC
            genes, relationships = parseHGNCDatabase()
            genes_outputfile = os.path.join(importDirectory, "Gene.csv")

            with open(genes_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'name', 'family', 'synonyms', 'taxid'])
                taxid = 9606
                for symbol, name, family, synonyms in genes:
                    writer.writerow([symbol, "Gene", name, family, synonyms, taxid])
        if database.lower() == "refseq":
            #RefSeq
            headers = config.headerEntities
            entities, relationships = parseRefSeqDatabase()
            for entity in entities:
                header = headers[entity]
                outputfile = os.path.join(importDirectory, entity+".csv")
                with open(outputfile, 'w') as csvfile:
                    writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(header)
                    taxid = 9606
                    for item in entities[entity]:
                        row = list(item)
                        row.append(taxid)
                        writer.writerow(row)
            for rel in relationships:
                header = headers[rel]
                outputfile = os.path.join(importDirectory, "refseq_"+rel.lower()+".csv")
                df = pd.DataFrame(list(relationships[rel]))
                df.columns = header
                df.to_csv(path_or_buf=outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')
        if database.lower() == "uniprot":
            #UniProt
            uniprot_id_file = config.uniprot_id_file
            uniprot_texts_file = config.uniprot_text_file

            proteins, relationships = parseUniProtDatabase(uniprot_id_file)
            addUniProtTexts(uniprot_texts_file, proteins)

            proteins_outputfile = os.path.join(importDirectory, "Protein.csv")

            with open(proteins_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'accession','name', 'synonyms', 'description', 'taxid'])
                for protein in proteins:
                    accession = ""
                    name = ""
                    synonyms = []
                    taxid = 9606
                    description = ""
                    if "UniProtKB-ID" in proteins[protein]:
                        accession  = proteins[protein]["UniProtKB-ID"]
                    if "Gene_Name" in proteins[protein]:
                        name = proteins[protein]["Gene_Name"]
                    if "synonyms" in proteins[protein]:
                        synonyms = proteins[protein]["synonyms"]
                    if "NCBI_TaxID" in proteins[protein]:
                        taxid = int(proteins[protein]["NCBI_TaxID"])
                    if "description" in proteins[protein]:
                        description = proteins[protein]["description"]

                    writer.writerow([protein, "Protein", accession , name, ",".join(synonyms), description, taxid])

            for entity, rel in relationships:
                outputfile = os.path.join(importDirectory, "uniprot_"+entity.lower()+"_"+rel.lower()+".csv")
                df = pd.DataFrame(list(relationships[(entity,rel)]))
                df.columns = ['START_ID', 'END_ID','TYPE', 'source']
                df.to_csv(path_or_buf=outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')


        if database.lower() == "intact":
            #IntAct
            intact_file = os.path.join(config.databasesDir,config.intact_file)
            interactions = parseIntactDatabase(intact_file, proteins)
            interactions_outputfile = os.path.join(importDirectory, "intact_interacts_with.csv")

            interactionsDf = pd.DataFrame(list(interactions))
            interactionsDf.columns = ['START_ID', 'END_ID','TYPE', 'score', 'interaction_type', 'method', 'source', 'publications']
    
            interactionsDf.to_csv(path_or_buf=interactions_outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')
        if database.lower() == "disgenet":
            #DisGeNet
            disease_relationships = parseDisGeNetDatabase()

            for idType in disease_relationships:
                disease_outputfile = os.path.join(importDirectory, "disgenet_associated_with.csv")
                disease_relationshipsDf = pd.DataFrame(list(disease_relationships[idType]))
                disease_relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications']    
                disease_relationshipsDf.to_csv(path_or_buf=disease_outputfile, 
                                                header=True, index=False, quotechar='"', 
                                                line_terminator='\n', escapechar='\\')

        if database.lower() == "pathwaycommons":
            #PathwayCommons pathways
            ontologyType = config.pathway_type
            entities, relationships = parsePathwayCommons()
            entity_outputfile = os.path.join(importDirectory, "Pathway.csv")
            with open(entity_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'name', 'description', 'type', 'source', 'linkout'])
                for term, label, name, source, linkout in entities:
                    writer.writerow([term, label, name, name, ontologyType, source, linkout])
            
            pathway_outputfile = os.path.join(importDirectory, "pathwaycommons_protein_associated_with_pathway.csv")
            relationshipsDf = pd.DataFrame(list(relationships))
            relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'source']    
            relationshipsDf.to_csv(path_or_buf= pathway_outputfile, 
                                            header=True, index=False, quotechar='"', 
                                            line_terminator='\n', escapechar='\\')
        if database.lower() == "dgidb":
            relationships = parseDGIdb(mapping = mapping)
            dgidb_outputfile = os.path.join(importDirectory, "dgidb_targets.csv")
            relationshipsDf = pd.DataFrame(list(relationships))
            relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'type', 'source']    
            relationshipsDf.to_csv(path_or_buf= dgidb_outputfile, 
                                            header=True, index=False, quotechar='"', 
                                            line_terminator='\n', escapechar='\\')

        if database.lower() == "oncokb":
            relationships = parseOncoKB(mapping = mapping)
            oncokb_outputfile = os.path.join(importDirectory, "oncokb_targets.csv")
            relationshipsDf = pd.DataFrame(list(relationships))
            relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'type', 'source']    
            relationshipsDf.to_csv(path_or_buf= oncokb_outputfile, 
                                            header=True, index=False, quotechar='"', 
                                            line_terminator='\n', escapechar='\\')
