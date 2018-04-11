import os.path
import gzip
import databases_config as config
import ontologies_config as onto_config
from collections import defaultdict
import mapping as mp
import csv
import pandas as pd
import re
from lxml import etree
import zipfile
import utils

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
    
    urllib.URLopener.version = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36 SE 2.X MetaSr 1.0'
    requestedFile = urllib.URLopener()
    requestedFile.retrieve(databaseURL, os.path.join(directory, fileName))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#############################################
#   Internal Databases (JensenLab.org)      # 
#############################################
def parseInternalDatabasePairs(qtype, mapping, download = True):
    url = config.internal_db_url
    ifile = config.internal_db_files[qtype]
    source = config.internal_db_sources[qtype]
    relationships = set()
    directory = os.path.join(config.databasesDir, "InternalDatabases")
    if download:
        downloadDB(url.replace("FILE", ifile), os.path.join(directory,"integration"))
    ifile = os.path.join(directory,os.path.join("integration",ifile))
    with open(ifile, 'r') as idbf:
        for line in idbf:
            data = line.rstrip("\r\n").split('\t')
            id1 = "9606."+data[0]
            id2 = data[2]
            score = float(data[4])

            if id1 in mapping:
                for ident in mapping[id1]:
                    relationships.add((ident, id2, "ASSOCIATED_WITH_INTEGRATED", source, score))
            else:
                continue
                
    return relationships

def parseInternalDatabaseMentions(qtype, mapping, importDirectory, download = True):
    url = config.internal_db_url
    ifile = config.internal_db_mentions_files[qtype]
    filters = []
    if qtype in config.internal_db_mentions_filters:
        filters = config.internal_db_mentions_filters[qtype]
    entity1, entity2 = config.internal_db_mentions_types[qtype]
    outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_mentioned_in_publication.csv")
    entities = set()
    relationships = pd.DataFrame()
    directory = os.path.join(config.databasesDir, "InternalDatabases")
    if download:
        downloadDB(url.replace("FILE", ifile), os.path.join(directory,"textmining"))
    ifile = os.path.join(directory,os.path.join("textmining",ifile))
    
    with open(outputfile,'a') as f:
        f.write("START_ID,END_ID,TYPE\n")
        with open(ifile, 'r') as idbf:
            for line in idbf:
                data = line.rstrip("\r\n").split('\t')
                id1 = data[0]
                pubmedids = data[1].split(" ")
                entities.update(set(pubmedids))
#            print entities
                
                if qtype == "9606":
                    id1 = "9606."+id1
                    if id1 in mapping:
                        ident = mapping[id1]
                    else:
                        continue
                else:
                    ident = [id1]
                for i in ident:
                    if i not in filters:
                        aux = pd.DataFrame(data = {"Pubmedids":pubmedids})
                        aux["START_ID"] = i
                        aux["TYPE"] = "MENTIONED_IN_PUBLICATION"
                        aux.to_csv(path_or_buf=f, header=False, index=False, quotechar='"', line_terminator='\n', escapechar='\\')
    
    return entities
    
    
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

#######################################
#   The Cancer Genome Interpreter     # 
#######################################
def parseCGI(download = True, mapping = {}):
    regex = r"chr(\d+)\:g\.(\d+)(\w)>(\w)"
    url = config.cancerBiomarkers_url
    fileName = config.cancerBiomarkers_variant_file
    relationships = defaultdict(set)
    entities = set()
    directory = os.path.join(config.databasesDir,"CancerGenomeInterpreter")
    zipFile = os.path.join(directory, url.split('/')[-1])

    if download:
        downloadDB(url, "CancerGenomeInterpreter")
    #-f1,11,12,13,14,15,17,22
    with zipfile.ZipFile(zipFile) as z:
        if fileName in z.namelist():
            with z.open(fileName, 'r') as associations:
                first = True
                for line in associations:
                    if first:
                        first = False
                        continue
                    data = line.rstrip("\r\n").split("\t")
                    alteration = data[0]
                    alterationType = data[1]
                    association = data[3]
                    drugs = data[10].split(';')
                    status = data[11].split(';')
                    evidence = data[12]
                    gene = data[13]
                    tumors = data[16].split(';')
                    publications = data[17].split(';')
                    identifier = data[21]
                    matches = re.match(regex, identifier)
                    if matches is not None:
                        chromosome, position, reference, alternative = list(matches.groups())
                        alteration = alteration.split(':')[1]
                    else:
                        continue

                    for variant in alteration.split(','):
                        entities.add((variant, "Known_variant", identifier, "chr"+chromosome, position, reference, alternative, "", ""))
                        for tumor in tumors:                         
                            if tumor.lower() in mapping:
                                tumor = mapping[tumor.lower()]
                            relationships["associated_with"].add((variant, tumor, "ASSOCIATED_WITH", "curated","curated", "Cancer Genome Interpreter", len(publications)))
                            for drug in drugs:
                                if drug.lower() in mapping:
                                    drug = mapping[drug.lower()]
                                elif drug.split(" ")[0].lower() in mapping:
                                    drug = mapping[drug.split(" ")[0].lower()]
                                relationships["targets_known_variant"].add((drug, variant, "TARGETS_KNOWN_VARIANT", evidence, association, tumor, "curated", "Cancer Genome Interpreter"))
                                relationships["targets"].add((drug, gene, "CURATED_TARGETS", "curated", "OncoKB"))

                        relationships["variant_found_in_gene"].add((variant, gene, "VARIANT_FOUND_IN_GENE"))
                        relationships["variant_found_in_chromosome"].add((variant, chromosome, "VARIANT_FOUND_IN_CHROMOSOME"))
        
    return entities, relationships

#########################
#   OncoKB database     #
#########################
def parseOncoKB(download = False, mapping = {}):
    url_actionable = config.OncoKB_actionable_url
    url_annotated = config.OncoKB_annotated_url
    levels = config.OncoKB_levels
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(config.databasesDir,"OncoKB")
    acfileName = os.path.join(directory,url_actionable.split('/')[-1])
    anfileName = os.path.join(directory,url_annotated.split('/')[-1])
    if download:
        downloadDB(url_actionable, "OncoKB")
        downloadDB(url_annotation, "OncoKB")

    regex = r"\w\d+(\w|\*|\.)"
    with open(anfileName, 'r') as variants:
        first = True
        for line in variants:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            gene = data[3]
            variant = data[4]
            oncogenicity = data[5]
            effect = data[6]          
            entities.add((variant,"Known_variant", "", "", "", "", "", effect, oncogenicity))
            relationships["variant_found_in_gene"].add((variant, gene, "VARIANT_FOUND_IN_GENE"))

    with open(acfileName, 'r') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            isoform = data[1]
            gene = data[3]
            variant = data[4]
            disease = data[5]
            level = data[6]
            drugs = data[7].split(', ')
            pubmed_ids = data[8].split(',')
            if level in levels:
                level = levels[level]
            for drug in drugs:
                if drug.lower() in mapping:
                    drug = mapping[drug.lower()]
                else:
                    pass
                    #print drug
                if disease.lower() in mapping:
                    disease = mapping[disease.lower()]
                else:
                    pass
                    #print disease
                relationships["targets_known_variant"].add((drug, variant, "TARGETS_KNOWN_VARIANT", level[0], level[1], disease, "curated", "OncoKB"))
                relationships["associated_with"].add((variant, disease, "ASSOCIATED_WITH", "curated","curated", "OncoKB", len(pubmed_ids)))   
                relationships["targets"].add((drug, gene, "CURATED_TARGETS", "curated", "OncoKB"))
        relationships["variant_found_in_chromosome"].add(("","",""))
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
            drug = data[8].lower()
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

#################################
#   Human Metabolome Database   # 
#################################
def parseHMDB(download = True, mapping = {}):
    data = defaultdict()
    prefix = "{http://www.hmdb.ca}"
    url = config.HMDB_url
    relationships = set()
    directory = os.path.join(config.databasesDir,"HMDB")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        downloadDB(url, "HMDB")
    fields = config.HMDB_fields
    with zipfile.ZipFile(filename, 'r') as associations:
        context = etree.iterwalk(associations, events=("end",), tag=prefix+"metabolite")
        for _,elem in context:
            [ child.tag for child in root.iterchildren() ]
            values = {child.tag.replace(prefix,''):child.text for child in element.iterchildren() if child.tag.replace(prefix,'') in fileds and child.text is not None}
            print values
            sys.exit()
                    

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

def parseUniProtVariants(dataFile, download = True):
    data = defaultdict()
    url = config.uniprot_variant_file
    relationships = set()
    directory = os.path.join(config.databasesDir,"UniProt")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        downloadDB(url, "UniProt")
    with gzip.open(filename, 'r') as f:
        din = False
        i = 0
        for line in f:
            if not line.startswith('#') and not din:
                continue
            elif i<2:
                din = True
                i += 1
                continue
            data = line.rstrip("\r\n").split("\t")

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
    intact_dictionary = defaultdict()
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
                if (intA, intB) in intact_dictionary:
                    intact_dictionary[(intA,intB)]['methods'].add(method)
                    intact_dictionary[(intA,intB)]['sources'].add(source)
                    intact_dictionary[(intA,intB)]['publications'].add(publications.replace('|',','))
                    intact_dictionary[(intA,intB)]['itype'].add(itype)
                else:
                    intact_dictionary[(intA,intB)]= {'methods': set([method]),'sources':set([source]),'publications':set([publications]), 'itype':set([itype]), 'score':score}
    for (intA, intB) in intact_dictionary:
        intact_interactions.add((intA,intB,"CURATED_INTERACTS_WITH",intact_dictionary[(intA, intB)]['score'], ",".join(intact_dictionary[(intA, intB)]['itype']), ",".join(intact_dictionary[(intA, intB)]['methods']), ",".join(intact_dictionary[(intA, intB)]['sources']), ",".join(intact_dictionary[(intA, intB)]['publications'])))

    return intact_interactions

#########################
#   STRING like DBs     #
#########################
def getSTRINGMapping(source = "BLAST_UniProt_AC", download = True):
    mapping = defaultdict(set)
    url = config.STRING_mapping_url
    
    directory = os.path.join(config.databasesDir, "STRING")
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        downloadDB(url, "STRING")
    
    f = os.path.join(directory, fileName)
    mf = gzip.open(f, 'r')
    first = True
    for line in mf:
        if first:
            first = False
            continue
        data = line.rstrip("\r\n").split("\t")
        stringID = data[0]
        alias = data[1]
        sources = data[2].split(' ')
        if source in sources:
            mapping[stringID].add(alias)
        
    return mapping

def parseSTRINGLikeDatabase(mapping, db= "STRING", download = True):
    string_interactions = set()
    cutoff = config.STRING_cutoff
    
    if db == "STITCH":
        evidences = ["experimental", "prediction", "database","textmining", "score"]
        relationship = "COMPILED_INTERACTS_WITH"
        url = config.STITCH_url
    elif db == "STRING":
        evidences = ["experimental", "prediction", "database","textmining", "score"]
        relationship = "COMPILED_TARGETS"
        url = config.STRING_url

    directory = os.path.join(config.databasesDir, db)
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        downloadDB(url, db)
    
    f = os.path.join(directory, fileName)
    associations = gzip.open(f, 'r')
    first = True
    for line in associations:
        if first:
            first = False
            continue
        data = line.rstrip("\r\n").split()
        intA = data[0]
        intB = data[1]
        scores = data[2:]
        fscores = [str(float(score)/1000) for score in scores]
        if intA in mapping and intB in mapping and fscores[-1]>=cutoff:
            for aliasA in mapping[intA]:
                for aliasB in mapping[intB]:
                    string_interactions.add((aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1]))
        elif db == "STITCH":
            if intB in mapping and fscores[-1]>=cutoff:
                aliasA = intA
                for aliasB in mapping[intB]:
                    string_interactions.add((aliasA, aliasB, relationship, "association", db, ",".join(evidences), ",".join(fscores[0:-1]), fscores[-1]))
    return string_interactions


#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory):
    mapping = getMapping()
    string_mapping = getSTRINGMapping()
    databases = config.databases
    for database in databases:
        print database
        if database.lower() == "internal":
            for qtype in config.internal_db_types:
                relationships, entity1, entity2 = parseInternalDatabasePairs(qtype, string_mapping)
                entity1, entity2 = config.internal_db_types[qtype]
                outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_associated_with_integrated.csv")
                df = pd.DataFrame(list(relationships))
                df.columns = ["START_ID", "END_ID", "TYPE", "source", "score"]
                df.to_csv(path_or_buf=outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')
        if database.lower() == "mentions":
            plinkout = config.pubmed_linkout
            publications = set()
            for qtype in config.internal_db_mentions_types:
                entities = parseInternalDatabaseMentions(qtype, string_mapping, importDirectory)
                publications.update(entities)                
            publications_outputfile = os.path.join(importDirectory, "Publications.csv")
            with open(publications_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', "linkout"])
                for pubmedid in publications:
                    writer.writerow([pubmedid, plinkout.replace("PUBMEDID", pubmedid)])
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
            interactions_outputfile = os.path.join(importDirectory, "INTACT_interacts_with.csv")

            interactionsDf = pd.DataFrame(list(interactions))
            interactionsDf.columns = ['START_ID', 'END_ID','TYPE', 'score', 'interaction_type', 'method', 'source', 'publications']
    
            interactionsDf.to_csv(path_or_buf=interactions_outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')

        if database.lower() == "string":
            #STRING
            interactions = parseSTRINGLikeDatabase(string_mapping)
            interactions_outputfile = os.path.join(importDirectory, "STRING_interacts_with.csv")

            interactionsDf = pd.DataFrame(list(interactions))
            interactionsDf.columns = ['START_ID', 'END_ID','TYPE', 'interaction_type', 'source', 'evidences','scores', 'score']
    
            interactionsDf.to_csv(path_or_buf=interactions_outputfile, 
                                header=True, index=False, quotechar='"', 
                                line_terminator='\n', escapechar='\\')

        if database.lower() == "stitch":
            #STITCH
            evidences = ["experimental", "prediction", "database","textmining", "score"]
            interactions = parseSTRINGLikeDatabase(string_mapping, db = "STITCH")
            interactions_outputfile = os.path.join(importDirectory, "STITCH_associated_with.csv")

            interactionsDf = pd.DataFrame(list(interactions))
            interactionsDf.columns = ['START_ID', 'END_ID','TYPE', 'interaction_type', 'source', 'evidences','scores', 'score'] 
    
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
            entities, relationships = parseOncoKB(mapping = mapping)
            entity_outputfile = os.path.join(importDirectory, "oncokb_Known_variant.csv")
            with open(entity_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity'])
                for entity in entities:
                    writer.writerow(entity)
            for relationship in relationships:
                oncokb_outputfile = os.path.join(importDirectory, "oncokb_"+relationship+".csv")
                relationshipsDf = pd.DataFrame(list(relationships[relationship]))
                if relationship == "targets_known_variant":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'association', 'evidence', 'tumor', 'type', 'source']
                elif relationship == "targets":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'type', 'source']
                elif relationship == "associated_with":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications'] 
                else:
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE']
                relationshipsDf.to_csv(path_or_buf= oncokb_outputfile, 
                                            header=True, index=False, quotechar='"', 
                                            line_terminator='\n', escapechar='\\')
        
        if database.lower() == "cancergenomeinterpreter":
            entities, relationships = parseCGI(mapping = mapping)
            entity_outputfile = os.path.join(importDirectory, "cgi_Known_variant.csv")
            with open(entity_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity'])
                for entity in entities:
                    writer.writerow(entity)
            for relationship in relationships:
                cgi_outputfile = os.path.join(importDirectory, "cgi_"+relationship+".csv")
                relationshipsDf = pd.DataFrame(list(relationships[relationship]))
                if relationship == "targets_known_variant":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'evidence', 'association', 'tumor', 'type', 'source']
                elif relationship == "targets":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'type', 'source']
                elif relationship == "associated_with":
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications']
                else:
                    relationshipsDf.columns = ['START_ID', 'END_ID','TYPE']
                relationshipsDf.to_csv(path_or_buf= cgi_outputfile, 
                                            header=True, index=False, quotechar='"', 
                                            line_terminator='\n', escapechar='\\')

        if database.lower() == "HMDB":
            parseHMDB()
