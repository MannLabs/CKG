import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import cancerGenomeInterpreterConfig as iconfig
from collections import defaultdict
import zipfile
from KnowledgeGrapher import utils
from KnowledgeGrapher import mapping as mp
import re


#######################################
#   The Cancer Genome Interpreter     # 
#######################################
def parser(download = True):
    regex = r"chr(\d+)\:g\.(\d+)(\w)>(\w)"
    url = iconfig.cancerBiomarkers_url
    entities_header = iconfig.entities_header
    relationships_headers = iconfig.relationships_headers
    mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)
    
    drugsource = dbconfig.sources["Drug"]
    directory = os.path.join(dbconfig.databasesDir, drugsource)
    mappingFile = os.path.join(directory, "mapping.tsv")
    drugmapping = mp.getMappingFromDatabase(mappingFile)
    
    fileName = iconfig.cancerBiomarkers_variant_file
    relationships = defaultdict(set)
    entities = set()
    directory = os.path.join(dbconfig.databasesDir,"CancerGenomeInterpreter")
    zipFile = os.path.join(directory, url.split('/')[-1])

    if download:
        utils.downloadDB(url, "CancerGenomeInterpreter")
    with zipfile.ZipFile(zipFile) as z:
        if fileName in z.namelist():
            with z.open(fileName, 'r') as associations:
                first = True
                for line in associations:
                    if first:
                        first = False
                        continue
                    data = line.decode('utf-8').rstrip("\r\n").split("\t")
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
                        entities.add((variant, "Clinically_relevant_variant", identifier, "chr"+chromosome, position, reference, alternative, "", ""))
                        for tumor in tumors:                         
                            if tumor.lower() in mapping:
                                tumor = mapping[tumor.lower()]
                            relationships["associated_with"].add((variant, tumor, "ASSOCIATED_WITH", "curated","curated", "Cancer Genome Interpreter", len(publications)))
                            for drug in drugs:
                                if drug.lower() in drugmapping:
                                    drug = drugmapping[drug.lower()]
                                elif drug.split(" ")[0].lower() in drugmapping:
                                    drug = drugmapping[drug.split(" ")[0].lower()]
                                relationships["targets_clinically_relevant_variant"].add((drug, variant, "TARGETS_KNOWN_VARIANT", evidence, association, tumor, "curated", "Cancer Genome Interpreter"))
                                relationships["targets"].add((drug, gene, "CURATED_TARGETS", "curated", "CGI"))

                        #relationships["variant_found_in_gene"].add((variant, gene, "VARIANT_FOUND_IN_GENE"))
                        #relationships["variant_found_in_chromosome"].add((variant, chromosome, "VARIANT_FOUND_IN_CHROMOSOME"))
                        relationships["known_variant_is_clinically_relevant"].add((variant, variant, "KNOWN_VARIANT_IS_CLINICALLY_RELEVANT", "CGI"))
        
    return (entities, relationships, entities_header, relationships_headers)
