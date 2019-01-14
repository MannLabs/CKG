import os.path
from collections import defaultdict
import zipfile
import ckg_utils
from graphdb_builder import builder_utils, mapping as mp
import re

#######################################
#   The Cancer Genome Interpreter     # 
#######################################
def parser(databases_directory, download = True):
    regex = r"(chr\d+)\:g\.(\d+)(\w)>(\w)"
    
    config = ckg_utils.get_configuration('../databases/config/cancerGenomeInterpreterConfig.yml')
    url = config['cancerBiomarkers_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)
    drugmapping = mp.getMappingForEntity("Drug")
    
    fileName = config['cancerBiomarkers_variant_file']
    relationships = defaultdict(set)
    entities = set()
    directory = os.path.join(databases_directory,"CancerGenomeInterpreter")
    builder_utils.checkDirectory(directory)
    zipFile = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, directory)
    with zipfile.ZipFile(zipFile) as z:
        if fileName in z.namelist():
            with z.open(fileName, 'r') as responses:
                first = True
                for line in responses:
                    if first:
                        first = False
                        continue
                    data = line.decode('utf-8').rstrip("\r\n").split("\t") 
                    alterationType = data[1]
                    response = data[3]
                    drugs = data[10].split(';')
                    status = data[11].split(';')
                    evidence = data[12]
                    gene = data[13]
                    tumors = data[16].split(';')
                    publications = data[17].split(';')
                    identifier = data[21]
                    variant = data[22]
                    matches = re.match(regex, identifier)
                    chromosome = "NA"
                    position = "NA"
                    reference = "NA" 
                    alternative = "NA"

                    if matches is not None:
                        cpra = matches.groups()
                        chromosome, position, reference, alternative = cpra

                    if variant != "":
                        variant = variant.split(':')[1]
                        entities.add((variant, "Clinically_relevant_variant", identifier, chromosome, position, reference, alternative, "", ""))
                        relationships["known_variant_is_clinically_relevant"].add((variant, variant, "KNOWN_VARIANT_IS_CLINICALLY_RELEVANT", "CGI"))

                    for drug in drugs:
                        if drug.lower() in drugmapping:
                            drug = drugmapping[drug.lower()]
                        elif drug.split(" ")[0].lower() in drugmapping:
                            drug = drugmapping[drug.split(" ")[0].lower()]
                        elif " ".join(drug.split(" ")[1:]).lower() in drugmapping:
                            drug = drugmapping[" ".join(drug.split(" ")[1:]).lower()]
                        for tumor in tumors:                         
                            if tumor.lower() in mapping:
                                tumor = mapping[tumor.lower()]
                                relationships["targets"].add((drug, gene, "CURATED_TARGETS", evidence, response, tumor, "curated", "CGI"))
                                if variant != "":
                                    relationships["associated_with"].add((variant, tumor, "ASSOCIATED_WITH", "curated","curated", "CGI", len(publications)))
                                    relationships["targets_clinically_relevant_variant"].add((drug, variant, "TARGETS_CLINICALLY_RELEVANT_VARIANT", evidence, response, tumor, "curated", "CGI"))
                                    #relationships["clinically_relevant_variant_found_in_gene"].add((variant, gene, "CLINICALLY_RELEVANT_VARIANT_FOUND_IN_GENE"))
                                    #relationships["clinically_relevant_variant_found_in_chromosome"].add((variant, chromosome, "CLINICALLY_RELEVANT_VARIANT_FOUND_IN_CHROMOSOME"))

    return (entities, relationships, entities_header, relationships_headers)
