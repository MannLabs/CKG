import os.path
from collections import defaultdict
import zipfile
from ckg.graphdb_builder import builder_utils, mapping as mp
import re

#######################################
#   The Cancer Genome Interpreter     #
#######################################
def parser(databases_directory, download=True):
    variant_regex = r"(\D\d+\D)$"
    regex = r"(chr\d+)\:g\.(\d+)(\w)>(\w)"
    config = builder_utils.get_config(config_name="cancerGenomeInterpreterConfig.yml", data_type='databases')
    url = config['cancerBiomarkers_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    amino_acids = config['amino_acids']
    mapping = mp.getMappingFromOntology(ontology="Disease", source=None)
    drugmapping = mp.getMappingForEntity("Drug")
    protein_mapping = mp.getMultipleMappingForEntity("Protein")

    fileName = config['cancerBiomarkers_variant_file']
    relationships = defaultdict(set)
    entities = set()
    directory = os.path.join(databases_directory, "CancerGenomeInterpreter")
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
                    gene_variant = data[0].split(':')
                    if len(gene_variant) < 2:
                        continue
                    gene = gene_variant[0]
                    variants = gene_variant[1].split(',')
                    #alterationType = data[1]
                    response = data[3]
                    drugs = data[10].split(';')
                    #status = data[11].split(';')
                    evidence = data[12]
                    tumors = data[16].split(';')
                    publications = data[17].split(';')
                    identifier = data[21]
                    prot_variant = data[22]
                    matches = re.match(regex, identifier)
                    alternative_names = [identifier]
                    if matches is not None:
                        cpra = matches.groups()
                        chromosome, position, reference, alternative = cpra
                        variant = chromosome+":g."+position+reference+">"+alternative
                        if prot_variant != "":
                            prot_variant = prot_variant.split(':')[1]
                            alternative_names.append(prot_variant)

                    valid_variants = []
                    if gene in protein_mapping:
                        for protein in protein_mapping[gene]:
                            for variant in variants:
                                match = re.search(variant_regex, variant)
                                if match:
                                    if variant[0] in amino_acids and variant[-1] in amino_acids:
                                        valid_variant = protein + '_p.' + amino_acids[variant[0]] + ''.join(variant[1:-1]) + amino_acids[variant[-1]]
                                        valid_variants.append(valid_variant)
                                        entities.add((valid_variant, "Clinically_relevant_variant",  ",".join(alternative_names), chromosome, position, reference, alternative, "", "", "CGI"))
                                        relationships["known_variant_is_clinically_relevant"].add((valid_variant, valid_variant, "KNOWN_VARIANT_IS_CLINICALLY_RELEVANT", "CGI"))

                    for drug in drugs:
                        if drug.lower() in drugmapping:
                            drug = drugmapping[drug.lower()]
                        elif drug.split(" ")[0].lower() in drugmapping:
                            drug = drugmapping[drug.split(" ")[0].lower()]
                        elif " ".join(drug.split(" ")[1:]).lower() in drugmapping:
                            drug = drugmapping[" ".join(drug.split(" ")[1:]).lower()]
                        relationships["targets"].add((drug, gene, "CURATED_TARGETS", evidence, response, ",".join(tumors), "curated", "CGI"))

                        for valid_variant in valid_variants:
                            relationships["targets_clinically_relevant_variant"].add((drug, valid_variant, "TARGETS_CLINICALLY_RELEVANT_VARIANT", evidence, response, "".join(tumors), "curated", "CGI"))

                    for tumor in tumors:
                        if tumor.lower() in mapping:
                            tumor = mapping[tumor.lower()]
                            for valid_variant in valid_variants:
                                relationships["associated_with"].add((valid_variant, tumor, "ASSOCIATED_WITH", "curated","curated", "CGI", len(publications)))

    builder_utils.remove_directory(directory)

    return (entities, relationships, entities_header, relationships_headers)
