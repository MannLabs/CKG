import os.path
import re
from collections import defaultdict
import ckg_utils
from graphdb_builder import mapping as mp, builder_utils

#########################
#   OncoKB database     #
#########################
def parser(databases_directory, download = True):
    config = ckg_utils.get_configuration('../databases/config/oncokbConfig.yml')
    url_actionable = config['OncoKB_actionable_url']
    url_annotation = config['OncoKB_annotated_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)

    drugmapping = mp.getMappingForEntity("Drug")

    levels = config['OncoKB_levels']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"OncoKB")
    builder_utils.checkDirectory(directory)
    acfileName = os.path.join(directory,url_actionable.split('/')[-1])
    anfileName = os.path.join(directory,url_annotation.split('/')[-1])
    if download:
        builder_utils.downloadDB(url_actionable, directory)
        builder_utils.downloadDB(url_annotation, directory)

    regex = r"\w\d+(\w|\*|\.)"
    with open(anfileName, 'r', errors='replace') as variants:
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
            entities.add((variant,"Clinically_relevant_variant", "", "", "", "", "", effect, oncogenicity))
            relationships["variant_found_in_gene"].add((variant, gene, "VARIANT_FOUND_IN_GENE"))

    with open(acfileName, 'r', errors='replace') as associations:
        first = True
        for line in associations:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            isoform = data[1]
            gene = data[3]
            variant = data[5]
            disease = data[6]
            level = data[7]
            drugs = data[8].split(', ')
            pubmed_ids = data[9].split(',')
            if level in levels:
                level = levels[level]
            for drug in drugs:
                for d in drug.split(' + '):
                    if d.lower() in drugmapping:
                        drug = drugmapping[d.lower()]
                        relationships["targets_clinically_relevant_variant"].add((drug, variant, "TARGETS_KNOWN_VARIANT", level[0], level[1], disease, "curated", "OncoKB"))
                        relationships["targets"].add((drug, gene, "CURATED_TARGETS", "curated", "OncoKB"))
                    else:
                        pass
                        #print(drug)
            if disease.lower() in mapping:
                disease = mapping[disease.lower()]
                relationships["associated_with"].add((variant, disease, "ASSOCIATED_WITH", "curated","curated", "OncoKB", len(pubmed_ids)))   
            else:
                pass
                #print disease
            relationships["known_variant_is_clinically_relevant"].add((variant, variant, "KNOWN_VARIANT_IS_CLINICALLY_RELEVANT", "OncoKB"))
        relationships["variant_found_in_chromosome"].add(("","",""))


    return (entities, relationships, entities_header, relationships_headers)
