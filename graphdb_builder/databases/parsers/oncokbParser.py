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
    amino_acids = config['amino_acids']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    mapping = mp.getMappingFromOntology(ontology = "Disease", source = None)

    drug_mapping = mp.getMappingForEntity("Drug")
    protein_mapping = mp.getMappingForEntity("Protein")

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

    variant_regex = r"(\D\d+\D)$"
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
            if gene in protein_mapping:
                match = re.search(variant_regex, variant)
                if match:
                    protein = protein_mapping[gene]
                    if variant[0] in amino_acids and variant[-1] in amino_acids:
                        valid_variant = protein + '_p.' + amino_acids[variant[0]] + ''.join(variant[1:-1]) + amino_acids[variant[-1]]
                        entities.add((valid_variant,"Clinically_relevant_variant", "", "", "", "", "", effect, oncogenicity))
    
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
            
            valid_variant = None
            if gene in protein_mapping:
                match = re.search(variant_regex, variant)
                if match:
                    protein = protein_mapping[gene]
                    if variant[0] in amino_acids and variant[-1] in amino_acids:
                        valid_variant = protein + '_p.' + amino_acids[variant[0]] + ''.join(variant[1:-1]) + amino_acids[variant[-1]]
            for drug in drugs:
                for d in drug.split(' + '):
                    if d.lower() in drug_mapping:
                        drug = drug_mapping[d.lower()]
                        relationships["targets"].add((drug, gene, "CURATED_TARGETS", "curated", "NA", "NA", "curated", "OncoKB"))
                        if valid_variant is not None:
                            relationships["targets_clinically_relevant_variant"].add((drug, valid_variant, "TARGETS_KNOWN_VARIANT", level[0], level[1], disease, "curated", "OncoKB"))
            if valid_variant is not None:
                if disease.lower() in mapping:
                    disease = mapping[disease.lower()]
                    relationships["associated_with"].add((valid_variant, disease, "ASSOCIATED_WITH", "curated","curated", "OncoKB", len(pubmed_ids)))   
                else:
                    pass
                    #print disease
                relationships["known_variant_is_clinically_relevant"].add((valid_variant, valid_variant, "KNOWN_VARIANT_IS_CLINICALLY_RELEVANT", "OncoKB"))

    return (entities, relationships, entities_header, relationships_headers)
