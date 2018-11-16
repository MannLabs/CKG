##### Cancer Genome Interpreter ######
cancerGenes_url = "https://www.cancergenomeinterpreter.org/data/catalog_of_cancer_genes_latest.zip"
cancerVariants_url = "https://www.cancergenomeinterpreter.org/data/catalog_of_validated_oncogenic_mutations_latest.zip"
cancerBioactivities_url = "https://www.cancergenomeinterpreter.org/data/cancer_bioactivities_latest.zip"

cancerBiomarkers_url = "https://www.cancergenomeinterpreter.org/data/cgi_biomarkers_latest.zip"
cancerBiomarkers_variant_file = "cgi_biomarkers_per_variant.tsv"
entities_header = ['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity']
relationships_headers =  {"targets_clinically_relevant_variant":['START_ID', 'END_ID','TYPE', 'evidence', 'association', 'tumor', 'type', 'source'],
                            "targets":['START_ID', 'END_ID','TYPE', 'type', 'source'],
                            "associated_with":['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications'],
                            "known_variant_is_clinically_relevant":['START_ID', 'END_ID','TYPE', 'source']}

