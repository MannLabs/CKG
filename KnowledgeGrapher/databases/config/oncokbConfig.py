###### OncoKB Database #########
OncoKB_annotated_url = "http://oncokb.org/api/v1/utils/allAnnotatedVariants.txt"
OncoKB_actionable_url = "http://oncokb.org/api/v1/utils/allActionableVariants.txt"
OncoKB_levels = {"1": ("approved", "Responsive"), 
                "2A": ("approved","Responsive"), 
                "2B": ("approved", "Responsive other"), 
                "3A": ("Clinical evidence", "Responsive"), 
                "3B":("Clinical evidence", "Responsive other"), 
                "4":("Biological evidence","Responsive"), 
                "4B":("FDA approved","Resistant")}
entities_header = ['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity']
relationships_headers = {"targets_clinically_relevant_variant":['START_ID', 'END_ID','TYPE', 'association', 'evidence', 'tumor', 'type', 'source'],
                        "targets":['START_ID', 'END_ID','TYPE', 'type', 'source'],
                        "associated_with":['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications'],
                        "known_variant_is_clinically_relevant":['START_ID', 'END_ID','TYPE', 'source']}
