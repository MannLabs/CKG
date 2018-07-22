#Ontologies directory
ontologiesDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data/ontologies/"

ontologies = {#"Disease": "DO", 
              # "Tissue": "BTO", 
               "Clinical_variable": "SNOMED-CT",
               #"Experiment":"PSI-MS",
               #"Peptide_variant":"HGV",
               #"Gene_variant":"HGV",
              # "Drug":"STITCH",
               #"Food":"BBC",
               #"Physical_activity":"SMASH",
               #"Postranslational_modification":"PSI-MOD",
               #"Biological_process":"GOBP",
               #"Cellular_compartment":"GOCC",
               #"Molecular_function":"GOMF",
               #"Diagnose":"ICD"
            }

ontology_types = {"DO":-26, 
                  "BTO":-25,
                  "SNOMED-CT":-40,
                  "PSI-MOD":-41,
                  "PSI-MS":-42,
                  "STITCH":-1,
                  "BBC":-43,
                  "SMASH":-42,
                  "GOBP":-21,
                  "GOMF":-22,
                  "GOCC":-23,
                  "ICD":-44
            }

parser_filters = {-40:["308916002", "363787002", "373873005", "71388002", "48176007", "105590001"]}

files = {-44: ["ICD/version_10/icd10_2016.tsv"],
         -40: ["SNOMED-CT/Full/Terminology/sct2_Description_Full-en_INT_20170731.txt", 
               "SNOMED-CT/Full/Terminology/sct2_Relationship_Full_INT_20170731.txt", 
               "SNOMED-CT/Full/Terminology/sct2_TextDefinition_Full-en_INT_20170731.txt"
             ],
         -26: ["reflect/doid_entities.tsv",
             "reflect/doid_names_disambiguated.tsv",
             "reflect/doid_groups.tsv",
             "reflect/doid_texts.tsv"
             ],
         -25: ["reflect/bto_entities.tsv",
             "reflect/bto_names_disambiguated.tsv",
             "reflect/bto_groups.tsv",
             "reflect/bto_texts.tsv"
             ],
         -23: ["reflect/go_entities.tsv",
             "reflect/go_names_disambiguated.tsv",
             "reflect/go_groups.tsv",
             "reflect/go_texts.tsv"
             ],
         -22: ["reflect/go_entities.tsv",
             "reflect/go_names_disambiguated.tsv",
             "reflect/go_groups.tsv",
             "reflect/go_texts.tsv"
             ],
         -21: ["reflect/go_entities.tsv",
             "reflect/go_names_disambiguated.tsv",
             "reflect/go_groups.tsv",
             "reflect/go_texts.tsv"
             ],
         -1: ["reflect/stitch_entities.tsv",
             "reflect/stitch_names.tsv",
             "reflect/stitch_groups.tsv",
             "reflect/stitch_texts.tsv"
             ],
        -41: ["PSI/psi-mod.obo.txt"]
        }



