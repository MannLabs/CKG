#Ontologies directory
ontologiesDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data/ontologies/"

ontologies = {"Disease": "DO", 
              "Tissue": "BTO", 
              "Clinical_variable": "SNOMED-CT",
              "Phenotype":"HPO",
              "Experiment":"PSI-MS",
               #"Food":"BBC",
               #"Physical_activity":"SMASH",
              "Postranslational_modification":"PSI-MOD",
              "Gene_ontology": "GO"
            }

ontology_types = {"DO":-26, 
                  "BTO":-25,
                  "SNOMED-CT":-40,
                  "HPO":-44,
                  "PSI-MOD":-41,
                  "PSI-MS":-42,
                  "BBC":-43,
                  "SMASH":-42,
                  "GO":-21
            }

parser_filters = {-40:["308916002", "363787002", "373873005", "71388002", "48176007", "105590001"]}

files = {-26: ["DO/do.obo"],
         -25: ["BTO/bto.obo"],
         -21: ["GO/go.obo"],
         -40: ["SNOMED-CT/Full/Terminology/sct2_Description_Full-en_INT_20170731.txt", 
               "SNOMED-CT/Full/Terminology/sct2_Relationship_Full_INT_20170731.txt", 
               "SNOMED-CT/Full/Terminology/sct2_TextDefinition_Full-en_INT_20170731.txt"
              ],
        -41: ["PSI-MOD/psi-mod.obo.txt"],
        -42: ["PSI-MS/psi-ms.obo.txt"],
        -44: ["HPO/hp.obo",
              "phenotype_annotation_hpoteam.tab",
              "ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt"
             ]
        }



