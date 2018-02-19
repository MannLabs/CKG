#Database configuration
dbURL = "localhost"
dbPort = 7687
dbUser="neo4j"
dbPassword = "bioinfo1112"
########################

#Import directory
importDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/Data/imports"
#Datasets directory
datasetsImportDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/Data/imports/datasets/"
#Imports 
entities = ["Disease","Drug","Tissue","Biological_process", "Molecular_function", "Cellular_compartment"]
#Database resources
PPI_resources = ["IntAct","STRING"]
disease_resources = ["DisGEnet"]
drug_resources = ["DGIdb", "OncoKB", "STITCH"]

#Analyses configuration
similarityMeasures = ["pearson"]
########################

mappingFile = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/Data/ontologies/mapping.tsv"

#Dataset types
datasetsDir = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/Data/experiments/proteomics/"
#Proteomics
modifications = {"Glycation":{"code":"MOD:00764", "file":"GlycationSites.txt"}, "Oxidation(M)":{"code":"MOD:00256", "file":"Oxidation (M)Sites.txt"}}

dataTypes = {"clinical":{"files":{"clinical":"clinicalData.xlsx"}},
            "proteomicsData":{"columns":
                                    ["Majority protein IDs",
                                        "Gene names", "Q-value", 
                                        "Score", 
                                        "Intensity \d+_\d+", 
                                        "LFQ intensity \d+_\d+", 
                                        "Reverse",
                                        "Potential contaminant",
                                        "Only identified by site"],
                                    "filters" : ["Reverse",
                                        "Potential contaminant",
                                        "Only identified by site"],
                                    "proteinCol" : "Majority protein IDs",
                                    "geneCol":"Gene names",
                                    "log": "log2",
                                    "files":{"proteins":"proteinGroups.txt", "peptides":"peptides.txt"}},
            "PTMData":{"columns" : 
                            ["Protein", "Positions", 
                                "Amino acid", "Intensity \d+_\d+", 
                                "Reverse", "Potential contaminant"
                                ],
                            "filters" : 
                            ["Reverse", "Potential contaminant"],
                            "proteinCol" : "Protein",
                            "log": "log2"
                            },
            "WESData":{"columns" : [],
                "filters" : 
                ["Reverse", "Potential contaminant"],
                "proteinCol" : "Protein",
                "log": "log2"
                }
        }
###Variant sources
'''
COSMIC
dbNSFP
dbSNP
Linkage-Physical Map
Database of Genomic Variants
Exome Sequencing Project
gwasCatalog
HapMap
Thousand Genomes
'''
#####
