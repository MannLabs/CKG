#Database configuration
dbURL = "localhost"
dbPort = 7687
dbUser="neo4j"
dbPassword = "bioinfo1112"
########################
dataDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/Data"
#Import directory
importDirectory = dataDirectory + "/imports"
#Datasets directory
datasetsImportDirectory = importDirectory + "/datasets/"
#Imports 
entities = ["Disease","Drug","Tissue","Biological_process", "Molecular_function", "Cellular_compartment", "Pathway"]
#Database resources
PPI_resources = ["IntAct","STRING"]
disease_resources = ["DisGEnet"]
drug_resources = ["DGIdb", "OncoKB", "STITCH"]

#Analyses configuration
similarityMeasures = ["pearson"]
########################

mappingFile = dataDirectory + "/ontologies/mapping.tsv"

#Dataset types
datasetsDirectory = dataDirectory + "/experiments/"
#Proteomics
proteomicsDirectory = datasetsDirectory + "proteomics/"
modifications = {"Glycation":{"code":"MOD:00764", "file":"GlycationSites.txt"}, "Oxidation(M)":{"code":"MOD:00256", "file":"Oxidation (M)Sites.txt"}}
dataTypes = {"clinicalData":{"clinical":{"file":"clinicalData.xlsx"}},
            "proteomicsData":{"proteins":{"columns":
                                            ["Majority protein IDs",
                                            "Gene names", "Q-value", 
                                            "Score", 
                                            "LFQ intensity \d+_\d+_?\d*",  #subject_replicate_timepoint
                                            "Reverse",
                                            "Potential contaminant",
                                            "Only identified by site"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant",
                                                        "Only identified by site"],
                                        "proteinCol" : "Majority protein IDs",
                                        "valueCol" : "LFQ intensity",
                                        "log": "log10",
                                        "file": "proteinGroups.txt"},
                            "peptides":{"columns":
                                            ["Sequence",
                                            "Amino acid before",
                                            "First amino acid",
                                            "Second amino acid",
                                            "Second last amino acid",
                                            "Last amino acid",
                                            "Amino acid after", 
                                            "Proteins", 
                                            "Score", 
                                            "Intensity \d+_\d+_?\d*", 
                                            "Reverse",
                                            "Potential contaminant"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant"],
                                        "proteinCol" : "Proteins",
                                        "valueCol" : "Intensity",
                                        "log": "log10",
                                        "file":"peptides.txt"},
                            },
            "PTMData":{"columns" : 
                            ["Protein", "Positions", 
                                "Amino acid", "Intensity \d+_\d+", 
                                "Reverse", "Potential contaminant"],
                            "filters" : ["Reverse", "Potential contaminant"],
                            "proteinCol" : "Protein",
                            "log": "log10"
                            },
            "WESData":{"columns" : [],
                "filters" : ["Reverse", "Potential contaminant"],
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
