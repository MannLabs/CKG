#Database configuration
dbURL = "localhost"
dbPort = 7687
dbUser="neo4j"
dbPassword = "bioinfo1112"
########################
dataDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data"
#Import directory
importDirectory = dataDirectory + "/imports"
#Datasets directory
datasetsImportDirectory = importDirectory + "/datasets/"
#Imports 
entities = ["Disease","Drug","Tissue","Biological_process", "Molecular_function", "Cellular_compartment", "PTM", "Clinical_variable"]
#Database resources
PPI_resources = ["IntAct"]
disease_resources = ["DisGEnet"]
drug_resources = ["DGIdb", "OncoKB"]

#Internal Databases entities
internalEntities = [("Protein","Disease"), ("Protein", "Tissue"), ("Protein","Cellular_compartment")]

#Analyses configuration
similarityMeasures = ["pearson"]
########################

mappingFile = dataDirectory + "/ontologies/mapping.tsv"

#Dataset types
datasetsDirectory = dataDirectory + "/experiments/"
#Proteomics
proteomicsDirectory = datasetsDirectory + "proteomics/"
dataTypes = {"clinicalData":{"directory":proteomicsDirectory,
                            "file":"clinicalData.xlsx"},
            "proteomicsData":{"directory": proteomicsDirectory,
                            "proteins":{"columns":
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
                                        "indexCol" : "Majority protein IDs",
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
                                            "Start position",
                                            "End position",
                                            "Score", 
                                            "Intensity \d+_\d+_?\d*", 
                                            "Reverse",
                                            "Potential contaminant"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant"],
                                        "proteinCol" : "Proteins",
                                        "valueCol" : "Intensity",
                                        "indexCol" : "Sequence",
                                        "positionCols":["Start position","End position"],
                                        "type": "tryptic peptide",
                                        "log": "log10",
                                        "file":"peptides.txt"},
                            "Oxydation(M)":{"columns":
                                            ["Proteins",
                                            "Positions within proteins", 
                                            "Amino acid",
                                            "Sequence window",
                                            "Score", 
                                            "Intensity \d+_\d+", 
                                            "Reverse",
                                            "Potential contaminant"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant"],
                                        "proteinCol" : "Proteins",
                                        "indexCol" : "Proteins",
                                        "valueCol" : "Intensity",
                                        "multipositions": "Positions within proteins",
                                        "positionCols": ["Positions within proteins","Amino acid"],
                                        "sequenceCol": "Sequence window",
                                        "modId":"MOD:00256",
                                        "geneCol":"Gene names",
                                        "log": "log10",
                                        "file":"Oxidation (M)Sites.txt"},
                            "Glycation":{"columns":
                                            ["Proteins",
                                            "Positions within proteins", 
                                            "Amino acid",
                                            "Sequence window",
                                            "Score", 
                                            "Intensity \d+_\d+", 
                                            "Reverse",
                                            "Potential contaminant"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant"],
                                        "proteinCol" : "Proteins",
                                        "indexCol" : "Proteins",
                                        "valueCol" : "Intensity",
                                        "multipositions": "Positions within proteins",
                                        "positionCols": ["Positions within proteins","Amino acid"],
                                        "sequenceCol": "Sequence window",
                                        "modId":"MOD:00764",
                                        "geneCol":"Gene names",
                                        "log": "log10",
                                        "file":"GlycationSites.txt"}
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
