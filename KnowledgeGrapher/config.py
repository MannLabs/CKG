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
entities = ["Disease","Drug","Tissue","Biological_process", "Molecular_function", "Cellular_compartment", "Postranslational_modification", "Clinical_variable", "Metabolite"]
#Database resources
PPI_resources = ["IntAct"]
disease_resources = [("Protein","DisGEnet"),("Known_variant","CGI"),("Known_variant","OncoKB")]
drug_resources = ["DGIdb","CGI","OncoKB"]
side_effects_resources = ["SIDER"]
variant_resources = ["CGI","OncoKB"]
pathway_resources = ["PathwayCommons"]
metabolite_resources = ["hmdb"]

#Internal Databases entities
internalEntities = [("Protein","Disease"), ("Protein", "Tissue"), ("Protein","Cellular_compartment")]

#Mentions entities
mentionEntities = ["Disease", "Tissue", "Protein", "Cellular_compartment", "Chemical", "Metabolite"]

#Analyses configuration
similarityMeasures = ["pearson"]
########################


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
