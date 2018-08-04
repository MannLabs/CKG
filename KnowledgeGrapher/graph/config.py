#Database configuration
dbURL = "localhost"
dbPort = 7688 #Production environment
#dbPort = 7687 #Test environment
dbUser="neo4j"
dbPassword = "bioinfo1112"
version = 1.0
########################
dataDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data"
#Import directory
importDirectory = dataDirectory + "/imports"
#Archive directory
archiveDirectory = dataDirectory+"/archive"
#Stats directory
statsDirectory = importDirectory + "/stats"
#experiments import directory
experimentsDirectory = importDirectory +"/experiments"
#databases import diretory
databasesDirectory = importDirectory + "/databases"
#ontologies import diretory
ontologiesDirectory = importDirectory + "/ontologies"

statsFile = "stats.hdf"
statsCols = ["date", "time", "dataset", "filename", "file_size", "Imported_number", "Import_type", "name"]

#Full Graph
graph = ["ontologies", 
         "chromosomes", 
         "genes", 
         "transcripts", 
         "proteins", 
         "ppi", 
         "diseases", 
         "drugs", 
         'mentions',
         "internal",
         "side effects",
         'pathway',
         'metabolite',
         "gwas",
         "published",
         "known_variants",
         "clinical variants",
         "project",
         "experiment"
         ]

#Imports 
ontology_entities = ["Disease","Tissue","Biological_process", "Molecular_function", "Cellular_component", "Modification", "Clinical_variable", "Phenotype", "Experiment"]
#Database resources
PPI_resources = ["IntAct", "STRING"]
disease_resources = [("Protein","DisGEnet"),("Clinically_relevant_variant","CGI"),("Clinically_relevant_variant","OncoKB")]
drug_resources = ["DGIdb","CGI","OncoKB"]
side_effects_resources = ["SIDER"]
clinical_variant_resources = ["CGI","OncoKB"]
pathway_resources = ["PathwayCommons"]
metabolite_resources = ["hmdb"]

#Internal Databases entities
internalEntities = [("Protein","Disease"), ("Protein", "Tissue"), ("Protein","Cellular_compartment")]

#Mentions entities
mentionEntities = ["Disease", "Tissue", "Protein", "Cellular_compartment", "Chemical", "Metabolite"]
publicationEntities = ["GWAS_study"]
