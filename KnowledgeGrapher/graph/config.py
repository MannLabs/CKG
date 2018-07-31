#Database configuration
dbURL = "localhost"
dbPort = 7687
dbUser="neo4j"
dbPassword = "bioinfo1112"
version = 1.0
########################
dataDirectory = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data"
#Import directory
importDirectory = dataDirectory + "/imports"
#Stats directory
statsDirectory = importDirectory + "/stats"
statsFile = "stats.hdf"

#Imports 
entities = ["Disease","Drug","Tissue","Biological_process", "Molecular_function", "Cellular_compartment", "Postranslational_modification", "Clinical_variable", "Metabolite", "GWAS_study","Clinically_relevant_variant", "Known_variant", "Phenotype", "Experiment"]
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


#Import statistics
statsCols = ["date", "dataset", "dataset_type", "Original_number", "Imported_number", "Import_type", "name", "file_name", "file_size"]

