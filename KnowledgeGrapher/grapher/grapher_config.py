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
         "known_variants",
         "clinical variants",
         "drugs", 
         'mentions',
         "internal",
         "side effects",
         'pathway',
         'metabolite',
         "gwas",
         "published",
         "project",
         "experiment"
         ]

#Imports 
ontology_entities = ["Disease","Tissue","Biological_process", "Molecular_function", "Cellular_component", "Modification", "Clinical_variable", "Phenotype", "Experiment"]
#Database resources
#Protein-protein interactions
curated_PPI_resources = ["IntAct"]
compiled_PPI_resources = ["STRING"]
PPI_action_resources = ["STRING"]

#Disease associations
disease_resources = [("Protein","DisGEnet")]


#Drug associations, indications and actions
curated_drug_resources = ["DGIdb","CGI","OncoKB"]
compiled_drug_resources = ["STITCH"]
drug_action_resources = ["STITCH"]
side_effects_resources = ["SIDER"]

#Variants
clinical_variant_resources = ["CGI","OncoKB"]

#Pathways
pathway_resources = ["PathwayCommons"]

#Metabolites
metabolite_resources = ["hmdb"]

#Internal Databases entities
internalEntities = [("Protein","Disease"), ("Protein", "Tissue"), ("Protein","Cellular_component")]

#Mentions entities
mentionEntities = ["Disease", "Tissue", "Protein", "Metabolite"]
publicationEntities = ["GWAS_study"]
