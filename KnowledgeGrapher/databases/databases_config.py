#Database directory
databasesDir = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/data/databases/"
databasesImportDir = "databases"
#Databases
databases = [
            "Internal",
            "HGNC", 
            "RefSeq", 
            "UniProt", 
            "IntAct", 
            "DisGEnet", 
            'DrugBank',
            "DGIdb", 
            "OncoKB", 
            "STRING", 
            "STITCH", 
            "Mentions", 
            "CancerGenomeInterpreter", 
            "SIDER",
            "HMDB",
            "PathwayCommons",
            'GWASCatalog'
            ]

sources = {
            "Drug":"DrugBank",
            "Metabolite":"HMDB",
            "Protein":"UniProt",
            "Gene":"HGNC"    
            }
