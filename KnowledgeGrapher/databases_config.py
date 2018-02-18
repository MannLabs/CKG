#Database directory
databasesDir = "/Users/albertosantos/Development/Clinical_Proteomics_Department/ClinicalKnowledgeGraph(CKG)/databases/"

#Databases
#databases = ["UniProt", "IntAct", "DisGEnet", "HGNC", "DGIdb", "OncoKB"]
databases = ["HGNC", "RefSeq", "UniProt"]
###### UniProt Database ########
uniprot_id_url = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"

uniprot_id_file = "/Users/albertosantos/Development/UniProt/HUMAN_9606_idmapping.dat"
uniprot_text_file = "/Users/albertosantos/Development/UniProt/uniprot-human.tab" #### Downloaded manually from UniProt until we know url (organism:human AND reviewed:yes)

uniprot_ids = ["UniProtKB-ID", "NCBI_TaxID", "Gene_Name", "RefSeq", "PDB", "STRING", "KEGG", "Reactome", "HPA", "ChEMBL", "Ensembl"]

uniprot_synonyms = ["UniProtKB-ID", "Gene_Name", "STRING", "HPA", "Ensembl", "ChEMBL", "PDB"]
uniprot_protein_relationships = {"KEGG": ("Pathway", "IS_PART_OF_KEGG_PATHWAY"), 
                                "Reactome":("Pathway","IS_PART_OF_REACTOME_PATHWAY"), 
                                "RefSeq": ("Transcript", "TRANSLATED_INTO"), 
                                "Gene_Name":("Gene","TRANSLATED_INTO")
                                }

##### HUGO Gene Nomenclature #######
hgnc_url = "ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt"

##### RefSeq #########
refseq_url = "ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/latest_assembly_versions/GCF_000001405.37_GRCh38.p11/GCF_000001405.37_GRCh38.p11_feature_table.txt.gz"
headerEntities = {"Transcript":["ID", ":LABEL", "name", "class", "assembly", "taxid"], 
                "Chromosome":["ID", ":LABEL", "name", "taxid"],
                "LOCATED_IN": ['START_ID', 'END_ID','TYPE','start','end','strand', 'source'],
                "TRANSCRIBED_INTO": ['START_ID', 'END_ID','TYPE', 'source']}

###### PathwayCommons Database #######
pathwayCommons_pathways_url = "http://www.pathwaycommons.org/archives/PC2/v9/PathwayCommons9.All.uniprot.gmt.gz"
pathway_type = -45

###### The Drug Gene Interaction Database (DGIdb) #########
DGIdb_url = "http://www.dgidb.org/data/interactions.tsv"

###### OncoKB Database #########
OncoKB_url = "http://oncokb.org/api/v1/utils/allActionableVariants.txt"

##### Cancer Genome Interpreter ######
cancerGenes_url = "https://www.cancergenomeinterpreter.org/data/catalog_of_cancer_genes_latest.zip"
cancerVariants_url = "https://www.cancergenomeinterpreter.org/data/catalog_of_validated_oncogenic_mutations_latest.zip"
cancerBiomarkers_url = "https://www.cancergenomeinterpreter.org/data/cgi_biomarkers_latest.zip"
cancerBioactivities_url = "https://www.cancergenomeinterpreter.org/data/cancer_bioactivities_latest.zip"

###### IntAct Database #######
intact_psimitab_url = "ftp://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.txt"
intact_file = "Intact/intact/intact.txt"

##### DisGeNet Database ######
disgenet_url = "http://www.disgenet.org/ds/DisGeNET/results/"

disgenet_files = {"gene_curated":"curated_gene_disease_associations.tsv.gz",
                    "gene_befree": "befree_gene_disease_associations.tsv.gz",
                    "variant_curated":"curated_variant_disease_associations.tsv.gz",
                    "variant_befree":"befree_variant_disease_associations.tsv.gz"
                }
disgenet_mapping_files = {"protein_mapping":"mapa_geneid_4_uniprot_crossref.tsv.gz",
                            "disease_mapping":"disease_mappings.tsv.gz"
                        }
