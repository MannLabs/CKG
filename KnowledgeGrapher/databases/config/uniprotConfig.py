###### UniProt Database ########
uniprot_id_url = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"
uniprot_id_file = "/Users/albertosantos/Development/UniProt/HUMAN_9606_idmapping.dat"
uniprot_text_file = "/Users/albertosantos/Development/UniProt/uniprot-human.tab" #### Downloaded manually from UniProt until we know url (organism:human AND reviewed:yes)
uniprot_variant_file = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/homo_sapiens_variation.txt.gz"
uniprot_unique_peptides_file = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/proteomics_mapping/UP000005640_9606_uniquePeptides.tsv"
uniprot_ids = ["UniProtKB-ID", 
                "NCBI_TaxID", 
                "Gene_Name", 
                "RefSeq", 
                "PDB", 
                "STRING", 
                "KEGG", 
                "Reactome", 
                "HPA", 
                "ChEMBL", 
                "Ensembl"]
uniprot_synonyms = ["UniProtKB-ID", 
                    "Gene_Name", 
                    "STRING", 
                    "HPA", 
                    "Ensembl", 
                    "ChEMBL", 
                    "PDB"]
uniprot_protein_relationships = {"RefSeq": ("Transcript", "TRANSLATED_INTO"), 
                                "Gene_Name":("Gene","TRANSLATED_INTO")
                                }
proteins_header = ['ID', ':LABEL', 'accession','name', 'synonyms', 'description', 'taxid']
variants_header = ['ID', ':LABEL', 'alternative_names']
relationships_header = ['START_ID', 'END_ID','TYPE', 'source']
