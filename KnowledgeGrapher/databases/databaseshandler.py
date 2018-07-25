import os.path
import gzip
import databases_config as dbconfig
from collections import defaultdict
from KnowledgeGrapher import utils
import csv
import pandas as pd
import re
from lxml import etree
import zipfile
from parsers import *

#########################
# General functionality # 
#########################
def write_relationships(relationships, header, outputfile):
    df = pd.DataFrame(list(relationships))
    df.columns = header 
    df.to_csv(path_or_buf=outputfile, 
                header=True, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')

def write_entities(entities, header, outputfile):
    with open(outputfile, 'w') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for entity in entities:
            writer.writerow(entity)

#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory):
    mapping = getMapping()
    string_mapping = getSTRINGMapping()
    databases = dbconfig.databases
    for database in databases:
        print(database)
        if database.lower() == "internal":
            for qtype in internalDBsConfig.internal_db_types:
                relationships, entity1, entity2 = parseInternalDatabasePairs(qtype, string_mapping)
                entity1, entity2 = internalDBsConfig.internal_db_types[qtype]
                outputfile = os.path.join(importDirectory, entity1+"_"+entity2+"_associated_with_integrated.csv")
                header = ["START_ID", "END_ID", "TYPE", "source", "score"]
                write_relationships(relationships, header, outputfile)
        if database.lower() == "mentions":
            entities, header = parsePMClist()
            publications_outputfile = os.path.join(importDirectory, "Publications.csv")
            write_entities(entities, header, publications_outputfile)
            for qtype in internalDBsConfig.internal_db_mentions_types:
                parseInternalDatabaseMentions(qtype, mapping, importDirectory)
        if database.lower() == "hgnc":
            #HGNC
            genes, relationships = parser()
            genes_outputfile = os.path.join(importDirectory, "Gene.csv")
            header = ['ID', ':LABEL', 'name', 'family', 'synonyms', 'taxid']
            write_entities(genes, header, genes_outputfile)
        if database.lower() == "refseq":
            #RefSeq
            headers = refseqConfig.headerEntities
            entities, relationships = parser()
            for entity in entities:
                header = headers[entity]
                outputfile = os.path.join(importDirectory, entity+".csv")
                write_entities(entity, header, outputfile)
            for rel in relationships:
                header = headers[rel]
                outputfile = os.path.join(importDirectory, "refseq_"+rel.lower()+".csv")
                write_relationships(relationships[rel], header, outputfile)
        if database.lower() == "uniprot":
            #UniProt
            uniprot_id_file = uniprotConfig.uniprot_id_file
            uniprot_texts_file = uniprotConfig.uniprot_text_file
            proteins, relationships = parseUniProtDatabase(uniprot_id_file)
            addUniProtTexts(uniprot_texts_file, proteins)
            proteins_outputfile = os.path.join(importDirectory, "Protein.csv")
            with open(proteins_outputfile, 'w') as csvfile:
                writer = csv.writer(csvfile, escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(['ID', ':LABEL', 'accession','name', 'synonyms', 'description', 'taxid'])
                for protein in proteins:
                    accession = ""
                    name = ""
                    synonyms = []
                    taxid = 9606
                    description = ""
                    if "UniProtKB-ID" in proteins[protein]:
                        accession  = proteins[protein]["UniProtKB-ID"]
                    if "Gene_Name" in proteins[protein]:
                        name = proteins[protein]["Gene_Name"]
                    if "synonyms" in proteins[protein]:
                        synonyms = proteins[protein]["synonyms"]
                    if "NCBI_TaxID" in proteins[protein]:
                        taxid = int(proteins[protein]["NCBI_TaxID"])
                    if "description" in proteins[protein]:
                        description = proteins[protein]["description"]
                    writer.writerow([protein, "Protein", accession , name, ",".join(synonyms), description, taxid])

            for entity, rel in relationships:
                outputfile = os.path.join(importDirectory, "uniprot_"+entity.lower()+"_"+rel.lower()+".csv")
                header = ['START_ID', 'END_ID','TYPE', 'source']
                write_relationships(relationships[(entity,rel)], header, outputfile)
            #Variants
            entities, relationships = parseUniProtVariants()
            variants_outputfile = os.path.join(importDirectory, "Known_variant.csv")
            header = ['ID', ':LABEL', 'alternative_names']
            write_entities(entities, header, variants_outputfile)
            for relationship in relationships:
                outputfile = os.path.join(importDirectory, relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE']
                write_relationships(relationships[relationship], header, outputfile)
        if database.lower() == "intact":
            #IntAct
            intact_file = os.path.join(dbconfig.databasesDir,intactConfig.intact_file)
            interactions = parser(intact_file, proteins)
            interactions_outputfile = os.path.join(importDirectory, "INTACT_interacts_with.csv")
            header = ['START_ID', 'END_ID','TYPE', 'score', 'interaction_type', 'method', 'source', 'publications']
            write_relationships(interactions, header, interactions_outputfile)
        if database.lower() == "string":
            #STRING
            interactions = parser(string_mapping)
            interactions_outputfile = os.path.join(importDirectory, "STRING_interacts_with.csv")
            header = ['START_ID', 'END_ID','TYPE', 'interaction_type', 'source', 'evidences','scores', 'score']
            write_relationships(interactions, header, interactions_outputfile)
        if database.lower() == "stitch":
            #STITCH
            evidences = ["experimental", "prediction", "database","textmining", "score"]
            interactions = parser(string_mapping, db = "STITCH")
            interactions_outputfile = os.path.join(importDirectory, "STITCH_associated_with.csv")
            header = ['START_ID', 'END_ID','TYPE', 'interaction_type', 'source', 'evidences','scores', 'score']
            write_relationships(interactions, header, interactions_outputfile)
        if database.lower() == "disgenet":
            #DisGeNet
            disease_relationships = parseDisGeNetDatabase()

            for idType in disease_relationships:
                disease_outputfile = os.path.join(importDirectory, "disgenet_associated_with.csv")
                header = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications']
                write_relationships(disease_relationships[idType], header, disease_outputfile)

        if database.lower() == "pathwaycommons":
            #PathwayCommons pathways
            entities, relationships = parser()
            entity_outputfile = os.path.join(importDirectory, "Pathway.csv")
            header = ['ID', ':LABEL', 'name', 'description', 'type', 'source', 'linkout']
            write_entities(entities, header, entity_outputfile)
            pathway_outputfile = os.path.join(importDirectory, "pathwaycommons_protein_associated_with_pathway.csv")
            header = ['START_ID', 'END_ID','TYPE', 'evidence', 'linkout', 'source']
            write_relationships(relationships, header, pathway_outputfile)
        
        if database.lower() == "dgidb":
            relationships = parser()
            dgidb_outputfile = os.path.join(importDirectory, "dgidb_targets.csv")
            header = ['START_ID', 'END_ID','TYPE', 'type', 'source']
            write_relationships(relationships, header, dgidb_outputfile)

        if database.lower() == "sider":
            relationships = parser()
            sider_outputfile = os.path.join(importDirectory, "sider_has_side_effect.csv")
            header = ['START_ID', 'END_ID','TYPE', 'source', 'original_side_effect']
            write_relationships(relationships, header, sider_outputfile)

        if database.lower() == "oncokb":
            entities, relationships = parser(mapping = mapping)
            entity_outputfile = os.path.join(importDirectory, "oncokb_Clinically_relevant_variant.csv")
            header = ['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity']
            write_entities(entities, header, entity_outputfile)
            for relationship in relationships:
                oncokb_outputfile = os.path.join(importDirectory, "oncokb_"+relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE']
                if relationship == "targets_clinically_relevant_variant":
                    header = ['START_ID', 'END_ID','TYPE', 'association', 'evidence', 'tumor', 'type', 'source']
                elif relationship == "targets":
                    header = ['START_ID', 'END_ID','TYPE', 'type', 'source']
                elif relationship == "associated_with":
                    header = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications'] 
                elif relationship == "known_variant_is_clinically_relevant":
                    header = ['START_ID', 'END_ID','TYPE', 'source']
                write_relationships(relationships[relationship], header, oncokb_outputfile)
        
        if database.lower() == "cancergenomeinterpreter":
            entities, relationships = parser(mapping = mapping)
            entity_outputfile = os.path.join(importDirectory, "cgi_Clinically_relevant_variant.csv")
            header = ['ID', ':LABEL', 'alternative_names', 'chromosome', 'position', 'reference', 'alternative', 'effect', 'oncogeneicity']
            write_entities(entities, header, entity_outputfile)
            for relationship in relationships:
                cgi_outputfile = os.path.join(importDirectory, "cgi_"+relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE']
                if relationship == "targets_clinically_relevant_variant":
                    header = ['START_ID', 'END_ID','TYPE', 'evidence', 'association', 'tumor', 'type', 'source']
                elif relationship == "targets":
                    header = ['START_ID', 'END_ID','TYPE', 'type', 'source']
                elif relationship == "associated_with":
                    header = ['START_ID', 'END_ID','TYPE', 'score', 'evidence_type', 'source', 'number_publications']
                elif relationship == "known_variant_is_clinically_relevant":
                    header = ['START_ID', 'END_ID','TYPE', 'source']
                write_relationships(relationships[relationship], header, cgi_outputfile)

        if database.lower() == "hmdb":
            metabolites = parser()
            entities, attributes =  build_metabolite_entity(metabolites)
            relationships = build_relationships_from_HMDB(metabolites, mapping)
            entity_outputfile = os.path.join(importDirectory, "Metabolite.csv")
            header = ['ID'] + attributes
            write_entities(entities, header, entity_outputfile)
            
            for relationship in relationships:
                hmdb_outputfile = os.path.join(importDirectory, relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE', 'source']
                write_relationships(relationships[relationship], header, hmdb_outputfile)

        if database.lower() == "drugbank":
            drugs = parser()
            relationships = build_relationships_from_DrugBank(drugs)
            entities, attributes = build_metabolite_entity(drugs)
            entity_outputfile = os.path.join(importDirectory, "Drug.csv")
            header = ['ID'] + attributes
            write_entities(entities, header, entity_outputfile)
            
            for relationship in relationships:
                relationship_outputfile = os.path.join(importDirectory, relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE', 'source']
                if relationship == "drugbank_interacts_with_drug":
                    header = ['START_ID', 'END_ID','TYPE', 'interaction_description', 'source']
                write_relationships(relationships[relationship], header, relationship_outputfile)

        if database.lower() == "gwascatalog":
            entities, relationships = parser()
            entity_outputfile = os.path.join(importDirectory, "GWAS_study.csv")
            header = ['ID', 'TYPE', 'title', 'date', 'sample_size', 'replication_size', 'trait'] 
            write_entities(entities, header, entity_outputfile)
            for relationship in relationships:
                outputfile = os.path.join(importDirectory, "GWAS_study_"+relationship+".csv")
                header = ['START_ID', 'END_ID','TYPE', 'source']
                write_relationships(relationships[relationship], header, outputfile)


if __name__ == "__main__":
    hmdbParser.parseHMDB()
