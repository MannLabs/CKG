import os.path
import sys
import gzip
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_builder import builder_utils
from collections import defaultdict
import pandas as pd
import re
from lxml import etree
import zipfile
from graphdb_builder.databases.parsers import *
from joblib import Parallel, delayed
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="database_controller")

try:
    dbconfig = builder_utils.setup_config('databases')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

def parseDatabase(importDirectory, database, download=True):
    stats = set()
    try:
        logger.info("Parsing database {}".format(database))
        if database.lower() == "internal":
            result = internalDBsParser.parser(dbconfig["databasesDir"], download)
            for qtype in result:
                relationships, header, outputfileName = result[qtype]
                outputfile = os.path.join(importDirectory, outputfileName)
                builder_utils.write_relationships(relationships, header, outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, qtype, len(relationships)))
                stats.add(builder_utils.buildStats(len(relationships), "relationships", qtype, database, outputfile))
        elif database.lower() == "mentions":
            entities, header, outputfileName = internalDBsParser.parserMentions(dbconfig["databasesDir"], importDirectory, download)
            outputfile = os.path.join(importDirectory, outputfileName)
            builder_utils.write_entities(entities, header, outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Publication", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Publication", database, outputfile))
        elif database.lower() == "hgnc":
            #HGNC
            entities, header = hgncParser.parser(dbconfig["databasesDir"], download)
            outputfile = os.path.join(importDirectory, "Gene.tsv")            
            builder_utils.write_entities(entities, header, outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Gene", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Gene", database, outputfile))
        elif database.lower() == "refseq":
            entities, relationships, headers = refseqParser.parser(dbconfig["databasesDir"], download)
            for entity in entities:
                header = headers[entity]
                outputfile = os.path.join(importDirectory, entity+".tsv")
                builder_utils.write_entities(entities[entity], header, outputfile)
                logger.info("Database {} - Number of {} entities: {}".format(database, entity, len(entities[entity])))
                stats.add(builder_utils.buildStats(len(entities[entity]), "entity", entity, database, outputfile))
            for rel in relationships:
                header = headers[rel]
                outputfile = os.path.join(importDirectory, "refseq_"+rel.lower()+".tsv")
                builder_utils.write_relationships(relationships[rel], header, outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, rel, len(relationships[rel])))
                stats.add(builder_utils.buildStats(len(relationships[rel]), "relationships", rel, database, outputfile))
        elif database.lower() == "uniprot":
            #UniProt
            stats.update(uniprotParser.parser(dbconfig["databasesDir"], importDirectory, download))
        elif database.lower() == "pfam":
            #UniProt
            stats.update(pfamParser.parser(dbconfig["databasesDir"], importDirectory, download))
        elif database.lower() == "intact":
            #IntAct
            relationships, header, outputfileName = intactParser.parser(dbconfig["databasesDir"], download)
            outputfile = os.path.join(importDirectory, outputfileName)
            builder_utils.write_relationships(relationships, header, outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "curated_interacts_with", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "curated_interacts_with", database, outputfile))
        elif database.lower() == "mutationds":
            #MutationDs
            relationships, header, outputfileName = mutationDsParser.parser(dbconfig["databasesDir"], download)
            outputfile = os.path.join(importDirectory, outputfileName)
            builder_utils.write_relationships(relationships, header, outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "curated_affects_interaction_with", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "curated_affects_interaction_with", database, outputfile))
        elif database.lower() == "string":
            #STRING
            proteinMapping, drugMapping = stringParser.parser(dbconfig["databasesDir"], importDirectory, download=download)
            stringParser.parseActions(dbconfig["databasesDir"], importDirectory, proteinMapping, drugMapping, download = download, db="STRING")
        elif database.lower() == "stitch":
            #STITCH
            proteinMapping, drugMapping = stringParser.parser(dbconfig["databasesDir"], importDirectory, drug_source=dbconfig["sources"]["Drug"], download=download, db="STITCH")
            stringParser.parseActions(dbconfig["databasesDir"], importDirectory, proteinMapping, drugMapping, download = download, db="STITCH")
        elif database.lower() == "disgenet":
            #DisGeNet
            relationships, header, outputfileName = disgenetParser.parser(dbconfig["databasesDir"], download)
            for idType in relationships:
                outputfile = os.path.join(importDirectory, idType+"_"+outputfileName)
                builder_utils.write_relationships(relationships[idType], header, outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, idType, len(relationships[idType])))
                stats.add(builder_utils.buildStats(len(relationships[idType]), "relationships", idType, database, outputfile))
        elif database.lower() == "pathwaycommons":
            #PathwayCommons pathways
            entities, relationships, entities_header, relationships_header = pathwayCommonsParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "Pathway.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            stats.add(builder_utils.buildStats(len(entities), "entity", "Pathway", database, entity_outputfile))
            pathway_outputfile = os.path.join(importDirectory, "pathwaycommons_protein_associated_with_pathway.tsv")
            builder_utils.write_relationships(relationships, relationships_header, pathway_outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "protein_associated_with_pathway", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "protein_associated_with_pathway", database, pathway_outputfile))
        elif database.lower() == "reactome":
            #Reactome
            entities, relationships, entities_header, relationships_header = reactomeParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, database.lower()+"_Pathway.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            stats.add(builder_utils.buildStats(len(entities), "entity", "Pathway", database, entity_outputfile))
            for entity,relationship in relationships:
                reactome_outputfile = os.path.join(importDirectory, database.lower()+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity, relationship)], relationships_header[entity], reactome_outputfile)
                logger.info("Database {} - Number of {} {} relationships: {}".format(database, entity, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, reactome_outputfile))
        elif database.lower() == "smpdb":
            #SMPDB
            entities, relationships, entities_header, relationships_header = smpdbParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, database.lower()+"_Pathway.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            stats.add(builder_utils.buildStats(len(entities), "entity", "Pathway", database, entity_outputfile))
            for entity,relationship in relationships:
                smpdb_outputfile = os.path.join(importDirectory, database+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity, relationship)], relationships_header[entity], smpdb_outputfile)
                logger.info("Database {} - Number of {} {} relationships: {}".format(database, entity, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, smpdb_outputfile))
        elif database.lower() == "dgidb":
            relationships, header, outputfileName = drugGeneInteractionDBParser.parser(dbconfig["databasesDir"], download)
            outputfile = os.path.join(importDirectory, outputfileName)           
            builder_utils.write_relationships(relationships, header, outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "targets", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "targets", database, outputfile))
        elif database.lower() == "sider":
            relationships,header, outputfileName, drugMapping, phenotypeMapping = siderParser.parser(dbconfig["databasesDir"], dbconfig["sources"]["Drug"], download)
            outputfile = os.path.join(importDirectory, outputfileName)
            builder_utils.write_relationships(relationships, header, outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "has_side_effect", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "has_side_effect", database, outputfile))
            relationships, header, outputfileName = siderParser.parserIndications(dbconfig["databasesDir"], drugMapping, phenotypeMapping, download = True)
            outputfile = os.path.join(importDirectory, outputfileName)
            builder_utils.write_relationships(relationships, header, outputfile)
            logger.info("Database {} - Number of {} relationships: {}".format(database, "indicated_for", len(relationships)))
            stats.add(builder_utils.buildStats(len(relationships), "relationships", "indicated_for", database, outputfile))
        elif database.lower() == "oncokb":
            entities, relationships, entities_header,  relationships_headers = oncokbParser.parser(dbconfig["databasesDir"], download)
            outputfile = os.path.join(importDirectory, "oncokb_Clinically_relevant_variant.tsv")
            builder_utils.write_entities(entities, entities_header, outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Clinically_relevant_variant", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Clinically_relevant_variant", database, outputfile))
            for relationship in relationships:
                oncokb_outputfile = os.path.join(importDirectory, "oncokb_"+relationship+".tsv")
                if relationship in relationships_headers:
                    header = relationships_headers[relationship]
                else:
                    header = ['START_ID', 'END_ID','TYPE']
                builder_utils.write_relationships(relationships[relationship], header, oncokb_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[relationship])))
                stats.add(builder_utils.buildStats(len(relationships[relationship]), "relationships", relationship, database, outputfile))
        elif database.lower() == "cancergenomeinterpreter":
            entities, relationships, entities_header, relationships_headers = cancerGenomeInterpreterParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "cgi_Clinically_relevant_variant.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Clinically_relevant_variant", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Clinically_relevant_variant", database, entity_outputfile))
            for relationship in relationships:
                cgi_outputfile = os.path.join(importDirectory, "cgi_"+relationship+".tsv")
                header = ['START_ID', 'END_ID','TYPE']
                if relationship in relationships_headers:
                    header = relationships_headers[relationship]
                builder_utils.write_relationships(relationships[relationship], header, cgi_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[relationship])))
                stats.add(builder_utils.buildStats(len(relationships[relationship]), "relationships", relationship, database, cgi_outputfile))
        elif database.lower() == "hmdb":
            entities, relationships, entities_header, relationships_header = hmdbParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "Metabolite.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Metabolite", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Metabolite", database, entity_outputfile))
            for relationship in relationships:
                hmdb_outputfile = os.path.join(importDirectory, relationship+".tsv")
                builder_utils.write_relationships(relationships[relationship], relationships_header, hmdb_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[relationship])))
                stats.add(builder_utils.buildStats(len(relationships[relationship]), "relationships", relationship, database, hmdb_outputfile))
        elif database.lower() == "drugbank":
            entities, relationships, entities_header, relationships_headers = drugBankParser.parser(dbconfig["databasesDir"])
            entity_outputfile = os.path.join(importDirectory, "Drug.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Drug", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Drug", database, entity_outputfile))
            for relationship in relationships:
                relationship_outputfile = os.path.join(importDirectory, relationship+".tsv")
                header = ['START_ID', 'END_ID','TYPE', 'source']
                if relationship in relationships_headers:
                    header = relationships_headers[relationship]
                builder_utils.write_relationships(relationships[relationship], header, relationship_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[relationship])))
                stats.add(builder_utils.buildStats(len(relationships[relationship]), "relationships", relationship, database, relationship_outputfile))
        elif database.lower() == "gwascatalog":
            entities, relationships, entities_header, relationships_header = gwasCatalogParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "GWAS_study.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "GWAS_study", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "GWAS_study", database, entity_outputfile))
            for relationship in relationships:
                outputfile = os.path.join(importDirectory, "GWAS_study_"+relationship+".tsv")
                builder_utils.write_relationships(relationships[relationship], relationships_header, outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[relationship])))
                stats.add(builder_utils.buildStats(len(relationships[relationship]), "relationships", relationship, database, outputfile))
        elif database.lower() == "phosphositeplus":
            entities, relationships, entities_header, relationships_headers = pspParser.parser(dbconfig["databasesDir"])
            entity_outputfile = os.path.join(importDirectory, "Modified_protein.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Modified_protein", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Modified_protein", database, entity_outputfile))
            for entity,relationship in relationships:
                rel_header = ["START_ID", "END_ID", "TYPE", "source"]
                if entity in relationships_headers:
                    rel_header = relationships_headers[entity]
                outputfile = os.path.join(importDirectory, "psp_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity,relationship)], rel_header, outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, outputfile))
        elif database.lower() == "corum":
            entities, relationships, entities_header, relationships_headers = corumParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "Complex.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Complex", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Complex", database, entity_outputfile))
            for entity, relationship in relationships:
                corum_outputfile = os.path.join(importDirectory, database+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity,relationship)], relationships_headers[entity], corum_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, corum_outputfile))
        elif database.lower() == "foodb":
            entities, relationships, entities_header, relationships_headers = foodbParser.parser(dbconfig["databasesDir"], download)
            entity_outputfile = os.path.join(importDirectory, "Food.tsv")
            builder_utils.write_entities(entities, entities_header, entity_outputfile)
            logger.info("Database {} - Number of {} entities: {}".format(database, "Food", len(entities)))
            stats.add(builder_utils.buildStats(len(entities), "entity", "Food", database, entity_outputfile))
            for entity, relationship in relationships:
                foodb_outputfile = os.path.join(importDirectory, database.lower()+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity,relationship)], relationships_headers[entity], foodb_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, foodb_outputfile))
        elif database.lower() == "exposome explorer":
            relationships, header = exposomeParser.parser(dbconfig["databasesDir"], download)
            for entity, relationship in relationships:
                ee_outputfile = os.path.join(importDirectory, database.lower()+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity,relationship)], header[entity], ee_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, ee_outputfile))
        elif database.lower() == "hpa":
            relationships, headers = hpaParser.parser(dbconfig["databasesDir"], download)
            for entity, relationship in relationships:
                hpa_outputfile = os.path.join(importDirectory, database.lower()+"_"+entity.lower()+"_"+relationship.lower()+".tsv")
                builder_utils.write_relationships(relationships[(entity,relationship)], headers[relationship], hpa_outputfile)
                logger.info("Database {} - Number of {} relationships: {}".format(database, relationship, len(relationships[(entity,relationship)])))
                stats.add(builder_utils.buildStats(len(relationships[(entity,relationship)]), "relationships", relationship, database, hpa_outputfile))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Database {}: {}, file: {},line: {}".format(database, sys.exc_info(), fname, exc_tb.tb_lineno))
        #raise Exception("Error when importing database {}.\n {}".format(database, err))
    return stats
    

#########################
#       Graph files     # 
#########################
def generateGraphFiles(importDirectory, databases=None, download=True, n_jobs = 4):
    if databases is None:
        databases = dbconfig["databases"]
    stats = Parallel(n_jobs=n_jobs)(delayed(parseDatabase)(importDirectory,database, download) for database in databases)
    allstats = {val if type(sublist) == set else sublist for sublist in stats for val in sublist}
    return allstats


if __name__ == "__main__":
    pass
