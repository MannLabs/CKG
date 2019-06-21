import urllib
from graphdb_builder import mapping as mp, builder_utils
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_builder.ontologies.parsers import *
import os.path
from collections import defaultdict
import pandas as pd
import csv
import obonet
import re
import config.ckg_config as graph_config
import logging
import logging.config
import sys

log_config = graph_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="ontologies_controller")

try:
    config = builder_utils.setup_config('ontologies')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

#########################
# General functionality #
#########################

def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]

############################
# Calling the right parser #
############################
def parse_ontology(ontology, download=True):
    directory = config["ontologies_directory"]
    ontology_directory = os.path.join(directory, ontology)
    builder_utils.checkDirectory(ontology_directory)
    ontology_files = []
    if ontology in config["ontology_types"]:
        otype = config["ontology_types"][ontology]
        if download:
            if 'urls' in config:
                if otype in config['urls']:
                    urls = config['urls'][otype]
                    for url in urls:
                        f = url.split('/')[-1]
                        ontology_files.append(os.path.join(ontology_directory, f))
                        builder_utils.downloadDB(url, directory=ontology_directory)
                elif otype in config["files"]:
                    ofiles = config["files"][otype]
                    ###Check SNOMED-CT files exist
                    for f in ofiles:
                        if os.path.isfile(os.path.join(directory, f)):
                            ontology_files.append(os.path.join(directory, f))
        filters = None
        if otype in config["parser_filters"]:
            filters = config["parser_filters"][otype]
    if len(ontology_files) > 0:
        if ontology == "SNOMED-CT":
            ontologyData = snomedParser.parser(ontology_files, filters)
        elif ontology == "ICD":
            ontologyData = icdParser.parser(ontology_files)
        else:
            ontologyData = oboParser.parser(ontology, ontology_files)
            mp.buildMappingFromOBO(ontology_files[0], ontology)
    else:
        if ontology == "SNOMED-CT":
            logger.info("WARNING: SNOMED-CT terminology needs to be downloaded manually since it requires UMLS License. More information available here: https://www.nlm.nih.gov/databases/umls.html")
        else:
            logger.info("WARNING: Ontology {} could not be downloaded. Check that the link in configuration works.".format(ontology))
    return ontologyData

#########################
#       Graph files     #
#########################
def generate_graphFiles(import_directory, ontologies=None, download=True):
    entities = config["ontologies"]
    if ontologies is not None:
        for ontology in ontologies:
            entities = {ontology:ontologies[ontology]}

    stats = set()
    for entity in entities:
        ontology = config["ontologies"][entity]
        if ontology in config["ontology_types"]:
            ontologyType = config["ontology_types"][ontology]
        try:
            terms, relationships, definitions = parse_ontology(ontology, download)
            for namespace in terms:
                if namespace in config["entities"]:
                    name = config["entities"][namespace]
                entity_outputfile = os.path.join(import_directory, name+".tsv")
                with open(entity_outputfile, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(['ID', ':LABEL', 'name', 'description', 'type', 'synonyms'])
                    for term in terms[namespace]:
                        writer.writerow([term, entity, list(terms[namespace][term])[0], definitions[term], ontologyType, ",".join(terms[namespace][term])])
                logger.info("Ontology {} - Number of {} entities: {}".format(ontology, name, len(terms[namespace])))
                stats.add(builder_utils.buildStats(len(terms[namespace]), "entity", name, ontology, entity_outputfile))
                if namespace in relationships:
                    relationships_outputfile = os.path.join(import_directory, name+"_has_parent.tsv")
                    relationshipsDf = pd.DataFrame(list(relationships[namespace]))
                    relationshipsDf.columns = ['START_ID', 'END_ID', 'TYPE']
                    relationshipsDf.to_csv(path_or_buf=relationships_outputfile,
                                                sep='\t',
                                                header=True, index=False, quotechar='"',
                                                quoting=csv.QUOTE_ALL,
                                                line_terminator='\n', escapechar='\\')
                    logger.info("Ontology {} - Number of {} relationships: {}".format(ontology, name+"_has_parent", len(relationships[namespace])))
                    stats.add(builder_utils.buildStats(len(relationships[namespace]), "relationships", name+"_has_parent", ontology, relationships_outputfile))
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Ontology {}: {}, file: {},line: {}".format(ontology, sys.exc_info(), fname, exc_tb.tb_lineno))
            raise Exception("Error when importing ontology {}.\n {}".format(ontology, err))
    return stats

if __name__ == "__main__":
    generate_graphFiles(import_directory, download=True) 
