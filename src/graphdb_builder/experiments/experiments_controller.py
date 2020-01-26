import sys
import re
import os.path
import pandas as pd
import numpy as np
from collections import defaultdict
from graphdb_builder import builder_utils
from graphdb_builder.experiments.parsers import clinicalParser, proteomicsParser, wesParser
import config.ckg_config as ckg_config
import ckg_utils
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="experiments_controller")

def generate_dataset_imports(projectId, dataType, dataset_import_dir):
    stats = set()
    builder_utils.checkDirectory(dataset_import_dir)
    try:
        if dataType == 'clinical':
            data = clinicalParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType == "proteomics":
            data = proteomicsParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType == "wes":
            data = wesParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        else:
            raise Exception("Error when importing experiment for project {}. Non-existing parser for data type {}".format(projectId, dataType))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Experiment {}: {} file: {}, line: {}".format(projectId, sys.exc_info(), fname, exc_tb.tb_lineno))
        raise Exception("Error when importing experiment {}.\n {}".format(projectId, err))

def generate_graph_files(data, dataType, projectId, stats, ot = 'w', dataset_import_dir='experiments'):
    if dataType.lower() == '':
        outputfile = os.path.join(dataset_import_dir, projectId+".tsv")
    else:
        outputfile = os.path.join(dataset_import_dir, projectId+"_"+dataType.lower()+".tsv")
    
    with open(outputfile, ot) as f:
        data.to_csv(path_or_buf = f, sep='\t',
            header=True, index=False, quotechar='"',
            line_terminator='\n', escapechar='\\')
    
    logger.info("Experiment {} - Number of {} relationships: {}".format(projectId, dataType, data.shape[0]))
    stats.add(builder_utils.buildStats(data.shape[0], "relationships", dataType, "Experiment", outputfile))


if __name__ == "__main__":
    pass
