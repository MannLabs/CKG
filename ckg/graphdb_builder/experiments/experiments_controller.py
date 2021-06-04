import sys
import os.path
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder.experiments.parsers import clinicalParser, proteomicsParser, wesParser
from ckg import ckg_utils

ckg_config = ckg_utils.read_ckg_config()
log_config = ckg_config['graphdb_builder_log']
logger = builder_utils.setup_logging(log_config, key="experiments_controller")


def generate_dataset_imports(projectId, dataType, dataset_import_dir):
    stats = set()
    builder_utils.checkDirectory(dataset_import_dir)
    try:
        if dataType in ['project', 'experimental_design', 'clinical']:
            data = clinicalParser.parser(projectId, dataType)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)], dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType in ["proteomics", "interactomics", "phosphoproteomics"]:
            data = proteomicsParser.parser(projectId, dataType)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)], dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType == "wes":
            data = wesParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)], dtype, projectId, stats, ot, dataset_import_dir)
        else:
            raise Exception("Error when importing experiment for project {}. Non-existing parser for data type {}".format(projectId, dataType))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Error: {}. Experiment {}: {} file: {}, line: {}".format(err, projectId, sys.exc_info(), fname, exc_tb.tb_lineno))
        raise Exception("Error {}. Importing experiment {}. Data type {}.".format(err, projectId, dataType))


def generate_graph_files(data, dataType, projectId, stats, ot='w', dataset_import_dir='experiments'):
    if dataType == '':
        outputfile = os.path.join(dataset_import_dir, projectId+".tsv")
    else:
        outputfile = os.path.join(dataset_import_dir, projectId+"_"+dataType.lower()+".tsv")

    with open(outputfile, ot, encoding="utf-8") as f:
        data.to_csv(path_or_buf=f, sep='\t',
                    header=True, index=False, quotechar='"',
                    line_terminator='\n', escapechar='\\')

    logger.info("Experiment {} - Number of {} relationships: {}".format(projectId, dataType, data.shape[0]))
    stats.add(builder_utils.buildStats(data.shape[0], "relationships", dataType, "Experiment", outputfile))


if __name__ == "__main__":
    pass
