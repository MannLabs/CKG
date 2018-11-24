"""
    **import.py**
    Generates all the import files: Ontologies, Databases and Experiments.
    The module is reponsible for generating all the csv files that will
    be loaded into the Graph database and also updates a stats object
    (hdf table) with the number of entities and relationships from each
    dataset imported. A new stats object is created the first time a
    full import is run.
"""
import os.path
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_builder.ontologies import ontologies_controller as oh
from graphdb_builder.databases import databases_controller as dh
from graphdb_builder.experiments import experiments_controller as eh
from graphdb_builder import builder_utils
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="importer")

try:
    config = ckg_utils.get_configuration(ckg_config.builder_config_file)
    oconfig = ckg_utils.get_configuration(ckg_config.ontologies_config_file)
    dbconfig = ckg_utils.get_configuration(ckg_config.databases_config_file)
    econfig = ckg_utils.get_configuration(ckg_config.experiments_config_file)
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

START_TIME = datetime.now()

def ontologiesImport(importDirectory, ontologies=None, import_type="partial"):
    """
    Generates all the entities and relationships
    from the provided ontologies. If the ontologies list is
    not provided, then all the ontologies listed in the configuration
    will be imported (full_import).
    This function also updates the stats object with numbers from the
    imported ontologies
    Args:
        importDirectory (string): path of the import directory where
                                files will be created
        ontologies (list): A list of ontology names to be imported
        import_type: type of import (full or partial)
    """
    #Ontologies
    ontologiesImportDirectory = os.path.join(importDirectory, oconfig["ontologiesImportDir"])
    builder_utils.checkDirectory(ontologiesImportDirectory)
    stats = oh.generateGraphFiles(ontologiesImportDirectory, ontologies)
    statsDf = generateStatsDataFrame(stats)
    writeStats(statsDf, import_type)

def databasesImport(importDirectory, databases=None, n_jobs=1, download=True, import_type="partial"):
    """
    Generates all the entities and relationships
    from the provided databases. If the databases list is
    not provided, then all the databases listed in the configuration
    will be imported (full_import).
    This function also updates the stats object with numbers from the
    imported databases.
    Args:
        importDirectory (string): path of the import directory where
                                files will be created
        databases (list): A list of database names to be imported
        n_jobs (int): Number of jobs to run in parallel. 1 by default
                    when updating one database
        import_type: type of import (full or partial)
    """
    #Databases
    databasesImportDirectory = os.path.join(importDirectory, dbconfig["databasesImportDir"])
    builder_utils.checkDirectory(databasesImportDirectory)
    stats = dh.generateGraphFiles(databasesImportDirectory, databases, download, n_jobs)
    statsDf = generateStatsDataFrame(stats)
    writeStats(statsDf, import_type)

def experimentsImport(projects=None, n_jobs=1, import_type="partial"):
    """
    Generates all the entities and relationships
    from the specified Projects. If the projects list is
    not provided, then all the projects the experiments directory
    will be imported (full_import). Calls function experimentImport.
    Args:
        projects (list): A list of project identifiers to be imported
        n_jobs (int): Number of jobs to run in parallel. 1 by default
                    when updating one project
        import_type: type of import (full or partial)
    """
    #Experiments
    experimentsImportDirectory = econfig["experimentsImportDirectory"]
    builder_utils.checkDirectory(experimentsImportDirectory)
    experimentsDirectory = econfig["experimentsDir"]
    if projects is None:
        projects = builder_utils.listDirectoryFolders(experimentsDirectory)
    Parallel(n_jobs=n_jobs)(delayed(experimentImport)(experimentsImportDirectory, experimentsDirectory, project) for project in projects)

def experimentImport(importDirectory, experimentsDirectory, project):
    """
    Generates all the entities and relationships
    from the specified Project. Called from function experimentsImport.
    Args:
        importDirectory (string): path to the directory where all the import
                        files are generated
        experimentDirectory (string): path to the directory where all the
                        experiments are located
        project (string): Identifier of the project to be imported
    """
    projectPath = os.path.join(importDirectory, project)
    builder_utils.checkDirectory(projectPath)
    projectDirectory = os.path.join(experimentsDirectory, project)
    datasets = builder_utils.listDirectoryFolders(projectDirectory)
    for dataset in datasets:
        datasetPath = os.path.join(projectPath, dataset)
        builder_utils.checkDirectory(datasetPath)
        eh.generateDatasetImports(project, dataset)

def fullImport():
    """
    Calls the different importer functions: Ontologies, databases,
    experiments. The first step is to check if the stats object exists
    and create it otherwise. Calls setupStats.
    """
    try:
        download = config["download"]
        importDirectory = config["importDirectory"]
        builder_utils.checkDirectory(importDirectory)
        setupStats(import_type='full')
        logger.info("Full import: importing all Ontologies")
        ontologiesImport(importDirectory, import_type='full')
        logger.info("Full import: Ontologies import took {}".format(datetime.now() - START_TIME))
        logger.info("Full import: importing all Databases")
        databasesImport(importDirectory, n_jobs=4, download=download, import_type='full')
        logger.info("Full import: Databases import took {}".format(datetime.now() - START_TIME))
        logger.info("Full import: importing all Experiments")
        experimentsImport(n_jobs=4, import_type='full')
        logger.info("Full import: Experiments import took {}".format(datetime.now() - START_TIME))
    except EOFError as err:
        logger.error("Full import > {}.".format(err))
    except IOError as err:
        logger.error("Full import > {}.".format(err))
    except IndexError as err:
        logger.error("Full import > {}.".format(err)) 
    except KeyError as err:
        logger.error("Full import > {}.".format(err))
    except MemoryError as err:
        logger.error("Full import > {}.".format(err))
    except OSError as err:
        logger.error("Full import > {}.".format(err))
    except FileNotFoundError as err:
        logger.error("Full import > {}.".format(err))
    except Exception as err:
        logger.error("Full import > {}.".format(err))

def generateStatsDataFrame(stats):
    """
    Generates a dataframe with the stats from each import.
    Args:
        stats (list): A list with statistics collected from each importer
                        function
    Returns:
        statsDf: pandas dataframe with the collected statistics
    """
    statsDf = pd.DataFrame.from_records(list(stats), columns=config["statsCols"])
    
    return statsDf

def setupStats(import_type):
    """
    Creates a stats object that will collect all the statistics collected from
    each import.
    """
    statsDirectory = config["statsDirectory"]
    statsFile = os.path.join(statsDirectory, config["statsFile"])
    statsCols = config["statsCols"]
    statsName = getStatsName(import_type)
    try:
        if not os.path.exists(statsDirectory) or not os.path.isfile(statsFile):
            if not os.path.exists(statsDirectory):
                os.makedirs(statsDirectory)
            createEmptyStats(statsCols, statsFile, statsName)
    except Exception as err:
        logger.error("Setting up Stats object {} in file:{} > {}.".format(statsName, statsFile, err))

def createEmptyStats(statsCols, statsFile, statsName):
    """
    Creates a HDFStore object with a empty dataframe with the collected stats columns.
    Args:
        statsCols (list): A list of columns with the fields collected from the
                            import statistics
        statsFile (string): path where the object should be stored
        statsName (string): name if the file containing the stats object
    """
    try:
        statsDf = pd.DataFrame(columns=statsCols)
        hdf = pd.HDFStore(statsFile)
        hdf.put(statsName, statsDf, format='table', data_columns=True, min_itemsize=2000)
        hdf.close()
    except Exception as err:
        logger.error("Creating empty Stats object {} in file:{} > {}.".format(statsName, statsFile, err))

def loadStats(statsFile):
    """
    Loads the statistics object.
    Args:
        statsFile (string): File path where the stats object is stored.
    Returns:
        hdf (HDFStore object): object with the collected statistics.
                                stats can be accessed using a key
                                (i.e stats_ version)
    """
    try:
        hdf = None
        if os.path.isfile(statsFile):
            hdf = pd.HDFStore(statsFile)
    except Exception as err:
        logger.error("Loading Stats object {} in file:{} > {}.".format(stats_name, statsFile, err))

    return hdf

def writeStats(statsDf, import_type, stats_name=None):
    """
    Appends the new collected statistics to the existing stats object.
    Args:
        statsDf (dataframe): A pandas dataframe with the new statistics
                            from the importing.
        statsName (string): If the statistics should be stored with a
                            specific name
    """
    stats_directory = config["statsDirectory"]
    stats_file = os.path.join(stats_directory, config["statsFile"])
    try: 
        if stats_name is None:
            stats_name = getStatsName(import_type)
        hdf = loadStats(stats_file)
        hdf.append(stats_name, statsDf, min_itemsize=2000)
        hdf.close()
    except Exception as err:
        logger.error("Writing Stats object {} in file:{} > {}.".format(stats_name, stats_file, err))

def getStatsName(import_type):
    """
    Generates the stats object name where to store the importing
    statistics from the CKG version, which is defined in the configuration.
    Returns:
        statsName (string): key used to store in the stats object.
    """
    version = ckg_config.version
    statsName = import_type+'_stats_'+ str(version).replace('.', '_')

    return statsName


if __name__ == "__main__":
    fullImport()
    #experimentsImport(projects=["P0000003"], n_jobs=1)
