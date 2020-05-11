"""
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
from uuid import uuid4
import config.ckg_config as ckg_config
from graphdb_builder.ontologies import ontologies_controller as oh
from graphdb_builder.databases import databases_controller as dh
from graphdb_builder.experiments import experiments_controller as eh
from graphdb_builder.users import users_controller as uh
from graphdb_builder import builder_utils

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="importer")
import_id = uuid4()

try:
    cwd = os.path.abspath(os.path.dirname(__file__))
    config = builder_utils.setup_config('builder')
    directories = builder_utils.get_full_path_directories()
    oconfig = builder_utils.setup_config('ontologies')
    dbconfig = builder_utils.setup_config('databases')
    econfig = builder_utils.setup_config('experiments')
    uconfig = builder_utils.setup_config('users')
except Exception as err:
    logger.error("importer - Reading configuration > {}.".format(err))

START_TIME = datetime.now()


def ontologiesImport(importDirectory, ontologies=None, download=True, import_type="partial"):
    """
    Generates all the entities and relationships from the provided ontologies. If the ontologies list is\
    not provided, then all the ontologies listed in the configuration will be imported (full_import). \
    This function also updates the stats object with numbers from the imported ontologies.

    :param str importDirectory: path of the import directory where files will be created.
    :param list ontologies: a list of ontology names to be imported.
    :param bool download: wether database is to be downloaded.
    :param str import_type: type of import (´full´ or ´partial´).
    """
    ontologiesImportDirectory = os.path.join(importDirectory, oconfig["ontologies_importDir"])
    builder_utils.checkDirectory(ontologiesImportDirectory)
    stats = oh.generate_graphFiles(ontologiesImportDirectory, ontologies, download)
    statsDf = generateStatsDataFrame(stats)
    setupStats(import_type=import_type)
    writeStats(statsDf, import_type)


def databasesImport(importDirectory, databases=None, n_jobs=1, download=True, import_type="partial"):
    """
    Generates all the entities and relationships from the provided databases. If the databases list is\
    not provided, then all the databases listed in the configuration will be imported (full_import).\
    This function also updates the stats object with numbers from the imported databases.

    :param str importDirectory: path of the import directory where files will be created.
    :param list databases: a list of database names to be imported.
    :param int n_jobs: number of jobs to run in parallel. 1 by default when updating one database.
    :param str import_type: type of import (´full´ or ´partial´).
    """
    databasesImportDirectory = os.path.join(importDirectory, dbconfig["databasesImportDir"])
    builder_utils.checkDirectory(databasesImportDirectory)
    stats = dh.generateGraphFiles(databasesImportDirectory, databases, download, n_jobs)
    statsDf = generateStatsDataFrame(stats)
    setupStats(import_type=import_type)
    writeStats(statsDf, import_type)


def experimentsImport(projects=None, n_jobs=1, import_type="partial"):
    """
    Generates all the entities and relationships from the specified Projects. If the projects list is\
    not provided, then all the projects the experiments directory will be imported (full_import). \
    Calls function experimentImport.

    :param list projects:  list of project identifiers to be imported.
    :param int n_jobs: number of jobs to run in parallel. 1 by default when updating one project.
    :param str import_type: type of import (´full´ or ´partial´).
    """
    experiments_import_directory = os.path.join(directories['importDirectory'], econfig["import_directory"])
    builder_utils.checkDirectory(experiments_import_directory)
    experiments_directory = os.path.join(directories['dataDirectory'], econfig["experiments_directory"])
    if projects is None:
        projects = builder_utils.listDirectoryFolders(experiments_directory)
    if len(projects) > 0:
        Parallel(n_jobs=n_jobs)(delayed(experimentImport)(experiments_import_directory, experiments_directory, project) for project in projects)


def experimentImport(importDirectory, experimentsDirectory, project):
    """
    Generates all the entities and relationships from the specified Project. Called from function experimentsImport.

    :param str importDirectory: path to the directory where all the import files are generated.
    :param str experimentDirectory: path to the directory where all the experiments are located.
    :param str project: identifier of the project to be imported.
    """
    projectPath = os.path.join(importDirectory, project)
    builder_utils.checkDirectory(projectPath)
    projectDirectory = os.path.join(experimentsDirectory, project)
    datasets = builder_utils.listDirectoryFolders(projectDirectory)
    for dataset in datasets:
        if dataset != "experimental_design":
            datasetPath = os.path.join(projectPath, dataset)
            builder_utils.checkDirectory(datasetPath)
            eh.generate_dataset_imports(project, dataset, datasetPath)


def usersImport(importDirectory, import_type='partial'):
    """
    Generates User entities from excel file and grants access of new users to the database.
    This function also writes the relevant information to a tab-delimited file in the import \
    directory.

    :param str importDirectory: path to the directory where all the import files are generated.
    :param str import_type: type of import (´full´ or ´partial).
    """
    usersImportDirectory = os.path.join(importDirectory, uconfig['usersImportDirectory'])
    builder_utils.checkDirectory(usersImportDirectory)
    uh.parseUsersFile(usersImportDirectory, expiration=365)


def fullImport(download=True, n_jobs=4):
    """
    Calls the different importer functions: Ontologies, databases, \
    experiments. The first step is to check if the stats object exists \
    and create it otherwise. Calls setupStats.
    """
    try:
        importDirectory = directories["importDirectory"]
        builder_utils.checkDirectory(importDirectory)
        setupStats(import_type='full')
        logger.info("Full import: importing all Ontologies")
        ontologiesImport(importDirectory, download=download, import_type='full')
        logger.info("Full import: Ontologies import took {}".format(datetime.now() - START_TIME))
        logger.info("Full import: importing all Databases")
        databasesImport(importDirectory, n_jobs=n_jobs, download=download, import_type='full')
        logger.info("Full import: Databases import took {}".format(datetime.now() - START_TIME))
        logger.info("Full import: importing all Experiments")
        experimentsImport(n_jobs=n_jobs, import_type='full')
        logger.info("Full import: Experiments import took {}".format(datetime.now() - START_TIME))
        logger.info("Full import: importing all Users")
        usersImport(importDirectory, import_type='full')
        logger.info("Full import: Users import took {}".format(datetime.now() - START_TIME))
    except FileNotFoundError as err:
        logger.error("Full import > {}.".format(err))
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
    except Exception as err:
        logger.error("Full import > {}.".format(err))


def generateStatsDataFrame(stats):
    """
    Generates a dataframe with the stats from each import.
    :param list stats: a list with statistics collected from each importer function.
    :return: Pandas dataframe with the collected statistics.
    """
    statsDf = pd.DataFrame.from_records(list(stats), columns=config["statsCols"])
    statsDf['import_id'] = import_id
    statsDf['import_id'] = statsDf['import_id'].astype('str')

    return statsDf


def setupStats(import_type):
    """
    Creates a stats object that will collect all the statistics collected from each import.
    """
    statsDirectory = directories["statsDirectory"]
    statsFile = os.path.join(statsDirectory, config["statsFile"])
    statsCols = config["statsCols"]
    statsName = getStatsName(import_type)
    try:
        if not os.path.exists(statsDirectory) or not os.path.isfile(statsFile):
            if not os.path.exists(statsDirectory):
                os.makedirs(statsDirectory)
            else:
                pass
            createEmptyStats(statsCols, statsFile, statsName)
        else:
            pass
    except Exception as err:
        logger.error("Setting up Stats object {} in file:{} > {}.".format(statsName, statsFile, err))


def createEmptyStats(statsCols, statsFile, statsName):
    """
    Creates a HDFStore object with a empty dataframe with the collected stats columns.

    :param list statsCols: a list of columns with the fields collected from the import statistics.
    :param str statsFile: path where the object should be stored.
    :param str statsName: name if the file containing the stats object.
    """
    try:
        statsDf = pd.DataFrame(columns=statsCols)
        with pd.HDFStore(statsFile) as hdf:
            hdf.put(statsName, statsDf, format='table', data_columns=True)
            hdf.close()
    except Exception as err:
        logger.error("Creating empty Stats object {} in file:{} > {}.".format(statsName, statsFile, err))

# def loadStats(statsFile):
#     """
#     Loads the statistics object.
    
#     :param str statsFile: file path where the stats object is stored.
#     :returns: HDFStore object with the collected statistics. \
#                 stats can be accessed using a key (i.e stats_ version).
#     """
#     try:
#         hdf = None
#         if os.path.isfile(statsFile):
#             hdf = pd.HDFStore(statsFile)
#     except Exception as err:
#         logger.error("Loading Stats file:{} > {}.".format(statsFile, err))

#     return hdf


def writeStats(statsDf, import_type, stats_name=None):
    """
    Appends the new collected statistics to the existing stats object.
    :param statsDf: a pandas dataframe with the new statistics from the importing.
    :param str statsName: If the statistics should be stored with a specific name.
    """
    stats_directory = directories["statsDirectory"]
    stats_file = os.path.join(stats_directory, config["statsFile"])
    try:
        if stats_name is None:
            stats_name = getStatsName(import_type)
        with pd.HDFStore(stats_file) as hdf:
            hdf.append(stats_name, statsDf, data_columns=True, min_itemsize={'time': 8})
    except Exception as err:
        logger.error("Writing Stats object {} in file:{} > {}.".format(stats_name, stats_file, err))


def getStatsName(import_type):
    """
    Generates the stats object name where to store the importing statistics from the CKG version, \
    which is defined in the configuration.

    :return: statsName: key used to store in the stats object.
    :rtype: str
    """
    version = ckg_config.version
    statsName = import_type+'_stats_' + str(version).replace('.', '_')

    return statsName


if __name__ == "__main__":
    fullImport()
