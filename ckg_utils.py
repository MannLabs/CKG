import yaml
import json
import os
from os.path import isfile, join
import logging

def read_yaml(yaml_file):
    content = None
    with open(yaml_file, 'r') as stream:
        try:
            content = yaml.load(stream)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
    return content

def get_queries(queries_file):
    queries  = None
    if queries_file.endswith("yml"):
        queries = read_yaml(queries_file)
    else:
        raise Exception("The format specified in the queries file {} is not supported. {}".format(queries_file, err))

    return queries

def get_configuration(configuration_file):
    configuration  = None
    if configuration_file.endswith("yml"):
        configuration = read_yaml(configuration_file)
    else:
        raise Exception("The format specified in the configuration file {} is not supported. {}".format(configuration_file, err))

    return configuration

def get_configuration_variable(configuration_file, variable):
    configuration = get_configuration(configuration_file)
    if variable in configuration:
        return configuration[variable]
    else:
        raise Exception("The varible {} is not found in the configuration file {}. {}".format(variable, configuration_file, err))

def setup_logging(path='log.config', key=None):
    """Setup logging configuration"""
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(key)
    
    return logger


def listDirectoryFiles(directory):
    onlyfiles = [f for f in os.listdir(directory) if isfile(join(directory, f)) and not f.startswith('.')]

    return onlyfiles

def listDirectoryFolders(directory):
    dircontent = [f for f in os.listdir(directory) if isdir(join(directory, f)) and not f.startswith('.')]
    return dircontent

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
