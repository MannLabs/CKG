import pandas as pd
import csv
import certifi
import urllib3
import urllib
import wget
import requests
import ftplib
import json
from Bio import Entrez
from Bio import Medline
import os.path
import collections
import pprint
import obonet
import datetime
import tarfile
import logging
import logging.config
from config import ckg_config
import ckg_utils
import subprocess


def readDataset(uri):
    if uri.endswith('.xlsx'):
        data = readDataFromExcel(uri)
    elif uri.endswith(".csv") or uri.endswith(".tsv") or uri.endswith(".txt"):
        if uri.endswith(".tsv") or uri.endswith(".txt"):
            data = readDataFromTXT(uri)
        else:
            data = readDataFromCSV(uri)
    data = data.dropna(how='all')

    return data

def readDataFromCSV(uri):
    """
    Read the data from csv file

    """
    data = pd.read_csv(uri, sep = ',', low_memory=False)

    return data

def readDataFromTXT(uri):
    """
    Read the data from tsv or txt file

    """
    data = pd.read_csv(uri, sep = '\t', low_memory=False)

    return data

def readDataFromExcel(uri):
    """
    Read the data from Excel file

    """
    data = pd.read_excel(uri, index_col=None, na_values=['NA'], convert_float = True)

    return data

def write_relationships(relationships, header, outputfile):
    """
    Reads a set of relationships and saves them to a file.

    :param set relationships: set of tuples with relationship data: source node, target node, \
                                relationship type, source and other attributes.
    :param list header: list of column names.
    :param str outputfile: path to file to be saved (including filename and extention).
    """
    try:
        df = pd.DataFrame(list(relationships), columns=header)
        df.to_csv(path_or_buf=outputfile, sep='\t',
                header=True, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')
    except Exception as err:
        raise csv.Error("Error writing relationships to file: {}.\n {}".format(outputfile, err))

def write_entities(entities, header, outputfile):
    """
    Reads a set of entities and saves them to a file.

    :param set entities: set of tuples with entities data: identifier, label, name\
                        and other attributes.
    :param list header: list of column names.
    :param str outputfile: path to file to be saved (including filename and extention).
    """
    try:
        df = pd.DataFrame(list(entities), columns=header)
        df.to_csv(path_or_buf=outputfile, sep='\t',
                header=True, index=False, quotechar='"', 
                line_terminator='\n', escapechar='\\')
    except csv.Error as err:
        raise csv.Error("Error writing etities to file: {}.\n {}".format(outputfile, err))

def get_config(config_name, data_type='databases'):
    """
    Reads YAML configuration file and converts it into a Python dictionary.

    :param str config_name: name of the configuration YAML file.
    :param str data_type: configuration type ('databases' or 'ontologies').
    :return: Dictionary.

    .. note:: Use this function to obtain configuration for individual database/ontology parsers.
    """
    cwd = os.path.abspath(os.path.dirname(__file__))
    config = ckg_utils.get_configuration(os.path.join(cwd, '{}/config/{}'.format(data_type, config_name)))

    return config

def setup_config(data_type="databases"):
    """
    Reads YAML configuration file and converts it into a Python dictionary.

    :param data_type: configuration type ('databases', 'ontologies', 'experiments' or 'builder').
    :return: Dictionary.

    .. note:: This function should be used to obtain the configuration for databases_controller.py, \
                ontologies_controller.py, experiments_controller.py and builder.py.
    """
    try:
        dirname = os.path.abspath(os.path.dirname(__file__))
        if data_type == 'databases':
            config = ckg_utils.get_configuration(os.path.join(dirname, ckg_config.databases_config_file))
        elif data_type == 'ontologies':
            config = ckg_utils.get_configuration(os.path.join(dirname, ckg_config.ontologies_config_file))
        elif data_type == "experiments":
            config = ckg_utils.get_configuration(os.path.join(dirname, ckg_config.experiments_config_file))
        elif data_type == 'builder':
            config = ckg_utils.get_configuration(os.path.join(dirname, ckg_config.builder_config_file))
        elif data_type == 'users':
            config = ckg_utils.get_configuration(os.path.join(dirname, ckg_config.users_config_file))

    except Exception as err:
        raise Exception("builder_utils - Reading configuration > {}.".format(err))

    return config

def list_ftp_directory(ftp_url, user='', password=''):
    """
    Lists all files present in folder from FTP server.

    :param str ftp_url: link to access ftp server.
    :param str user: username to access ftp server if required.
    :param str password: password to access ftp server if required.
    :return: List of files contained in ftp server folder provided with ftp_url.
    """
    try:
        domain = ftp_url.split('/')[2]
        if len(ftp_url.split('/')) > 3:
            ftp_dir = '/'.join(ftp_url.split('/')[3:])
        else:
            ftp_dir = ''
        with ftplib.FTP(domain) as ftp:
            ftp.login(user=user, passwd=password)
            files = ftp.nlst(ftp_dir)
    except ftplib.error_perm as err:
        raise Exception("builder_utils - Problem listing file at {} ftp directory > {}.".format(ftp_dir, err))

    return files

def setup_logging(path='log.config', key=None):
    """
    Setup logging configuration.

    :param str path: path to file containing configuration for logging file.
    :param str key: name of the logger.
    :return: Logger with the specified name from 'key'. If key is *None*, returns a logger which is \
                the root logger of the hierarchy.
    """
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        try:
            logging.config.dictConfig(config)
        except:
            logging.basicConfig(level=logging.DEBUG)        
    else:
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(key)
    
    return logger

def downloadDB(databaseURL, directory=None, file_name=None, user="", password=""):
    """
    This function downloads the raw files from a biomedical database server when a link is provided.

    :param str databaseURL: link to access biomedical database server.
    :param directory:
    :type directory: str or None
    :param file_name: name of the file to dowload. If None, 'databaseURL' must contain \
                        filename after the last '/'.
    :type file_name: str or None
    :param str user: username to access biomedical database server if required.
    :param str password: password to access biomedical database server if required.
    """
    dbconfig = setup_config()
    if directory is None:
        directory = dbconfig["databasesDir"]
    if file_name is None:
        file_name = databaseURL.split('/')[-1]
    header = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
    try:
        mode = 'wb'
        if databaseURL.startswith('ftp:'):
            domain = databaseURL.split('/')[2]
            ftp_file = '/'.join(databaseURL.split('/')[3:])
            with ftplib.FTP(domain) as ftp:
                ftp.login(user=user, passwd = password)
                ftp.retrbinary("RETR " + ftp_file ,  open(os.path.join(directory, file_name), mode).write)
        else:
            if os.path.exists(os.path.join(directory, file_name)):
                os.remove(os.path.join(directory, file_name))
            try:
                wget.download(databaseURL, os.path.join(directory, file_name))
            except:
                r = requests.get(databaseURL, headers=header)
                with open(os.path.join(directory, file_name), 'wb') as out:
                    out.write(r.content)
            #os.system("wget -O {0} {1}".format(os.path.join(directory, file_name), databaseURL))
    except ftplib.error_reply as err:
        raise ftplib.error_reply("Exception raised when an unexpected reply is received from the server. {}.\nURL:{}".format(err,databaseURL))
    except ftplib.error_temp as err:
        raise ftplib.error_temp("Exception raised when an error code signifying a temporary error. {}.\nURL:{}".format(err,databaseURL))
    except ftplib.error_perm as err:
        raise ftplib.error_perm("Exception raised when an error code signifying a permanent error. {}.\nURL:{}".format(err,databaseURL))
    except ftplib.error_proto:
        raise ftplib.error_proto("Exception raised when a reply is received from the server that does not fit the response specifications of the File Transfer Protocol. {}.\nURL:{}".format(err,databaseURL))
    except Exception as err:
        raise Exception("Something went wrong. {}.\nURL:{}".format(err,databaseURL))

def searchPubmed(searchFields, sortby = 'relevance', num ="10", resultsFormat = 'json'):
    """
    Searches PubMed database for MeSH terms and other additional fields ('searchFields'), sorts them by relevance and \
    returns the top 'num'.

    :param list searchFields: list of search fields to query for.
    :param str sortby: parameter to use for sorting.
    :param str num: number of PubMed identifiers to return.
    :param str resultsFormat: format of the PubMed result.
    :return: Dictionary with total number of PubMed ids, and top 'num' ids.
    """
    pubmedQueryUrl = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=TERM&retmode=json&retmax=NUM'
    if len(searchFields) > 1:
        query = " [MeSH Terms] AND ".join(searchFields)
    else:
        query = searchFields[0] +" [MeSH Terms] AND"
    try:
        url = pubmedQueryUrl.replace('TERMS',query).replace('NUM', num)
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        response = http.request("GET", urllib.parse.quote(url))
        jsonResponse = response.read()
        resultDict = json.loads(jsonResponse)
    except urllib3.exceptions.InvalidHeader:
        raise urllib3.exceptions.InvalidHeader("Invalid HTTP header provided. {}.\nURL:{}".format(err,url))
    except urllib3.exceptions.ConnectTimeoutError:
        raise urllib3.exceptions.ConnectTimeoutError("Connection timeout requesting URL. {}.\nURL:{}".format(err,url))
    except urllib3.exceptions.ConnectionError:
        raise urllib3.exceptions.ConnectionError("Protocol error when downloading. {}.\nURL:{}".format(err,url))
    except urllib3.exceptions.DecodeError:
        raise urllib3.exceptions.DecodeError("Decoder error when downloading. {}.\nURL:{}".format(err,url))
    except urllib3.exceptions.SecurityWarning:
        raise urllib3.exceptions.SecurityWarning("Security warning when downloading. {}.\nURL:{}".format(err,url))
    except urllib3.exceptions.ProtocolError:
        raise urllib3.exceptions.ProtocolError("Protocol error when downloading. {}.\nURL:{}".format(err,url))
    except ftplib.error_reply as err:
        raise ftplib.error_reply("Exception raised when an unexpected reply is received from the server. {}.\nURL:{}".format(err,url))
    except ftplib.error_temp as err:
        raise ftplib.error_temp("Exception raised when an error code signifying a temporary error. {}.\nURL:{}".format(err,url))
    except ftplib.error_perm as err:
        raise ftplib.error_perm("Exception raised when an error code signifying a permanent error. {}.\nURL:{}".format(err,url))
    except ftplib.error_proto:
        raise ftplib.error_proto("Exception raised when a reply is received from the server that does not fit the response specifications of the File Transfer Protocol. {}.\nURL:{}".format(err,url))
    except Exception as err:
        raise Exception("Something went wrong. {}.\nURL:{}".format(err,url))

    result = []
    if 'esearchresult' in resultDict:
        result = resultDict['esearchresult']
    
    return result

def is_number(s):
    """
    This function checks if given input is a float and returns True if so, and False if it is not.

    :param s: input
    :return: Boolean.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def getMedlineAbstracts(idList):
    """
    This function accesses NCBI over the WWWW and returns Medline data as a handle object, \
    which is parsed and converted to a Pandas DataFrame.

    :param idList: single identifier or comma-delimited list of identifiers. All the identifiers \
                    must be from the database PubMed.
    :type idList: str or list
    :return: Pandas DataFrame with columns: 'title', 'authors', 'journal', 'keywords', 'abstract', 'PMID' and 'url'.
    """

    fields = {"TI":"title", "AU":"authors", "JT":"journal", "DP":"date", "MH":"keywords", "AB":"abstract", "PMID":"PMID"}
    pubmedUrl = "https://www.ncbi.nlm.nih.gov/pubmed/"
    handle = Entrez.efetch(db="pubmed", id=idList, rettype="medline", retmode="json")
    records = Medline.parse(handle)
    results = []
    for record in records:
        aux = {}
        for field in fields:
            if field in record:
                aux[fields[field]] = record[field]
        if "PMID" in aux:
            aux["url"] = pubmedUrl + aux["PMID"]
        else:
            aux["url"] = ""
        
        results.append(aux)

    abstracts = pd.DataFrame.from_dict(results)

    return abstracts

def listDirectoryFiles(directory):
    """
    Lists all files in a specified directory.

    :param str directory: path to folder.
    :return: List of file names.
    """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and not f.startswith('.')]

    return onlyfiles

def listDirectoryFolders(directory):
    """
    Lists all directories in a specified directory.

    :param str directory: path to folder.
    :return: List of folder names.
    """
    from os import listdir
    from os.path import isdir, join
    dircontent = [f for f in listdir(directory) if isdir(join(directory, f)) and not f.startswith('.')]
    return dircontent

def checkDirectory(directory):
    """
    Checks if given directory exists and if not, creates it.

    :param str directory: path to folder.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def flatten(t):
    """
    Code from: https://gist.github.com/shaxbee/0ada767debf9eefbdb6e
    Acknowledgements: Zbigniew Mandziejewicz (shaxbee)
    Generator flattening the structure
    
    >>> list(flatten([2, [2, (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
    [2, 2, 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
    """
    for x in t:
        if not isinstance(x, collections.Iterable) or isinstance(x, str):
            yield x
        else:
            yield from flatten(x)

def pretty_print(data):
    """
    This function provides a capability to "pretty-print" arbitrary Python data structures in a forma that can be \
    used as input to the interpreter. For more information visit https://docs.python.org/2/library/pprint.html.

    :param data: python object.
    """
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def convertOBOtoNet(ontologyFile):
    """
    Takes an .obo file and returns a NetworkX graph representation of the ontology, that holds multiple \
    edges between two nodes.

    :param str ontologyFile: path to ontology file.
    :return: NetworkX graph.
    """
    graph = obonet.read_obo(ontologyFile)
    
    return graph

def getCurrentTime():
    """
    Returns current date (Year-Month-Day) and time (Hour-Minute-Second).

    :return: Two strings: date and time.
    """
    now = datetime.datetime.now()
    return '{}-{}-{}'.format(now.year, now.month, now.day), '{}:{}:{}'.format(now.hour, now.minute, now.second) 

def convert_bytes(num):
    """
    This function will convert bytes to MB.... GB... etc.

    :param num: float, integer or pandas.Series.
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    This function returns the file size.

    :param str file_path: path to file.
    :return: Size in bytes of a plain file.
    :rtype: str
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return str(file_info.st_size)

def buildStats(count, otype, name, dataset, filename, updated_on=None):
    """
    Returns a tuple with all the information needed to build a stats file.

    :param int count: number of entities/relationships.
    :param str otype: 'entity' or 'relationsgips'.
    :param str name: entity/relationship label.
    :param str dataset: database/ontology.
    :param str filename: path to file where entities/relationships are stored.
    :return: Tuple with date, time, database name, file where entities/relationships are stored, \
    file size, number of entities/relationships imported, type and label.
    """
    y,t = getCurrentTime()
    size = file_size(filename)
    filename = filename.split('/')[-1]
    
    return(str(y), str(t), dataset, filename, size, count, otype, name, updated_on)

def compress_directory(folder_to_backup, dest_folder, file_name):
    """
    Compresses folder to .tar.gz to create data backup archive file.

    :param str folder_to_backup: path to folder to compress and backup.
    :param str dest_folder: path where to save compressed folder.
    :param str file_name: name of the compressed file.
    """
    #tar cf - paths-to-archive | pigz -9 -p 32 > archive.tar.gz
    filePath = os.path.join(dest_folder,file_name+".tar.gz")
    filePath = filePath.replace("(","\(").replace(")","\)")
    folder_to_backup = folder_to_backup.replace("(","\(").replace(")","\)")
    os.system("tar -zcf {} {}".format(filePath, folder_to_backup))


def read_gzipped_file(filepath):
    """
    Opens an underlying process to access a gzip file through the creation of a new pipe to the child.

    :param str filepath: path to gzip file.
    :return: A bytes sequence that specifies the standard output.
    """
    p = subprocess.Popen(["gzcat", filepath],
        stdout=subprocess.PIPE
    )
    return p.stdout
