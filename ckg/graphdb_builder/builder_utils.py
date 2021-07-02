from neo4j.meta import experimental
import pandas as pd
import csv
import certifi
import urllib3
import urllib
import wget
import base64
import glob
import io
import re
import requests
import ftplib
import json
import gzip
import shutil
from Bio import Entrez, Medline, SeqIO
import os.path
import collections
import pprint
import obonet
import datetime
import logging
import logging.config
from ckg import ckg_utils
from ckg.graphdb_connector import connector
import zipfile
import rarfile


def readDataset(uri):
    data = pd.DataFrame()
    if uri.endswith('.xlsx'):
        data = readDataFromExcel(uri)
    elif uri.endswith(".tsv") or uri.endswith(".txt") or uri.endswith(".sdrf"):
        data = readDataFromTXT(uri)
    elif uri.endswith(".csv"):
        data = readDataFromCSV(uri)

    data = data.dropna(how='all')

    return data


def readDataFromCSV(uri, sep=',', header=0, comment=None):
    """
    Read the data from csv file

    """
    data = pd.read_csv(uri, sep=sep, header=header, comment=comment, low_memory=False)

    return data


def readDataFromTXT(uri):
    """
    Read the data from tsv or txt file

    """
    data = pd.read_csv(uri, sep='\t', low_memory=False)

    return data


def readDataFromExcel(uri):
    """
    Read the data from Excel file

    """
    data = pd.read_excel(uri, index_col=None, na_values=['NA'], convert_float=True)

    return data


def get_files_by_pattern(regex_path):
    files = glob.glob(regex_path)

    return files


def get_extra_pairs(directory, extra_file):
    extra = set()
    file_path = os.path.join(directory, extra_file)

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                data = line.rstrip("\r\n").split("\t")
                extra.add(tuple(data))

    return extra


def parse_contents(contents, filename):
    """
    Reads binary string files and returns a Pandas DataFrame.
    """
    df = None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_format = filename.split('.')[-1]

    if file_format == 'txt' or file_format == 'tsv':
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', low_memory=True)
    elif file_format == 'csv':
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), low_memory=True)
    elif file_format == 'xlsx' or file_format == 'xls':
        df = pd.read_excel(io.BytesIO(decoded))
    elif file_format == 'mztab':
        df = parse_mztab_filehandler(io.StringIO(decoded.decode('utf-8')))
    elif file_format == 'sdrf':
        df = parse_sdrf_filehandler(io.StringIO(decoded.decode('utf-8')))

    return df


def export_contents(data, dataDir, filename):
    """
    Export Pandas DataFrame to file, with UTF-8 endocing.

    """
    file_format = filename.split('.')[-1]
    if file_format == 'txt' or file_format == 'tsv':
        data.to_csv(os.path.join(dataDir, filename), sep='\t', index=False, encoding='utf-8')
    elif file_format == 'csv':
        data.to_csv(os.path.join(dataDir, filename), sep=',', index=False, encoding='utf-8')
    elif file_format == 'xlsx' or file_format == 'xls':
        data.to_excel(os.path.join(dataDir, filename), index=False, encoding='utf-8')
    elif file_format == 'mztab':
        for dataset in data:
            data[dataset].to_csv(os.path.join(dataDir, dataset+'.tsv'), sep='\t', index=False, encoding='utf-8')
    elif file_format == 'sdrf':
        for dataset in data:
            directory = dataDir
            if dataset == "experimental_design":
                directory = os.path.join(dataDir, "../experimental_design")
                checkDirectory(directory)
            
            data[dataset].to_excel(os.path.join(directory, dataset+'.xlsx'), index=False, encoding='utf-8')


def parse_mztab_filehandler(mztabf):
    content = {'MTD': [], 
               'PRH': None, 'PRT': [],
               'PEH': None, 'PEP': [],
               'PSH': None, 'PSM': [],
               'SMH': None, 'SML': []}
    
    datasets = {}
    
    headers = {'PRT': 'PRH', 'PEP': 'PEH', 'PSM': 'PSH', 'SML': 'SMH'}

    for line in mztabf:
        data = line.rstrip().split('\t')
        dtype = data[0]
        if dtype in content:
            if content[dtype] is None:
                content[dtype] = data[1:]
            else:
                content[dtype].append(data[1:])

    for dtype in content:
        columns = None
        if dtype not in headers.values():
            if dtype in headers and dtype in content:
                header = headers[dtype]
                columns = content[header]
            
            if len(content[dtype]) > 0:
                datasets[dtype] = pd.DataFrame(content[dtype], columns=columns)
            
    return datasets


def parse_mztab_file(mztab_file):
    with open(mztab_file, 'r') as mztabf:
        datasets = parse_mztab_filehandler(mztabf)
            
    return datasets

def parse_sdrf_filehandler(sdrf_fh):
    data = {}
    datasets = {'experimental_design': ['subject external_id',
                                    'biological_sample external_id',
                                    'analytical_sample external_id',
                                    'grouping1'], 
            'clinical_data':[]}
    
    sdrf_df = pd.read_csv(sdrf_fh, sep='\t', low_memory=True)
    df = convert_sdrf_to_ckg(sdrf_df)

    for dataset in datasets:
        cols = datasets[dataset]
        if len(cols)>0:
            data[dataset] = df[cols]
        else:
            data[dataset] = df.copy()
        
    return data

def convert_ckg_to_sdrf(df):
    out_mapping = {'tissue':'characteristics[organism part]',
                   'disease': 'characteristics[disease]',
                   'grouping1': 'characteristics[phenotype]',
                   'analytical_sample': 'comment[data file]',
                   'subject': 'characteristics[individual]',
                   'biological_sample': 'source name'}
    
    if not df.empty:
        df = pd.pivot_table(df, index=['subject','biological_sample', 'analytical_sample', 'grouping1', 'tissue'], columns=['exp_factor'], values=['exp_factor_value'])
        df.columns = ["characteristics[{}]".format(c[1]) for c in df]
        df = df.reset_index()
        df = df.rename(out_mapping, axis=1)
        
    return df

def convert_sdrf_to_ckg(df):
    in_mapping = {'organism part': 'tissue',
                  'disease': 'disease',
                  'phenotype': 'grouping1',
                  'data file': 'analytical_sample external_id',
                  'individual':'subject external_id',
                  'source name':'biological_sample external_id'}
    cols = {}
    for c in df.columns:
        matches = re.search(r'\[(.+)\]', c)
        if matches:
            cols[c] = matches.group(1)
    
    driver = connector.getGraphDatabaseConnectionConfiguration()
    query = '''MATCH (ef:Experimental_factor)-[r:MAPS_TO]-(c:Clinical_variable)
                WHERE ef.name IN {} RETURN ef.name AS from, c.name+' ('+c.id+')' AS to, LABELS(c)'''
    
    mapping = connector.getCursorData(driver, query.format(list(cols.values())))
    mapping = dict(zip(mapping['from'], mapping['to']))
    mapping.update(in_mapping)
    df = df.rename(cols, axis=1).rename(mapping, axis=1)
    
    return df

def convert_ckg_clinical_to_sdrf(df):
    out_mapping = {'tissue':'characteristics[organism part]',
                   'disease': 'characteristics[disease]',
                   'grouping1': 'characteristics[phenotype]',
                   'analytical_sample': 'comment[data file]',
                   'subject': 'characteristics[individual]',
                   'biological_sample': 'source name'}
    cols = []
    for c in df.columns:
        matches = re.search(r'(\d+)', c)
        if matches:
            cols.append(c)
    
    driver = connector.getGraphDatabaseConnectionConfiguration()
    query = '''MATCH (ef:Experimental_factor)-[r:MAPS_TO]-(c:Clinical_variable)
                WHERE c.name+' ('+c.id+')' IN {} RETURN c.name+' ('+c.id+')' AS from, "characteristic["+ef.name+"]" AS to, LABELS(c)'''
    
    mapping = connector.getCursorData(driver, query.format(cols))
    mapping = dict(zip(mapping['from'], mapping['to']))
    mapping.update(out_mapping)
    df = df.rename(mapping, axis=1)
    
    return df

def convert_sdrf_file_to_ckg(file_path):
    sdrf_df = pd.read_csv(file_path, sep='\t')
    df = convert_sdrf_to_ckg(sdrf_df)
    
    return df


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
    directory = os.path.join(ckg_utils.read_ckg_config(key='ckg_directory'), 'graphdb_builder')
    config = ckg_utils.get_configuration(os.path.join(directory, '{}/config/{}'.format(data_type, config_name)))

    return config


def expand_cols(data, col, sep=';'):
    """
    Expands the rows of a dataframe by splitting the specified column

    :param data: dataframe to be expanded
    :param str col: column that contains string to be expanded (i.e. 'P02788;E7EQB2;E7ER44;P02788-2;C9JCF5')
    :param str sep: separator (i.e. ';')
    :return: expanded pandas dataframe
    """
    s = data[col].str.split(sep).apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del data[col]
    pdf = s.to_frame(col)
    data = data.join(pdf)

    return data


def setup_config(data_type="databases"):
    """
    Reads YAML configuration file and converts it into a Python dictionary.

    :param data_type: configuration type ('databases', 'ontologies', 'experiments' or 'builder').
    :return: Dictionary.

    .. note:: This function should be used to obtain the configuration for databases_controller.py, \
                ontologies_controller.py, experiments_controller.py and builder.py.
    """
    try:
        dirname = os.path.join(ckg_utils.read_ckg_config(key='ckg_directory'), 'graphdb_builder')
        file_name = '{}/{}_config.yml'.format(data_type, data_type)
        config = ckg_utils.get_configuration(os.path.join(dirname, file_name))
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
        except Exception:
            logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(key)

    return logger


def download_from_ftp(ftp_url, user, password, to, file_name):
    try:
        domain = ftp_url.split('/')[2]
        ftp_file = '/'.join(ftp_url.split('/')[3:])
        with ftplib.FTP(domain) as ftp:
            ftp.login(user=user, passwd=password)
            with open(os.path.join(to, file_name), 'wb') as fp:
                ftp.retrbinary("RETR " + ftp_file,  fp.write)
    except ftplib.error_reply as err:
        raise ftplib.error_reply("Exception raised when an unexpected reply is received from the server. {}.\nURL:{}".format(err,ftp_url))
    except ftplib.error_temp as err:
        raise ftplib.error_temp("Exception raised when an error code signifying a temporary error. {}.\nURL:{}".format(err,ftp_url))
    except ftplib.error_perm as err:
        raise ftplib.error_perm("Exception raised when an error code signifying a permanent error. {}.\nURL:{}".format(err,ftp_url))
    except ftplib.error_proto:
        raise ftplib.error_proto("Exception raised when a reply is received from the server that does not fit the response specifications of the File Transfer Protocol. {}.\nURL:{}".format(err,ftp_url))


def download_PRIDE_data(pxd_id, file_name, to='.', user='', password='', date_field='publicationDate'):
    """
    This function downloads a project file from the PRIDE repository

    :param str pxd_id: PRIDE project identifier (id. PXD013599).
    :param str file_name: name of the file to dowload
    :param str to: local directory where the file should be downloaded
    :param str user: username to access biomedical database server if required.
    :param str password: password to access biomedical database server if required.
    :param str date_field: projects deposited in PRIDE are search based on date, either
        submissionData or publicationDate (default)
    """
    ftp_PRIDE = 'ftp://ftp.pride.ebi.ac.uk/pride/data/archive/YEAR/MONTH/PXDID/FILE_NAME'
    url_PRIDE_API = 'http://www.ebi.ac.uk/pride/ws/archive/project/' + pxd_id
    data = None
    try:
        r = requests.get(url_PRIDE_API)
        data = r.json()
        submission_date = data[date_field]
        year, month, day = submission_date.split('-')

        ftp_url = ftp_PRIDE.replace('YEAR', year).replace('MONTH', month).replace('PXDID', pxd_id).replace('FILE_NAME', file_name)
        download_from_ftp(ftp_url, user, password, to, file_name)
    except Exception as err:
        print(err)

    return data


def downloadDB(databaseURL, directory=None, file_name=None, user="", password="", avoid_wget=False):
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
        directory = ckg_utils.read_ckg_config()["databases_directory"]
    if file_name is None:
        file_name = databaseURL.split('/')[-1].replace('?', '_').replace('=', '_')
    header = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
    try:
        if databaseURL.startswith('ftp:'):
            download_from_ftp(databaseURL, user, password, directory, file_name)
        else:
            if os.path.exists(os.path.join(directory, file_name)):
                os.remove(os.path.join(directory, file_name))
            try:
                if not avoid_wget:
                    wget.download(databaseURL, os.path.join(directory, file_name))
                else:
                    r = requests.get(databaseURL, headers=header)
                    with open(os.path.join(directory, file_name), 'wb') as out:
                        out.write(r.content)
            except Exception:
                r = requests.get(databaseURL, headers=header)
                with open(os.path.join(directory, file_name), 'wb') as out:
                    out.write(r.content)
    except Exception as err:
        raise Exception("Something went wrong. {}.\nURL:{}".format(err, databaseURL))


def searchPubmed(searchFields, sortby='relevance', num="10", resultsFormat='json'):
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
        query = searchFields[0] + " [MeSH Terms] AND"
    try:
        url = pubmedQueryUrl.replace('TERMS', query).replace('NUM', num)
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

    fields = {"TI": "title", "AU": "authors", "JT": "journal", "DP": "date", "MH": "keywords", "AB": "abstract", "PMID": "PMID"}
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


def remove_directory(directory):
    if os.path.exists(directory):
        files = listDirectoryFiles(directory)
        folders = listDirectoryFolders(directory)
        if 'complete_mapping.tsv' in files:
            for f in files:
                if f != 'complete_mapping.tsv':
                    os.remove(os.path.join(directory, f))
            for d in folders:
                remove_directory(os.path.join(directory, d))
        else:
            shutil.rmtree(directory, ignore_errors=False, onerror=None)
    else:
        print("Done")


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


def listDirectoryFoldersNotEmpty(directory):
    """
    Lists all directories in a specified directory.

    :param str directory: path to folder.
    :return: List of folder names.
    """
    from os import listdir
    from os.path import isdir, join
    dircontent = []
    if isdir(directory):
        dircontent = [f for f in listdir(directory) if not f.startswith('.') and isdir(join(directory, f)) and listdir(join(directory, f))]

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


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        checkDirectory(dst)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


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

def unrar(filepath, to):
    """
    Decompress RAR file
    :param str filepath: path to rar file
    :param str to: where to extract all files

    """
    try:
        with rarfile.RarFile(filepath) as opened_rar:
            opened_rar.extractall(to)
    except Exception as err:
        print("Error: {}. Could not unrar file {}".format(filepath, err))

def unzip_file(filepath, to):
    """
    Decompress zipped file
    :param str filepath: path to zip file
    :param str to: where to extract all files

    """
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(to)
    except Exception as err:
        print("Error: {}. Could not unzip file {}".format(filepath, err))

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
    handle = gzip.open(filepath, "rt")

    return handle


def parse_fasta(file_handler):
    """
    Using BioPython to read fasta file as SeqIO objects

    :param file_handler file_handler: opened fasta file
    :return iterator records: iterator of sequence objects
    """
    records = SeqIO.parse(file_handler,format="fasta")

    return records


def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.

    :param iterator iterator: batch to be extracted
    :param integer batch_size: size of the batch
    :return list batch: list with the batch elements of size batch_size

    source: https://biopython.org/wiki/Split_large_file
    """
    entry = True
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch
