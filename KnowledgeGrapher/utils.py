import urllib3
import urllib
import json
import urllib
from Bio import Entrez
from Bio import Medline
import os.path
import collections
import pprint
import obonet
import datetime
import tarfile
from KnowledgeGrapher.ontologies import ontologies_config as oconfig
from KnowledgeGrapher.databases import databases_config as dbconfig


def downloadDB(databaseURL, extraFolder =""):
    if extraFolder == "":
        directory = dbconfig.databasesDir
    else:
        directory = extraFolder
    fileName = databaseURL.split('/')[-1]    
    urllib.request.URLopener.version = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36 SE 2.X MetaSr 1.0'
    try:
        requestedFile = urllib.request.URLopener()
        requestedFile.retrieve(databaseURL, os.path.join(directory, fileName))
    except urllib.HTTPError:
        pass
    except urllib.URLError:
        pass
    except TimeoutError:
        pass
    except OSError:
        pass
    except Exception:
        pass

def searchPubmed(searchFields, sortby = 'relevance', num ="10", resultsFormat = 'json'):
    pubmedQueryUrl = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=TERM&retmode=json&retmax=NUM'
    if len(searchFields) > 1:
        query = " [MeSH Terms] AND ".join(searchFields)
    else:
        query = searchFields[0] +" [MeSH Terms] AND"

    response = urllib3.urlopen(urllib.quote_plus(pubmedQueryUrl.replace('TERMS',query).replace('NUM', num)))
    jsonResponse = response.read()
    resultDict = json.loads(jsonResponse)

    result = []
    if 'esearchresult' in resultDict:
        result = resultDict['esearchresult']
    
    return result

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def getMedlineAbstracts(idList):
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

    return results

def listDirectoryFiles(directory):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and not f.startswith('.')]

    return onlyfiles

def listDirectoryFolders(directory):
    from os import listdir
    from os.path import isdir, join
    dircontent = [f for f in listdir(directory) if isdir(join(directory, f)) and not f.startswith('.')]
    return dircontent

def checkDirectory(directory):
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
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def convertOBOtoNet(ontologyFile):
    graph = obonet.read_obo(ontologyFile)
    
    return graph

def getCurrentTime():
    now = datetime.datetime.now()
    return '{}-{}-{}'.format(now.year, now.month, now.day), '{}:{}:{}'.format(now.hour, now.minute, now.second) 

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return str(file_info.st_size)

def buildStats(count, otype, name, dataset, filename):
    y,t = getCurrentTime()
    size = file_size(filename)
    filename = filename.split('/')[-1]
    
    return(y, t, dataset, filename, size, count, otype, name)

def compress_directory(folder_to_backup, dest_folder, file_name):
    #tar cf - paths-to-archive | pigz -9 -p 32 > archive.tar.gz
    filePath = os.path.join(dest_folder,file_name+".tar.gz")
    filePath = filePath.replace("(","\(").replace(")","\)")
    folder_to_backup = folder_to_backup.replace("(","\(").replace(")","\)")
    os.system("tar -zcf {} {}".format(filePath, folder_to_backup))

    
