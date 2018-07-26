import urllib3
import urllib
import json
import urllib
from Bio import Entrez
from Bio import Medline
import os.path
import collections
import pprint
from KnowledgeGrapher import mapping as mp

def downloadDB(databaseURL, extraFolder =""):
    directory = os.path.join(config.databasesDir,extraFolder)
    fileName = databaseURL.split('/')[-1]    
    urllib.request.URLopener.version = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36 SE 2.X MetaSr 1.0'
    requestedFile = urllib.request.URLopener()
    requestedFile.retrieve(databaseURL, os.path.join(directory, fileName))

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

def getMapping():
    mapping = mp.generateMappingFromReflect()

    return mapping

def getMappingFromOntology(ontology, source):
    mapping = mp.getMappingFromOntology(ontology, source)

    return mapping

def getMappingFromDatabase(mappingFile):
    mapping = {}
    with open(mappingFile, 'r') as mf:
        for line in mf:
            data = line.rstrip("\r\n")
            ident = data[0]
            alias = data[1]
            mapping[alias] = ident

    return mapping

def getSTRINGMapping(url, source = "BLAST_UniProt_AC", download = True):
    mapping = collections.defaultdict(set)
    
    directory = os.path.join(config.databasesDir, "STRING")
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        downloadDB(url, "STRING")
    
    f = os.path.join(directory, fileName)
    mf = gzip.open(f, 'r')
    first = True
    for line in mf:
        if first:
            first = False
            continue
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        stringID = data[0]
        alias = data[1]
        sources = data[2].split(' ')
        if source in sources:
            mapping[stringID].add(alias)
        
    return mapping

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
