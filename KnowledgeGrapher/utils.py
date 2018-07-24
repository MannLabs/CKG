import urllib3
import json
import urllib
from Bio import Entrez
from Bio import Medline
import os.path
import collections

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

def getMappingFromDatabase(mappingFile):
    mapping = {}
    with open(mappingFile, 'r') as mf:
        for line in mf:
            data = line.rstrip("\r\n")
            ident = data[0]
            alias = data[1]
            mapping[alias] = ident

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
