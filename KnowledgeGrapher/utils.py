import urllib2
import json
import urllib
from Bio import Entrez
from Bio import Medline


def searchPubmed(searchFields, sortby = 'relevance', num ="10", resultsFormat = 'json'):
    pubmedQueryUrl = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=TERM&retmode=json&retmax=NUM'
    if len(searchFields) > 1:
        query = " [MeSH Terms] AND ".join(searchFields)
    else:
        query = searchFields[0] +" [MeSH Terms] AND"

    response = urllib2.urlopen(urllib.quote_plus(pubmedQueryUrl.replace('TERMS',query).replace('NUM', num)))
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
