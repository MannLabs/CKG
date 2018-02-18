import utils


def createProject():
    pass

def generateSampleQRs():
    pass

def updateProject():
    pass

def getListOfClinicalVariables(disease):
    pass

def getListOfRelatedPublications(disease, tissue, technology):
    publications = utils.searchPubmed(searchFields = [disease, tissue, technology], sortby = 'relevance', num ="10", resultsFormat = 'json')
    if len(publications) > 1:
        if 'idlist' in publications:
            idList = publications['idlist']
            results = getMedlineAbstracts(idList)

    return results
        
