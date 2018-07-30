import config
from KnowledgeGrapher.ontologies import ontologieshandler as oh, ontologies_config as oconfig
from KnowledgeGrapher.databases import databaseshandler as dh, databases_config as dbconfig
import os.path

def ontologyImport():
    pass

def fullImport():
    importDirectory = config.importDirectory
    checkDirectory(importDirectory)
    #Ontologies
    ontologiesImportDirectory = os.path.join(importDirectory, oconfig.ontologiesImportDir)
    checkDirectory(ontologiesImportDirectory)
    #oh.generateGraphFiles(ontologiesImportDirectory)
    #Databases
    databasesImportDirectory = os.path.join(importDirectory, dbconfig.databasesImportDir)
    checkDirectory(databasesImportDirectory)
    dh.generateGraphFiles(databasesImportDirectory)

if __name__ == "__main__":
    fullImport()
