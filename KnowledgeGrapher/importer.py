import config
import ontologieshandler as oh
import databaseshandler as dh

def ontologyImport():
    pass

def fullImport():
    importDirectory = config.importDirectory
    #Ontologies
    oh.generateGraphFiles(importDirectory)
    #Databases
    dh.generateGraphFiles(importDirectory)


if __name__ == "__main__":
    fullImport()
