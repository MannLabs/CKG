import utils
import config
import cypher as cy
from os.path import join
from py2neo import Graph

def getGraphDatabaseConnectionConfiguration():
    host = config.dbURL
    port = config.dbPort
    user = config.dbUser
    password = config.dbPassword

    driver = connectToDB(host, port, user, password)

    return driver

def connectToDB(host = "localhost", port= 7687, user="neo4j", password = "password"):
    driver = Graph(host=host, port = port, user = user, password = password)

    return driver

def removeRelationshipDB(entity1, entity2, relationship):
    driver = getGraphDatabaseConnectionConfiguration()

    countCy = cy.COUNT_RELATIONSHIPS 
    deleteCy = cy.REMOVE_RELATIONSHIPS
    countst = countCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    deletest = deleteCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    print countst
    print driver.run(countst).data()[0]
    print "Removing %d entries in the database" % driver.run(countst).data()[0]['count']
    driver.run(deletest)
    print "Existing entries after deletion: %d" % driver.run(countst).data()[0]['count']

def removeNodes(entity):
   pass 

def createDB(imports = ["ontologies","proteins", "ppi"]):
    entities = config.entities
    importDir = config.importDirectory
    driver = getGraphDatabaseConnectionConfiguration()

    #Get the cypher queries to build the graph
    #Ontologies
    if "ontologies" in imports:
        ontologyDataImportCode = cy.IMPORT_ONTOLOGY_DATA
        for entity in entities:
            cypherCode = ontologyDataImportCode.replace("ENTITY", entity).replace("IMPORTDIR",importDir).split(';')[0:-1]
            for statement in cypherCode:
                print statement
                driver.run(statement+";")
    #Databases
    #Chromosomes
    if "chromosomes" in imports:
        chromosomeDataImportCode = cy.IMPORT_CHROMOSOME_DATA
        for statement in chromosomeDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
            print statement+";"
            driver.run(statement+";")
    #Genes
    if "genes" in imports:
        geneDataImportCode = cy.IMPORT_GENE_DATA
        for statement in geneDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
            print statement+";"
            driver.run(statement+";")
    #Transcripts
    if "transcripts" in imports:
        transcriptDataImportCode = cy.IMPORT_TRANSCRIPT_DATA
        for statement in transcriptDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
            print statement+";"
            driver.run(statement+";")
    #Proteins
    if "proteins" in imports:
        proteinDataImportCode = cy.IMPORT_PROTEIN_DATA
        for statement in proteinDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
            print statement+";"
            driver.run(statement+";")
    #PPIs
    if "ppi" in imports:
        ppiDataImportCode = cy.IMPORT_CURATED_PPI_DATA
        for resource in config.PPI_resources:
            for statement in ppiDataImportCode.replace("IMPORTDIR",importDir).replace("RESOURCE",resource.upper()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Diseases
    if "diseases" in imports:
        diseaseDataImportCode = cy.IMPORT_DISEASE_DATA
        for entity,resource in config.disease_resources:
            for statement in diseaseDataImportCode.replace("IMPORTDIR",importDir).replace("ENTITY", entity).replace("RESOURCE",resource.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Drugs
    if "drugs" in imports:
        drugsDataImportCode = cy.IMPORT_CURATED_DRUG_DATA
        for resource in config.drug_resources:
            for statement in drugsDataImportCode.replace("IMPORTDIR",importDir).replace("RESOURCE",resource.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Internal
    if "internal" in imports:
        internalDataImportCode = cy.IMPORT_INTERNAL_DATA
        for (entity1, entity2) in config.internalEntities:
            for statement in internalDataImportCode.replace("IMPORTDIR", importDir).replace("ENTITY1", entity1).replace("ENTITY2", entity2).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Mentions
    if "mentions" in imports:
        publicationsImportCode = cy.CREATE_PUBLICATIONS
        for statement in publicationsImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
            print statement+";"
            driver.run(statement+";")
        
        mentionsImportCode = cy.IMPORT_MENTIONS
        for entity in config.mentionEntities:
            for statement in mentionsImportCode.replace("IMPORTDIR", importDir).replace("ENTITY", entity).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Known_variants
    if "variants" in imports:
        variantsImportCode = cy.IMPORT_KNOWN_VARIANT_DATA
        for source in config.variant_resources:
            for statement in variantsImportCode.replace("IMPORTDIR", importDir).replace("SOURCE",source.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")
    #Projects
    if "project" in imports:
        importDir = config.datasetsImportDirectory
        projects = utils.listDirectoryFolders(importDir)
        projectCode = cy.IMPORT_PROJECT
        for project in projects:
            projectDir = join(importDir, project)
            for code in projectCode:
                for statement in code.replace("IMPORTDIR",projectDir).replace('PROJECTID', project).split(';')[0:-1]:
                    print statement
                    driver.run(statement+';')
    #Datasets
    if "datasets" in imports:
        importDir = config.datasetsImportDirectory
        datasetsCode = cy.IMPORT_DATASETS
        projects = utils.listDirectoryFolders(importDir)
        for project in projects:
            projectDir = join(importDir, project)
            datasetTypes = utils.listDirectoryFolders(projectDir)
            for dtype in datasetTypes:
                datasetDir = join(projectDir, dtype)
                code = datasetsCode[dtype]
                for statement in code.replace("IMPORTDIR",datasetDir).replace('PROJECTID', project).split(';')[0:-1]:
                    print statement
                    driver.run(statement+';')

        


def updateDB(dataset):
    pass

def updateSimilarity(context, entity):
    pass

def acquireKnowledge(database = None):
    pass

def acquireOntologies(ontology = None):
    pass


if __name__ == "__main__":
    #removeRelationshipDB(entity1 = 'Protein', entity2 = 'Protein', relationship = "intact_INTERACTS_WITH")
    #removeRelationshipDB(entity1 = 'Protein', entity2 = 'Protein', relationship = "IntAct_INTERACTS_WITH")

    #createDB(imports=["ontologies","chromosomes", "genes", "transcripts", "proteins", "ppi", "drugs", "diseases","internal", "variants", "project", "datasets"])
    #createDB(imports=["ontologies","datasets"])
    createDB(imports=["drugs", "diseases", "variants", "project", "datasets"])
