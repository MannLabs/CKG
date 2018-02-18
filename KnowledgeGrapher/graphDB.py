import config
import cypher as cy
from py2neo import Graph

def connectToDB(host = "localhost", port= 7687, user="neo4j", password = "password"):
    driver = Graph(host=host, port = port, user = user, password = password)

    return driver

def createDB(imports = ["ontologies","proteins", "ppi"]):
    host = config.dbURL
    port = config.dbPort
    user = config.dbUser
    password = config.dbPassword

    entities = config.entities
    importDir = config.importDirectory
    driver = connectToDB(host, port, user, password)

    #Get the cypher queries to build the graph
    #Ontologies
    if "ontologies" in imports:
        ontologyDataImportCode = cy.IMPORT_ONTOLOGY_DATA
        for entity in entities:
            cypherCode = ontologyDataImportCode.replace("ENTITY", entity).replace("IMPORTDIR",importDir).split(';')[0:-1]
            for statement in cypherCode:
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
        ppiDataImportCode = cy.IMPORT_PPI_DATA
        for resource in config.PPI_resources:
            for statement in ppiDataImportCode.replace("IMPORTDIR",importDir).replace("RESOURCE",resource.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")

    #Diseases
    if "diseases" in imports:
        diseaseDataImportCode = cy.IMPORT_DISEASE_DATA
        for resource in config.disease_resources:
            for statement in diseaseDataImportCode.replace("IMPORTDIR",importDir).replace("RESOURCE",resource.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")

    #Drugs
    if "drugs" in imports:
        drugsDataImportCode = cy.IMPORT_DRUG_DATA
        for resource in config.drug_resources:
            for statement in drugsDataImportCode.replace("IMPORTDIR",importDir).replace("RESOURCE",resource.lower()).split(';')[0:-1]:
                print statement+";"
                driver.run(statement+";")

def updateDB(dataset):
    pass

def updateSimilarity(context, entity):
    pass

def acquireKnowledge(database = None):
    pass

def acquireOntologies(ontology = None):
    pass


if __name__ == "__main__":
    createDB(imports=["ontologies","chromosomes", "genes", "transcripts", "proteins", "ppi", "drugs", "diseases"])
