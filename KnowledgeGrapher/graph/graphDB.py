from os.path import join
from py2neo import Graph
import config
import cypher as cy
from KnowledgeGrapher import utils

def getGraphDatabaseConnectionConfiguration():
    host = config.dbURL
    port = config.dbPort
    user = config.dbUser
    password = config.dbPassword

    driver = connectToDB(host, port, user, password)

    return driver

def connectToDB(host="localhost", port=7687, user="neo4j", password="password"):
    driver = Graph(host=host, port=port, user=user, password=password)

    return driver

def removeRelationshipDB(entity1, entity2, relationship):
    driver = getGraphDatabaseConnectionConfiguration()

    countCy = cy.COUNT_RELATIONSHIPS
    deleteCy = cy.REMOVE_RELATIONSHIPS
    countst = countCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    deletest = deleteCy.replace('ENTITY1', entity1).replace('ENTITY2', entity2).replace('RELATIONSHIP', relationship)
    print(countst)
    print(driver.run(countst).data()[0])
    print("Removing %d entries in the database" % driver.run(countst).data()[0]['count'])
    driver.run(deletest)
    print("Existing entries after deletion: %d" % driver.run(countst).data()[0]['count'])

def createDB(imports=None):
    if imports is None:
        imports = config.graph
    
    driver = getGraphDatabaseConnectionConfiguration()
    for i in imports:
        importDir = config.databasesDirectory
        #Get the cypher queries to build the graph
        #Ontologies
        if "ontologies" == i:
            entities = config.ontology_entities
            importDir = config.ontologiesDirectory
            ontologyDataImportCode = cy.IMPORT_ONTOLOGY_DATA
            for entity in entities:
                cypherCode = ontologyDataImportCode.replace("ENTITY", entity).replace("IMPORTDIR", importDir).split(';')[0:-1]
                for statement in cypherCode:
                    print(statement)
                    driver.run(statement+";")
        #Databases
        #Chromosomes
        elif "chromosomes" == i:
            chromosomeDataImportCode = cy.IMPORT_CHROMOSOME_DATA
            for statement in chromosomeDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")
        #Genes
        elif "genes" == i:
            geneDataImportCode = cy.IMPORT_GENE_DATA
            for statement in geneDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")
        #Transcripts
        elif "transcripts" == i:
            transcriptDataImportCode = cy.IMPORT_TRANSCRIPT_DATA
            for statement in transcriptDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")
        #Proteins
        elif "proteins" == i:
            proteinDataImportCode = cy.IMPORT_PROTEIN_DATA
            for statement in proteinDataImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")
        #PPIs
        elif "ppi" == i:
            ppiDataImportCode = cy.IMPORT_CURATED_PPI_DATA
            for resource in config.PPI_resources:
                for statement in ppiDataImportCode.replace("IMPORTDIR", importDir).replace("RESOURCE", resource.upper()).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Diseases
        elif "diseases" == i:
            diseaseDataImportCode = cy.IMPORT_DISEASE_DATA
            for entity, resource in config.disease_resources:
                for statement in diseaseDataImportCode.replace("IMPORTDIR", importDir).replace("ENTITY", entity).replace("RESOURCE", resource.lower()).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Drugs
        elif "drugs" == i:
            drugsDataImportCode = cy.IMPORT_CURATED_DRUG_DATA
            for resource in config.drug_resources:
                for statement in drugsDataImportCode.replace("IMPORTDIR", importDir).replace("RESOURCE", resource.lower()).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Side effects
        elif "side effects" == i:
            sideEffectsDataImportCode = cy.IMPORT_DRUG_SIDE_EFFECTS
            for resource in config.side_effects_resources:
                for statement in sideEffectsDataImportCode.replace("IMPORTDIR", importDir).replace("RESOURCE", resource.lower()).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Pathway
        elif 'pathway' == i:
            pathwayImportCode = cy.IMPORT_PATHWAY_DATA
            for source in config.pathway_resources:
                for statement in pathwayImportCode.replace("IMPORTDIR", importDir).replace("SOURCE", source.lower()).split(';')[0:-1]:
                    print(statement+';')
                    driver.run(statement+';')
        #Metabolite
        elif 'metabolite' == i:
            metaboliteImportCode = cy.IMPORT_METABOLITE_DATA
            for source in config.metabolite_resources:
                for statement in metaboliteImportCode.replace("IMPORTDIR", importDir).replace("SOURCE", source.lower()).split(';')[0:-1]:
                    print(statement+';')
                    driver.run(statement+';')
        #GWAS
        elif "gwas" == i:
            code = cy.IMPORT_GWAS
            for statement in code.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+';')
                driver.run(statement+';')
        #Known variants
        elif "known_variants" == i:
            variantsImportCode = cy.IMPORT_KNOWN_VARIANT_DATA
            for statement in variantsImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")
        #Clinically_relevant_variants
        elif "clinical variants" == i:
            variantsImportCode = cy.IMPORT_CLINICALLY_RELEVANT_VARIANT_DATA
            for source in config.clinical_variant_resources:
                for statement in variantsImportCode.replace("IMPORTDIR", importDir).replace("SOURCE", source.lower()).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Internal
        elif "internal" == i:
            internalDataImportCode = cy.IMPORT_INTERNAL_DATA
            for (entity1, entity2) in config.internalEntities:
                for statement in internalDataImportCode.replace("IMPORTDIR", importDir).replace("ENTITY1", entity1).replace("ENTITY2", entity2).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Mentions
        elif "mentions" == i:
            publicationsImportCode = cy.CREATE_PUBLICATIONS
            for statement in publicationsImportCode.replace("IMPORTDIR", importDir).split(';')[0:-1]:
                print(statement+";")
                driver.run(statement+";")

            mentionsImportCode = cy.IMPORT_MENTIONS
            for entity in config.mentionEntities:
                for statement in mentionsImportCode.replace("IMPORTDIR", importDir).replace("ENTITY", entity).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")

         #Published in
        elif "published" == i:
            publicationImportCode = cy.IMPORT_PUBLISHED_IN
            for entity in config.publicationEntities:
                for statement in publicationImportCode.replace("IMPORTDIR", importDir).replace("ENTITY", entity).split(';')[0:-1]:
                    print(statement+";")
                    driver.run(statement+";")
        #Projects
        elif "project" == i:
            importDir = config.experimentsDirectory
            projects = utils.listDirectoryFolders(importDir)
            projectCode = cy.IMPORT_PROJECT
            for project in projects:
                projectDir = join(importDir, project)
                for code in projectCode:
                    for statement in code.replace("IMPORTDIR", projectDir).replace('PROJECTID', project).split(';')[0:-1]:
                        print(statement+';')
                        driver.run(statement+';')
        #Datasets
        elif "experiment" == i:
            importDir = config.experimentsDirectory
            datasetsCode = cy.IMPORT_DATASETS
            projects = utils.listDirectoryFolders(importDir)
            for project in projects:
                projectDir = join(importDir, project)
                datasetTypes = utils.listDirectoryFolders(projectDir)
                for dtype in datasetTypes:
                    datasetDir = join(projectDir, dtype)
                    code = datasetsCode[dtype]
                    for statement in code.replace("IMPORTDIR", datasetDir).replace('PROJECTID', project).split(';')[0:-1]:
                        print(statement+';')
                        driver.run(statement+';')

def updateDB(dataset):
    createDB(imports=[dataset])

def populateDB():
    imports = config.graph
    createDB(imports)
    archiveImportDirectory()
    
def archiveImportDirectory():
    dest_folder = config.archiveDirectory
    utils.checkDirectory(dest_folder)
    folder_to_backup = config.importDirectory
    date, time = utils.getCurrentTime()
    file_name = "{}_{}".format(date.replace('-', ''), time.replace(':', ''))

    utils.compress_directory(folder_to_backup, dest_folder, file_name)

if __name__ == "__main__":
    populateDB()
