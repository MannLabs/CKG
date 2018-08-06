queries = {
    "NUMBER_PROTEINS_ANALYTICAL_SAMPLE":("Number of Proteins", 
                    '''MATCH (p:Project) WHERE p.id="PROJECTID" RETURN p.name AS Name, p.id AS Identifier, p.description AS Description, p.responsible AS Responsible, p.status AS Status, p.participants AS Participants, p.types AS Datasets;'''),
    "NUMBER_PEPTIDES_ANALYTICAL_SAMPLE":("Number of Peptides", 
                                                '''MATCH (p:Project{id:"PROJECTID"})-[:HAS_ENROLLED]-(s:Subject) RETURN COUNT(DISTINCT(s)) AS Number_of_Subjects;'''),
    "NUMBER_MODIFIED_PROTEINS_ANALYTICAL_SAMPLE":("Number of Modified Proteins", 
                                                            '''MATCH (p:Project{id:"PROJECTID"})-[*3]-(a:Analytical_sample) RETURN COUNT(DISTINCT(a)) AS Number_of_Analytical_Samples;''')
}
