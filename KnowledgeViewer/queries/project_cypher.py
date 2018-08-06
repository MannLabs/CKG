queries = {
    "OVERVIEW":("Project details", 
                    '''MATCH (p:Project) WHERE p.id="PROJECTID" RETURN p.name AS Name, p.id AS Identifier, p.description AS Description, p.responsible AS Responsible, p.status AS Status, p.participants AS Participants, p.types AS Datasets;'''),
    "NUMBER_SUBJECTS":("Number of Enrolled Subjects:", '''MATCH (p:Project{id:"PROJECTID"})-[:HAS_ENROLLED]-(s:Subject) RETURN COUNT(DISTINCT(s)) AS Number_of_Subjects;'''),
    "NUMBER_ANALYTICAL_SAMPLES":("Number of Analytical Samples:", '''MATCH (p:Project{id:"PROJECTID"})-[*3]-(a:Analytical_sample) RETURN COUNT(DISTINCT(a)) AS Number_of_Analytical_Samples;''')
}
