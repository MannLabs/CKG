queries = {
    "OVERVIEW":("Project details", 
            '''MATCH (p:Project) WHERE p.id="PROJECTID" RETURN p.name AS Name, p.id AS Identifier, p.description AS Description, p.responsible AS Responsible, p.status AS Status, p.participants AS Participants, p.types AS Datasets;'''),
    "NUMBER_SUBJECTS":("Number of Enrolled Subjects:",
            '''MATCH (p:Project{id:"PROJECTID"})-[:HAS_ENROLLED]-(s:Subject) RETURN COUNT(DISTINCT(s)) AS Number_of_Subjects;'''),
    "NUMBER_ANALYTICAL_SAMPLES":("Number of Analytical Samples:", 
            '''MATCH (p:Project{id:"PROJECTID"})-[*3]-(a:Analytical_sample) RETURN COUNT(DISTINCT(a)) AS Number_of_Analytical_Samples;'''),
    "NUMBER_PROTEINS_ANALYTICAL_SAMPLE":("Number of Proteins", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) WHERE project.id="PROJECTID" RETURN a.id AS name, a.id AS x,COUNT(protein.id) AS y, a.group AS group;'''),
    "NUMBER_PEPTIDES_ANALYTICAL_SAMPLE":("Number of Peptides", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PEPTIDE]->(peptide:Peptide) WHERE project.id="PROJECTID" RETURN a.id AS name, a.id AS x, a.group AS group,COUNT(DISTINCT(peptide.id)) AS y;'''),
    "NUMBER_MODIFIED_PROTEINS_ANALYTICAL_SAMPLE":("Number of Modified Proteins", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEINMODIFICATION]->(modifiedprotein:Modified_protein) WHERE project.id="P0000001" RETURN a.id AS name, a.id AS x,a.group AS group,COUNT(DISTINCT(modifiedprotein.id)) AS y;'''),
    "IDENTIFIED_PROTEINS_SAMPLE_GROUP":("Identified proteins per group", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) WHERE project.id="PROJECTID" RETURN a.id AS sample, protein.id AS protein, a.group AS group, toFloat(r.value) as LFQ_intensity;''')
}
